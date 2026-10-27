"""
scripts/run_queue.py  —  Cola de experimentos: corre hasta --parallel a la vez.

Cada linea no vacia del archivo --queue (las que empiezan con # se ignoran) es
un experimento:

    <nombre> <flags de train.py>

p.ej.  e01_seed1 --players 2 --num_train_episodes 50000 --seed 1

Se lanza como
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python scripts/train.py --runs_dir <runs_dir> \
        --run_name <nombre> --stop_at <stop_at> <flags> > <logs_dir>/<nombre>.log

La cola se relee en cada vuelta: se pueden agregar lineas mientras corre. Los
nombres ya lanzados se anotan en <queue>.started, asi que reiniciar la cola no
repite experimentos. No lanza nada nuevo despues de --no_launch_after; hasta
esa hora, si la cola se vacia, espera lineas nuevas. Termina cuando no queda
nada corriendo y ya no se puede (o no hay que) lanzar mas.

Con --max_total, antes de lanzar cuenta TODOS los entrenamientos que corren en
la maquina (de otras colas, la corrida larga, lanzados a mano) y solo lanza si
hay menos que ese limite; el limite se puede cambiar en caliente escribiendo un
numero en <queue>.max_total. Las corridas se lanzan en su propia sesion: si la
cola o su tmux mueren, siguen corriendo (y Ctrl+C en la cola no las corta).

Ejemplo (en el lab, dentro de tmux):
    python scripts/run_queue.py --queue ~/scratch/queue.txt --parallel 10 --max_total 11 \
        --stop_at 06:55 --no_launch_after 06:00
"""

import argparse
import datetime
import os
import shlex
import subprocess
import sys
import time


def _next_time(hhmm):
    if not hhmm:
        return None
    hh, mm = (int(x) for x in hhmm.split(":"))
    now = datetime.datetime.now()
    when = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return when if when > now else when + datetime.timedelta(days=1)


TRAINING_SCRIPTS = ("train.py",)


def count_training_processes(script_name=TRAINING_SCRIPTS):
    """Procesos de Python del usuario que estan corriendo `script_name` (un
    nombre o una tupla de nombres; por defecto train.py), lanzados por cualquier
    cola o a mano. Mira argv en /proc:
    cuenta solo si el ejecutable es python y el script es el primer argumento,
    asi no cuenta los `bash -lc "... python scripts/train.py ..."` que los
    envuelven."""
    names = (script_name,) if isinstance(script_name, str) else tuple(script_name)
    uid, n = os.getuid(), 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue
        if (len(argv) > 1 and b"python" in os.path.basename(argv[0])
                and os.path.basename(argv[1]).decode(errors="replace") in names):
            n += 1
    return n


def _read_max_total(queue_path, default):
    """El limite total, de <queue>.max_total si existe (se puede editar en caliente)."""
    try:
        with open(queue_path + ".max_total") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return default


def _read_queue(path):
    jobs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                name, *args = shlex.split(line)
                jobs.append((name, args))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", required=True)
    p.add_argument("--parallel", type=int, default=4,
                   help="maximo de corridas lanzadas por ESTA cola a la vez")
    p.add_argument("--max_total", type=int, default=None,
                   help="si se da: no lanzar si en la maquina ya corren esta cantidad de "
                        "entrenamientos (cuenta todos los del usuario, de cualquier cola o a mano). "
                        "Se puede cambiar en caliente escribiendo un numero en <queue>.max_total")
    p.add_argument("--script", default="scripts/train.py",
                   help="script de entrenamiento, relativo al repo")
    p.add_argument("--runs_dir", default=os.path.expanduser("~/scratch/runs"))
    p.add_argument("--logs_dir", default=os.path.expanduser("~/scratch/logs"))
    p.add_argument("--stop_at", default=None, help="HH:MM, se le pasa a cada corrida")
    p.add_argument("--no_launch_after", default=None, help="HH:MM, no lanzar nada nuevo despues")
    p.add_argument("--poll", type=int, default=30, help="segundos entre vueltas")
    a = p.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = tuple(set(TRAINING_SCRIPTS) | {os.path.basename(a.script)})
    started_path = a.queue + ".started"
    started = set()
    if os.path.exists(started_path):
        with open(started_path) as f:
            started = {line.strip() for line in f if line.strip()}
    os.makedirs(a.runs_dir, exist_ok=True)
    os.makedirs(a.logs_dir, exist_ok=True)
    stop_at, no_launch_after = _next_time(a.stop_at), _next_time(a.no_launch_after)
    # Las corridas son de CPU: sin GPU visible, torch no crea el contexto CUDA
    # (~386 MB de VRAM y ~90 MB de RAM por corrida; resultados identicos).
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    running = {}

    def log(msg):
        print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)

    while True:
        for name, proc in list(running.items()):
            if proc.poll() is not None:
                log(f"terminó {name} (código {proc.returncode})")
                del running[name]

        now = datetime.datetime.now()
        can_launch = no_launch_after is None or now < no_launch_after
        pending = [(n, args) for n, args in _read_queue(a.queue) if n not in started]
        max_total = _read_max_total(a.queue, a.max_total)

        def room():
            if len(running) >= a.parallel:
                return False
            return max_total is None or count_training_processes(script) < max_total

        while can_launch and pending and room():
            name, args = pending.pop(0)
            cmd = [sys.executable, os.path.join(repo, a.script),
                   "--runs_dir", a.runs_dir, "--run_name", name]
            if a.stop_at:
                cmd += ["--stop_at", a.stop_at]
            cmd += args
            with open(os.path.join(a.logs_dir, f"{name}.log"), "w") as out:
                # Sesion propia: si la cola (o su tmux) muere, las corridas siguen.
                running[name] = subprocess.Popen(cmd, cwd=repo, env=env, stdout=out,
                                                 stderr=subprocess.STDOUT,
                                                 start_new_session=True)
            started.add(name)
            with open(started_path, "a") as f:
                f.write(name + "\n")
            log(f"lancé {name}: {' '.join(args)}")

        # Con --no_launch_after, una cola vacia espera lineas nuevas hasta esa hora.
        if not running and (not can_launch or (not pending and no_launch_after is None)):
            log("no queda nada corriendo ni para lanzar: fin")
            return
        if stop_at is not None and now >= stop_at and not running:
            log("pasó --stop_at: fin")
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
