"""
scripts/run_queue.py  —  Cola de experimentos: corre hasta --parallel a la vez.

Cada linea no vacia del archivo --queue (las que empiezan con # se ignoran) es
un experimento:

    <nombre> <flags de train_nfsp.py>

p.ej.  e01_seed1 --players 2 --num_train_episodes 50000 --seed 1

Se lanza como
    OMP_NUM_THREADS=1 python scripts/train_nfsp.py --runs_dir <runs_dir> \
        --run_name <nombre> --stop_at <stop_at> <flags> > <logs_dir>/<nombre>.log

La cola se relee en cada vuelta: se pueden agregar lineas mientras corre. Los
nombres ya lanzados se anotan en <queue>.started, asi que reiniciar la cola no
repite experimentos. No lanza nada nuevo despues de --no_launch_after; hasta
esa hora, si la cola se vacia, espera lineas nuevas. Termina cuando no queda
nada corriendo y ya no se puede (o no hay que) lanzar mas.

Ejemplo (en el lab, dentro de tmux):
    python scripts/run_queue.py --queue ~/scratch/queue.txt --parallel 5 \
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
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--runs_dir", default=os.path.expanduser("~/scratch/runs"))
    p.add_argument("--logs_dir", default=os.path.expanduser("~/scratch/logs"))
    p.add_argument("--stop_at", default=None, help="HH:MM, se le pasa a cada corrida")
    p.add_argument("--no_launch_after", default=None, help="HH:MM, no lanzar nada nuevo despues")
    p.add_argument("--poll", type=int, default=30, help="segundos entre vueltas")
    a = p.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    started_path = a.queue + ".started"
    started = set()
    if os.path.exists(started_path):
        with open(started_path) as f:
            started = {line.strip() for line in f if line.strip()}
    os.makedirs(a.runs_dir, exist_ok=True)
    os.makedirs(a.logs_dir, exist_ok=True)
    stop_at, no_launch_after = _next_time(a.stop_at), _next_time(a.no_launch_after)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
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
        while can_launch and pending and len(running) < a.parallel:
            name, args = pending.pop(0)
            cmd = [sys.executable, os.path.join(repo, "scripts", "train_nfsp.py"),
                   "--runs_dir", a.runs_dir, "--run_name", name]
            if a.stop_at:
                cmd += ["--stop_at", a.stop_at]
            cmd += args
            with open(os.path.join(a.logs_dir, f"{name}.log"), "w") as out:
                running[name] = subprocess.Popen(cmd, cwd=repo, env=env, stdout=out,
                                                 stderr=subprocess.STDOUT)
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
