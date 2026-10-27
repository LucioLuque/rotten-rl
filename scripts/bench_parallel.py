"""
scripts/bench_parallel.py  —  Cuantas corridas conviene correr a la vez en una maquina.

Para cada N de --ns lanza N entrenamientos identicos y cortos al mismo tiempo, y
mide la velocidad de cada uno (ms por episodio entre dos lineas del log, sin el
arranque ni la evaluacion inicial) y el total (episodios por segundo sumando las
N). El N optimo para experimentos es el de mayor total; para una corrida larga
importa ademas cuanto se frena cada una.

Correrlo con la maquina libre (si hay otros entrenamientos, la medicion no
sirve). Ejemplo, en el lab:
    python scripts/bench_parallel.py --ns 1,2,4,6,8,12 --episodes 2000
Con --gpu deja la GPU visible (para medir corridas con --device cuda, que la
comparten), p.ej.:
    python scripts/bench_parallel.py --gpu --players 3 --ns 1,2,4,8 --episodes 4000 \
        --extra "--algo ppo --vector_games 32 --device cuda --round_episodes"
"""

import argparse
import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile

_LINE = re.compile(r"^I\d{4} (\d\d:\d\d:\d\d\.\d+) .*\[ep (\d+)\] train")


def _times(log_path):
    out = {}
    with open(log_path) as f:
        for line in f:
            m = _LINE.match(line)
            if m:
                out[int(m.group(2))] = datetime.datetime.strptime(m.group(1), "%H:%M:%S.%f")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", default="1,2,4,6,8,12")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--from_episode", type=int, default=500,
                   help="medir desde este episodio (los primeros llenan buffers)")
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--extra", default="--rl_learning_rate 0.001 --sl_learning_rate 0.001",
                   help="flags extra para train.py")
    p.add_argument("--gpu", action="store_true",
                   help="dejar la GPU visible (por defecto se oculta: sin ella torch no "
                        "reserva memoria de video)")
    a = p.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if not a.gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    print(f"{'N':>3} | {'ms/ep (min/media/max)':>24} | {'ep/s total':>10}", flush=True)
    for n in [int(x) for x in a.ns.split(",")]:
        tmp = tempfile.mkdtemp(prefix=f"bench{n}_")
        procs, logs = [], []
        for i in range(n):
            log = os.path.join(tmp, f"run{i}.log")
            cmd = [sys.executable, os.path.join(repo, "scripts", "train.py"),
                   "--players", str(a.players), "--num_train_episodes", str(a.episodes),
                   "--eval_every", str(10 * a.episodes), "--eval_games", "10",
                   "--eval_opponents", "random", "--log_every", str(a.from_episode),
                   "--checkpoint_every", str(10 * a.episodes), "--nosave_eval_snapshots",
                   "--seed", str(i), "--runs_dir", tmp, "--run_name", f"run{i}",
                   *a.extra.split()]
            with open(log, "w") as out:
                procs.append(subprocess.Popen(cmd, cwd=repo, env=env, stdout=out,
                                              stderr=subprocess.STDOUT))
            logs.append(log)
        for proc in procs:
            proc.wait()
        ms = []
        for log in logs:
            t = _times(log)
            if a.from_episode in t and a.episodes in t:
                dt = (t[a.episodes] - t[a.from_episode]).total_seconds()
                ms.append(1000 * dt / (a.episodes - a.from_episode))
        if len(ms) != n:
            print(f"{n:>3} | fallaron {n - len(ms)} corridas (ver {tmp})", flush=True)
            continue
        total = sum(1000 / m for m in ms)
        print(f"{n:>3} | {min(ms):7.1f} {sum(ms) / n:7.1f} {max(ms):7.1f} | {total:10.1f}",
              flush=True)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
