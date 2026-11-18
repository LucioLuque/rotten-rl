"""
scripts/lab/remaining_queue.py  —  What is left of a queue after the lab closed: a new queue file.

Reads a run_queue file and, from the line named by --from on, prints the lines still to do:

  - training runs that never started: the line as it is;
  - training runs cut before their --games (the last row of some seed's train.csv is below it):
    "<name>_resume --resume <runs>/<name>_* --games N --eval_every E", which continues them exactly;
  - training runs every seed of which reached --games: left out;
  - --resume lines: kept as they are while some run they resume is below their --games;
  - --script lines (tournaments): left out if the queue log says they ended with code 0, kept
    otherwise.

Meant for the morning after a night queue, so nothing has to be edited by hand at night:

    python scripts/lab/remaining_queue.py --queue ~/scratch/queue26.txt --log ~/scratch/logs/queue26.log \\
        --from s30full_jaxppo3p_ent0.06_24M > ~/scratch/queue30.txt
"""

import argparse
import glob
import os
import re
import shlex


def read_queue(path):
    lines = []
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if raw and not raw.startswith("#"):
                name, _, rest = raw.partition(" ")
                lines.append((name, shlex.split(rest), raw))
    return lines


def flag(args, key, default=None):
    return args[args.index(key) + 1] if key in args else default


def last_episode(run_dir):
    path = os.path.join(run_dir, "train.csv")
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        rows = f.read().strip().splitlines()
    return int(float(rows[-1].split(",")[0])) if len(rows) > 1 else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", required=True)
    p.add_argument("--log", required=True, help="the run_queue log of that queue")
    p.add_argument("--runs", default="~/scratch/runs")
    p.add_argument("--from", dest="start", default=None,
                   help="name of the first line to consider (default: the first line)")
    a = p.parse_args()
    runs = os.path.expanduser(a.runs)
    with open(os.path.expanduser(a.log)) as f:
        log = f.read()
    lines = read_queue(os.path.expanduser(a.queue))
    names = [n for n, _, _ in lines]
    if a.start is not None and a.start not in names:
        raise SystemExit(f"{a.start} is not in {a.queue}")
    for name, args, raw in lines[names.index(a.start) if a.start else 0:]:
        if "--resume" in args:
            dirs = glob.glob(os.path.expanduser(flag(args, "--resume")))
            if not dirs or min(last_episode(d) for d in dirs) < int(flag(args, "--games")):
                print(raw)
            continue
        if "--script" in args:
            if f"terminó {name} (código 0)" not in log:
                print(raw)
            continue
        dirs = [d for d in glob.glob(os.path.join(runs, name + "_s*"))
                if re.fullmatch(re.escape(name) + r"_s\d+_\d{8}-\d{6}", os.path.basename(d))]
        if not dirs:
            print(raw)
            continue
        target = int(flag(args, "--games"))
        if min(last_episode(d) for d in dirs) < target:
            print(f"{name}_resume --resume {a.runs}/{name}_* --games {target} "
                  f"--eval_every {flag(args, '--eval_every', '1000000')}")


if __name__ == "__main__":
    main()
