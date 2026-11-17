"""
scripts/lab/csv_to_tensorboard.py  —  TensorBoard logs from the runs' CSVs (the backups leave out tb/).

For every run under runs/lab/<machine>/runs/ (JAX runs: their config.json says "engine": "jax"),
writes TensorBoard events to <out>/<machine>/<run>/ from train.csv (every column: KL, clip fraction,
entropy, explained variance, losses, learning rate, games/s...) and from eval_<opponent>_overall.csv
(win rate and score difference against each opponent). A run is redone only when one of its CSVs is
newer than its events. Meant to run every hour on the server, next to a TensorBoard on <out>.

    python scripts/lab/csv_to_tensorboard.py --runs runs/lab --out /srv/datos2/PodridaRL-tb
"""

import argparse
import csv
import glob
import json
import os
import shutil

from torch.utils.tensorboard import SummaryWriter

TRAIN_SKIP = {"episode"}
EVAL_KEYS = ("win_rate", "mean_score_diff", "exact_bid_rate", "bid_bias")


def newest(paths):
    return max((os.path.getmtime(p) for p in paths), default=0.0)


def convert(run_dir, out_dir, every):
    csvs = glob.glob(os.path.join(run_dir, "*.csv"))
    stamp = os.path.join(out_dir, ".done")
    if os.path.exists(stamp) and os.path.getmtime(stamp) >= newest(csvs):
        return False
    shutil.rmtree(out_dir, ignore_errors=True)
    w = SummaryWriter(out_dir)
    train = os.path.join(run_dir, "train.csv")
    if os.path.exists(train):
        with open(train) as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i % every:
                    continue
                step = int(float(row["episode"]))
                for k, v in row.items():
                    if k not in TRAIN_SKIP and v not in ("", None):
                        w.add_scalar(f"train/{k}", float(v), step)
    for path in sorted(glob.glob(os.path.join(run_dir, "eval_*_overall.csv"))):
        opp = os.path.basename(path)[len("eval_"):-len("_overall.csv")]
        with open(path) as f:
            for row in csv.DictReader(f):
                step = int(float(row["episode"]))
                for k in EVAL_KEYS:
                    if row.get(k) not in ("", None):
                        w.add_scalar(f"eval_{opp}/{k}", float(row[k]), step)
    w.close()
    open(stamp, "w").close()
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/lab")
    p.add_argument("--out", default="/srv/datos2/PodridaRL-tb")
    p.add_argument("--every", type=int, default=5, help="keep one train.csv row of every N")
    a = p.parse_args()
    done = 0
    for cfg in sorted(glob.glob(os.path.join(a.runs, "*", "runs", "*", "config.json"))):
        run_dir = os.path.dirname(cfg)
        try:
            with open(cfg) as f:
                if json.load(f).get("engine") != "jax":
                    continue
        except (OSError, ValueError):
            continue
        machine = run_dir.split(os.sep)[-3]
        done += convert(run_dir, os.path.join(a.out, machine, os.path.basename(run_dir)), a.every)
    print(f"{done} runs converted")


if __name__ == "__main__":
    main()
