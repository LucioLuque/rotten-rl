"""
scripts/summarize_runs.py  —  Tabla comparativa de corridas, desde sus CSV.

Para cada corrida en --runs_dir (o las que matcheen --filter) muestra el ultimo
episodio evaluado y, contra cada rival, el promedio de las ultimas --last
evaluaciones (promediar baja el ruido de evaluar con pocas partidas).

Ejemplo:
    python scripts/summarize_runs.py --runs_dir ~/scratch/runs --filter e0 --last 3
"""

import argparse
import csv
import glob
import os


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default=os.path.expanduser("~/scratch/runs"))
    p.add_argument("--filter", default="", help="solo corridas cuyo nombre contenga esto")
    p.add_argument("--last", type=int, default=3, help="evaluaciones a promediar")
    p.add_argument("--opponents", default="random,heuristic")
    a = p.parse_args()

    opps = a.opponents.split(",")
    header = f"{'corrida':34s} {'ep':>7s}"
    for o in opps:
        header += f" | {o[:5]:>5s} win  diff  exact  sesgo"
    print(header)
    print("-" * len(header))
    for run in sorted(glob.glob(os.path.join(a.runs_dir, f"*{a.filter}*"))):
        rows = {o: _rows(os.path.join(run, f"eval_{o}_overall.csv")) for o in opps}
        if not any(rows.values()):
            continue
        last_ep = max(int(r[-1]["episode"]) for r in rows.values() if r)
        line = f"{os.path.basename(run)[:34]:34s} {last_ep:7d}"
        for o in opps:
            tail = [r for r in rows[o] if r["episode"] > 0][-a.last:]
            if not tail:
                line += " |" + " " * 30
                continue
            m = {k: sum(r[k] for r in tail) / len(tail)
                 for k in ("win_rate", "mean_score_diff", "exact_bid_rate", "bid_bias")}
            line += (f" | {m['win_rate']:9.3f} {m['mean_score_diff']:+5.1f} "
                     f"{m['exact_bid_rate']:6.3f} {m['bid_bias']:+6.2f}")
        print(line)


if __name__ == "__main__":
    main()
