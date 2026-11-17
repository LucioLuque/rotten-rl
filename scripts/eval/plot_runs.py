"""
scripts/eval/plot_runs.py  —  Plots of training and evaluation metrics, several runs compared.

Each group is "label=pattern": the run folders the pattern matches (usually the 4 seeds of a run)
are averaged, with a band from the lowest to the highest seed. Metrics are "train:<column>" (from
train.csv: approx_kl, clip_frac, entropy, explained_variance, loss_value, learning_rate, ...,
smoothed) or "eval_<opponent>:<column>" (from eval_<opponent>_overall.csv: win_rate,
mean_score_diff, exact_bid_rate, ...). The x axis is games played, in millions; with --offset a
group's games are shifted (e.g. a run started from another one at 24M).

    python scripts/eval/plot_runs.py --out l30m.png \\
        --group "l30m=runs/lab/labo-s004-14/runs/l30m_*" --group "c30=runs/lab/labo-s004-14/runs/c30_*" \\
        --metrics train:approx_kl,train:entropy,eval_heuristic:win_rate
"""

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {
    "approx_kl": "KL aproximada (cuánto cambia la política)",
    "clip_frac": "fracción de pasos recortados",
    "entropy": "entropía (exploración)",
    "explained_variance": "varianza explicada por el crítico",
    "loss_value": "pérdida del crítico",
    "loss_policy": "pérdida de la política",
    "learning_rate": "tasa de aprendizaje",
    "grad_norm": "norma del gradiente",
    "loss_aux": "pérdida de la tarea auxiliar",
    "win_rate": "win rate",
    "mean_score_diff": "diferencia de puntos por partida",
    "exact_bid_rate": "cantos exactos",
}


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def series(run_dir, metric, smooth):
    src, col = metric.split(":")
    path = os.path.join(run_dir, "train.csv" if src == "train" else f"{src}_overall.csv")
    if not os.path.exists(path):
        return None
    rows = [r for r in read(path) if r.get(col) not in ("", None)]
    if not rows:
        return None
    x = np.array([float(r["episode"]) for r in rows]) / 1e6
    y = np.array([float(r[col]) for r in rows])
    if src == "train" and smooth > 1 and len(y) > smooth:
        k = np.ones(smooth) / smooth
        y = np.convolve(y, k, mode="valid")
        x = x[smooth - 1:]
    return x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", action="append", required=True, help="label=pattern of run folders")
    p.add_argument("--offset", action="append", default=[], help="label=millions to shift its x")
    p.add_argument("--metrics", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--smooth", type=int, default=50, help="moving average of train metrics (rows)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    offsets = {k: float(v) for k, v in (o.split("=") for o in a.offset)}
    metrics = a.metrics.split(",")
    cols = 2 if len(metrics) > 1 else 1
    rows_n = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(6.5 * cols, 3.6 * rows_n), squeeze=False)
    for g in a.group:
        label, pattern = g.split("=", 1)
        runs = sorted(glob.glob(os.path.expanduser(pattern)))
        for ax, metric in zip(axes.flat, metrics):
            curves = [s for s in (series(r, metric, a.smooth) for r in runs) if s is not None]
            if not curves:
                continue
            x = min((c[0] for c in curves), key=len)
            ys = np.stack([np.interp(x, c[0], c[1]) for c in curves])
            x = x + offsets.get(label, 0.0)
            line, = ax.plot(x, ys.mean(axis=0), label=f"{label} ({len(curves)})")
            if len(curves) > 1:
                ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), color=line.get_color(), alpha=0.15)
    for ax, metric in zip(axes.flat, metrics):
        src, col = metric.split(":")
        ax.set_title(f"{LABELS.get(col, col)}" + ("" if src == "train" else f" — contra {src[5:]}"),
                     fontsize=10)
        ax.set_xlabel("partidas (millones)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    if a.title:
        fig.suptitle(a.title)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(a.out)


if __name__ == "__main__":
    main()
