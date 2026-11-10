"""
scripts/match_value/fit_win_prob.py  —  How well do we predict who wins the match?

Plays self-play matches of a policy (every seat the same network, sampling as PPO trains), fits
on half of them and measures on the other half, after every round, how well each model predicts
the final winner:
  normal   rotten_rl/match_value/win_prob.py (scores + variance of the rounds left)
  table    measured P(I end on top | my lead over the best other, rounds played), normalized over
           the players; buckets of --bucket points
  uniform  1/n (the reference: what knowing nothing gives)
Metrics: log loss and Brier score over the players (lower is better), and a calibration table
(predicted probability against how often it happened). Writes markdown and the fitted per-round
variances as .json (the potential of the match reward uses them).

    python scripts/match_value/fit_win_prob.py --players 2 --policy ~/scratch/runs/s20_<...>_s0_<ts> \\
        --matches 200000 --out ~/scratch/win_prob_2p.md
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np  # noqa: E402

from rotten_rl.match_value import data, win_prob  # noqa: E402


def top_share(final):
    top = final == final.max(axis=-1, keepdims=True)
    return top / top.sum(axis=-1, keepdims=True)


def lead(scores):
    """Each player's score minus the best of the others: [..., n]."""
    n = scores.shape[-1]
    others = np.stack([np.delete(scores, i, axis=-1).max(axis=-1) for i in range(n)], axis=-1)
    return scores - others


class LeadTable:
    def __init__(self, cum, target, bucket, clip=80):
        self.bucket, self.clip = bucket, clip
        nb = 2 * clip // bucket + 1
        R = cum.shape[1]
        self.wins = np.ones((R, nb)) * 0.5                 # Laplace-like prior: half a win...
        self.count = np.ones((R, nb))                      # ...out of one match
        b = self._b(lead(cum))                             # [G, R, n]
        for r in range(R):
            np.add.at(self.wins[r], b[:, r].ravel(), target.ravel())
            np.add.at(self.count[r], b[:, r].ravel(), 1)

    def _b(self, x):
        return ((np.clip(x, -self.clip, self.clip) + self.clip) // self.bucket).astype(int)

    def __call__(self, cum):
        b = self._b(lead(cum))
        p = np.stack([(self.wins[r] / self.count[r])[b[:, r]] for r in range(cum.shape[1])], 1)
        return p / p.sum(axis=-1, keepdims=True)


def metrics(p, target):
    eps = 1e-6
    logloss = -(target * np.log(np.clip(p, eps, 1))).sum(-1).mean(0)       # per round
    brier = ((p - target) ** 2).sum(-1).mean(0)
    return logloss, brier


def calibration(p, target, bins=10):
    p, t = p.ravel(), np.broadcast_to(target, p.shape).ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for k in range(bins):
        sel = idx == k
        if sel.sum():
            rows.append((edges[k], edges[k + 1], p[sel].mean(), t[sel].mean(), int(sel.sum())))
    return rows


def load_params(spec, players):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jax_engine"))
    from tournament import make_agent
    _, (kind, params) = make_agent(spec, players)
    return params


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--players", type=int, required=True)
    p.add_argument("--policy", default="heuristic", help="run dir (or <run>@<ep>) or heuristic")
    p.add_argument("--matches", type=int, default=100000)
    p.add_argument("--bucket", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="markdown; the variances go to the same name .json")
    a = p.parse_args()
    n = a.players

    if a.policy == "heuristic":
        params, kind = None, "heuristic"
    else:
        params, kind = load_params(a.policy, n), "sample"
    pts = data.round_points(params, n, a.matches, seed=a.seed, kind=kind).astype(np.float64)
    cum = np.concatenate([np.zeros((len(pts), 1, n)), np.cumsum(pts, axis=1)], axis=1)  # [G,14,n]
    target = top_share(cum[:, -1])                                                       # [G, n]
    half = len(pts) // 2
    fit, test = slice(0, half), slice(half, None)

    round_var = win_prob.fit_round_variances(pts[fit])
    before = cum[test, :-1]                                  # scores before round r+1, r = 0..12
    rounds = np.arange(13)
    var = np.broadcast_to(np.asarray(win_prob.var_left(round_var, rounds)), before.shape[:2])
    preds = {
        "normal": np.asarray(win_prob.normal_win_prob(before, var)),
        "table": LeadTable(cum[fit, :-1], target[fit], a.bucket)(before),
        "uniform": np.full(before.shape, 1.0 / n),
    }
    t = target[test][:, None, :]

    lines = [f"# Probabilidad de ganar la partida, {n} jugadores", "",
             f"Política: `{a.policy}` ({kind}), {a.matches} partidas (mitad para ajustar, mitad "
             f"para medir), semilla {a.seed}.", "",
             "Varianza por ronda (σ², mitad de la varianza de la diferencia entre dos jugadores): "
             + ", ".join(f"{v:.1f}" for v in round_var), "",
             "## Log loss por ronda jugada (menor es mejor)", "",
             "| rondas jugadas | " + " | ".join(preds) + " |",
             "|---|" + "---|" * len(preds)]
    m = {k: metrics(v, t) for k, v in preds.items()}
    for r in rounds:
        lines.append(f"| {r} | " + " | ".join(f"{m[k][0][r]:.4f}" for k in preds) + " |")
    lines += ["", "## Brier por ronda jugada (menor es mejor)", "",
              "| rondas jugadas | " + " | ".join(preds) + " |", "|---|" + "---|" * len(preds)]
    for r in rounds:
        lines.append(f"| {r} | " + " | ".join(f"{m[k][1][r]:.4f}" for k in preds) + " |")
    for k in ("normal", "table"):
        lines += ["", f"## Calibración de {k} (rondas jugadas 1 a 12)", "",
                  "| predicho | promedio predicho | pasó | casos |", "|---|---|---|---|"]
        for lo, hi, pm, tm, c in calibration(preds[k][:, 1:], t[:, :, :].repeat(12, 1)):
            lines.append(f"| {lo:.1f}–{hi:.1f} | {pm:.3f} | {tm:.3f} | {c} |")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.expanduser(a.out), "w") as f:
        f.write(text)
    with open(os.path.splitext(os.path.expanduser(a.out))[0] + ".json", "w") as f:
        json.dump({"players": n, "policy": a.policy, "matches": a.matches,
                   "round_var": [float(v) for v in round_var]}, f, indent=1)


if __name__ == "__main__":
    main()
