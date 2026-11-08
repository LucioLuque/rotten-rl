"""
rotten_rl/tournament_report.py  —  What both tournaments share (scripts/eval/tournament.py on
pyspiel, scripts/jax_engine/tournament.py on JAX): finding a run's policy file, the 2-player
mirror, and the markdown/JSON report.
"""

import glob
import json
import os
import re

import numpy as np


def policy_file(spec):
    """(.pt file, run directory, short name) of a network agent: <run>, <run>@<ep> or <file>.pt."""
    run_ep = re.fullmatch(r"(.+)@(\d+)", spec)
    if spec.endswith(".pt"):
        path = os.path.expanduser(spec)
        run_dir = os.path.dirname(os.path.dirname(path))
    else:
        run_dir = os.path.expanduser(run_ep.group(1) if run_ep else spec).rstrip("/")
        if run_ep:
            path = os.path.join(run_dir, "checkpoints", f"policy_ep{run_ep.group(2)}.pt")
        else:
            snaps = glob.glob(os.path.join(run_dir, "checkpoints", "policy_ep*.pt"))
            if not snaps:
                raise ValueError(f"No hay politicas guardadas en {run_dir}/checkpoints")
            path = max(snaps, key=lambda p: int(re.search(r"policy_ep(\d+)", p).group(1)))
    ep = int(re.search(r"policy_ep(\d+)", path).group(1))
    # Short name: the run's prefix (e.g. x21) and the episode (in thousands when round).
    short = os.path.basename(run_dir).split("_")[0]
    # Seeds trained together (train_ppo.py --seeds) share the prefix: add the seed (h20s1).
    try:
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)
        short += cfg.get("member_tag", "")                  # a sweep's value (p20lr1e-04)
        if len(cfg.get("seeds_together", [])) > 1:
            short += f"s{cfg['seed']}"
    except OSError:
        pass
    return path, run_dir, f"{short}@{ep // 1000}k" if ep and ep % 1000 == 0 else f"{short}@{ep}"


def mirror(o):
    """With 2 players, j against i: the same games as i against j seen from the other side."""
    return {"win_rate": 1.0 - o["win_rate"], "win_rate_se": o["win_rate_se"],
            "mean_score_diff": -o["mean_score_diff"],
            "mean_score_diff_se": o["mean_score_diff_se"], "mirrored": True}


def _table(names, cell):
    width = max(len(n) for n in names) + 2
    lines = ["| fila vs columna | " + " | ".join(names) + " |",
             "|---|" + "---|" * len(names)]
    for i, a in enumerate(names):
        lines.append(f"| {a:{width}s} | " + " | ".join(
            "—" if i == j else cell(i, j) for j in range(len(names))) + " |")
    return lines


def report(names, results, players, deals, deal_seed, out=None, engine="pyspiel"):
    """Markdown tables (score difference and win rate, ± standard error) and the ranking by
    average difference against the rest; printed, and saved to `out` (.md and .json) if given.
    `results[(i, j)]`: the `overall` metrics of agent i in one seat against j in the others."""
    n, k = players, len(names)
    lines = [f"# Torneo, {n} jugadores", "",
             f"{deals} repartos por enfrentamiento (semilla {deal_seed}, motor {engine}), cada uno "
             f"con el agente de la fila en cada asiento y el de la columna en los otros {n - 1}. "
             f"Empate: diferencia 0, win rate {1 / n:.3f}.", "",
             "## Diferencia de puntaje por partida (± error estándar)", ""]
    lines += _table(names, lambda i, j: "{:+.2f} ± {:.2f}".format(
        results[(i, j)]["mean_score_diff"], results[(i, j)]["mean_score_diff_se"]))
    lines += ["", "## Win rate (± error estándar)", ""]
    lines += _table(names, lambda i, j: "{:.3f} ± {:.3f}".format(
        results[(i, j)]["win_rate"], results[(i, j)]["win_rate_se"]))
    lines += ["", "## Ranking (diferencia promedio contra el resto)", ""]
    avg = {i: np.mean([results[(i, j)]["mean_score_diff"] for j in range(k) if j != i])
           for i in range(k)}
    for pos, i in enumerate(sorted(avg, key=avg.get, reverse=True), 1):
        lines.append(f"{pos}. {names[i]}: {avg[i]:+.2f}")
    text = "\n".join(lines) + "\n"
    print("\n" + text)
    if out:
        out = os.path.expanduser(out)
        with open(out, "w") as f:
            f.write(text)
        with open(os.path.splitext(out)[0] + ".json", "w") as f:
            json.dump({"players": n, "deals": deals, "deal_seed": deal_seed, "engine": engine,
                       "agents": names,
                       "results": {f"{names[i]} vs {names[j]}": o
                                   for (i, j), o in results.items()}}, f, indent=1)
    return text
