"""
scripts/eval/panel.py  —  An agent against the fixed panel of rivals (docs/panel/), on the GPU.

For each agent (a run, or a pattern with * for several seeds, averaged) and each rival of the panel
of that number of players (docs/panel/panel.json), a duplicate evaluation (rotten_rl/jax_engine/
evaluate.py, `match`): the agent in one seat, the rival in all the others, every deal once per seat.
The table gives, per rival, the agent's score difference per match and its win rate (with a tie =
1/n), and per agent the mean over the rivals but the heuristic (it would dominate the mean) and the worst rival. Rivals missing on this machine
are skipped (with a warning).

    python scripts/eval/panel.py --players 3 --agents ~/scratch/runs/m30_*,~/scratch/runs/s30full_* \\
        --formats --out ~/scratch/panel_3p.md

--formats: every deal draws its format (docs/rule_variants/); otherwise the 13-round ladder.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jax_engine"))

import jax  # noqa: E402

from rotten_rl.jax_engine import evaluate as je  # noqa: E402
from tournament import make_agent  # noqa: E402

PANEL = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "panel", "panel.json")


def resolve(spec, runs_dir):
    """The agent spec of a panel entry on this machine, or None if the run is not here."""
    if spec in ("heuristic", "random"):
        return spec
    pattern, _, ep = spec.partition("@")
    found = sorted(glob.glob(os.path.join(os.path.expanduser(runs_dir), pattern)))
    if not found:
        return None
    return found[0] + (f"@{ep}" if ep else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--players", type=int, required=True)
    p.add_argument("--agents", required=True,
                   help="comma separated; each a run, <run>@<ep> or a pattern with * (its runs are "
                        "averaged, e.g. the 4 seeds)")
    p.add_argument("--deals", type=int, default=1000)
    p.add_argument("--deal_seed", type=int, default=2027)
    p.add_argument("--formats", action="store_true")
    p.add_argument("--runs_dir", default="~/scratch/runs")
    p.add_argument("--panel", default=PANEL)
    p.add_argument("--out", help="markdown (and .json next to it)")
    a = p.parse_args()
    n = a.players
    with open(a.panel) as f:
        entries = json.load(f)[str(n)]
    rivals = []
    for e in entries:
        spec = resolve(e["spec"], a.runs_dir)
        if spec is None:
            print(f"(no esta {e['name']}: {e['spec']})", flush=True)
            continue
        rivals.append((e["name"], make_agent(spec, n)[1]))
    groups = []
    for item in a.agents.split(","):
        item = item.strip()
        runs = sorted(glob.glob(os.path.expanduser(item))) if "*" in item else [item]
        if not runs:
            raise ValueError(f"{item}: no hay corridas")
        groups.append((os.path.basename(item.rstrip("/")), [make_agent(r, n)[1] for r in runs]))
    print(f"JAX {jax.__version__}; {len(groups)} agentes contra {len(rivals)} rivales, {n} jugadores, "
          f"{a.deals} repartos{' con formatos' if a.formats else ''}", flush=True)
    t0 = time.time()
    res = {}
    for gname, agents in groups:
        for rname, rival in rivals:
            outs = [je.match(ag, rival, n, a.deals, a.deal_seed, jax.random.PRNGKey(i),
                             formats=a.formats)["overall"] for i, ag in enumerate(agents)]
            diff = float(np.mean([o["mean_score_diff"] for o in outs]))
            se = float(np.sqrt(np.mean([o["mean_score_diff_se"] ** 2 for o in outs]) / len(outs)))
            win = float(np.mean([o["win_rate"] for o in outs]))
            res[(gname, rname)] = {"diff": diff, "diff_se": se, "win_rate": win, "runs": len(outs)}
            print(f"{gname} vs {rname}: {diff:+.2f} ±{se:.2f}, win {win:.3f}", flush=True)
    print(f"{len(res)} cruces en {time.time() - t0:.0f} s", flush=True)

    names = [r for r, _ in rivals]
    lines = [f"# Panel, {n} jugadores{' (formatos sorteados)' if a.formats else ' (escalera 1..7..1)'}",
             "", f"{a.deals} repartos por cruce (semilla {a.deal_seed}): el agente en un asiento y el "
             f"rival en los otros. Diferencia de puntaje por partida (win rate; empate {1 / n:.3f}).",
             "", "| Agente | " + " | ".join(names) + " | Promedio (sin heurística) | Peor |",
             "|---|" + "---|" * (len(names) + 2)]
    for gname, agents in groups:
        cells = [res[(gname, r)] for r in names]
        worst = min(range(len(names)), key=lambda i: cells[i]["diff"])
        lines.append(f"| {gname} ({len(agents)}) | " + " | ".join(
            f"{c['diff']:+.2f} ({c['win_rate']:.3f})" for c in cells)
            + f" | {np.mean([c['diff'] for r, c in zip(names, cells) if r != 'heuristica']):+.2f} | {names[worst]} "
              f"{cells[worst]['diff']:+.2f} |")
    text = "\n".join(lines) + "\n"
    print("\n" + text)
    if a.out:
        out = os.path.expanduser(a.out)
        with open(out, "w") as f:
            f.write(text)
        with open(os.path.splitext(out)[0] + ".json", "w") as f:
            json.dump({"players": n, "deals": a.deals, "deal_seed": a.deal_seed,
                       "formats": a.formats, "rivals": names,
                       "results": {f"{g} vs {r}": v for (g, r), v in res.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
