"""
scripts/jax_engine/tournament.py  —  The tournament of scripts/eval/tournament.py, on the GPU.

Every pair of agents (A in one seat, B in the others, and the other way round) plays a duplicate
evaluation on the same deals: each deal once with A in every seat (rotten_rl/jax_engine/evaluate.py,
`match`). All the matches of a pair are played at once on the device, so a pair takes about a
second instead of minutes on the CPU. Same tables, ranking and .md/.json as the pyspiel tournament
(rotten_rl/tournament_report.py), but the deals are drawn by JAX from --deal_seed, so the numbers
agree with the pyspiel tournament only within their standard errors.

Agents (comma separated):
  heuristic | random
  <run>             the last policy saved in <run>/checkpoints/
  <run>@<ep>        the one of that episode (policy_ep<ep>.pt)
  <file>.pt         a single policy (in <run>/checkpoints/)
  A pattern with * (e.g. ~/scratch/runs/o71_*) stands for every run it matches, sorted: the queue
  does not go through a shell, and the runs' folders carry their start time.
  With the suffix :sample, the network samples from its softmax instead of playing its most
  probable action (as PPO trains).
Networks of one MLP (PPO in JAX or PyTorch, DQN), with the default observation; the architecture
comes from the run's config.json.

With 2 players, "B against A" is the same games as "A against B" (each deal already puts A in both
seats), so it is not played: it is written with the sign changed, unless one of them is random.

Run it between trainings (a line of the queue, see scripts/run_queue.py): sharing the GPU with a
training only splits its time.

    python scripts/jax_engine/tournament.py --players 7 --deals 1000 \\
        --agents heuristic,~/scratch/runs/n70_<...>_s0_<ts>,~/scratch/runs/o71_<...>_s0_<ts> \\
        --out ~/scratch/torneo_7p.md
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax  # noqa: E402
import torch  # noqa: E402

from rotten_rl.jax_engine import evaluate as je  # noqa: E402
from rotten_rl.jax_engine.ppo import from_torch_state_dict  # noqa: E402
from rotten_rl.tournament_report import mirror, policy_file, report  # noqa: E402


def make_agent(spec, players):
    """(name, (kind, params)) for evaluate.match."""
    if spec in ("heuristic", "random"):
        return spec, (spec, None)
    sample = spec.endswith(":sample")
    spec = spec[:-len(":sample")] if sample else spec
    path, run_dir, name = policy_file(spec)
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    if config.get("algo") not in ("dqn", "ppo"):
        raise ValueError(f"{run_dir}: solo politicas de una red (dqn, ppo)")
    if config["players"] != players:
        raise ValueError(f"{run_dir} es de {config['players']} jugadores, no {players}")
    if config.get("infer_trump_void", False):
        raise ValueError(f"{run_dir}: la observacion con infer_trump_void no esta en JAX")
    params = from_torch_state_dict(torch.load(path, map_location="cpu"))
    return (name + ":sample" if sample else name), ("sample" if sample else "greedy", params)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--agents", required=True, help="agentes separados por coma (ver arriba)")
    p.add_argument("--players", type=int, required=True)
    p.add_argument("--deals", type=int, default=1000,
                   help="repartos por enfrentamiento (cada uno se juega una vez por asiento)")
    p.add_argument("--deal_seed", type=int, default=2026)
    p.add_argument("--seed", type=int, default=0, help="azar de random y de :sample")
    p.add_argument("--out", help="guardar el resultado en markdown (y .json al lado)")
    a = p.parse_args()

    n = a.players
    specs = []
    for s in a.agents.split(","):
        s = s.strip()
        if "*" in s:
            suffix = ":sample" if s.endswith(":sample") else ""
            found = sorted(glob.glob(os.path.expanduser(s[:len(s) - len(suffix)])))
            if not found:
                raise ValueError(f"{s}: no hay corridas")
            specs += [f + suffix for f in found]
        else:
            specs.append(s)
    agents = [make_agent(s, n) for s in specs]
    names = [name for name, _ in agents]
    if len(set(names)) != len(names):
        raise ValueError(f"Agentes repetidos: {names}")
    print(f"JAX {jax.__version__} on {jax.devices()}; {len(agents)} agentes, {n} jugadores, "
          f"{a.deals} repartos por enfrentamiento", flush=True)

    k = len(agents)
    deterministic = [kind in ("greedy", "heuristic") for _, (kind, _) in agents]
    pairs = [(i, j) for i in range(k) for j in range(k) if i != j and not (
        n == 2 and i > j and deterministic[i] and deterministic[j])]
    results = {}
    t0 = time.time()
    for i, j in pairs:
        t = time.time()
        key = jax.random.fold_in(jax.random.PRNGKey(a.seed), i * k + j)
        o = je.match(agents[i][1], agents[j][1], n, a.deals, a.deal_seed, key)["overall"]
        results[(i, j)] = o
        print(f"{names[i]} vs {names[j]}: diff {o['mean_score_diff']:+.2f} "
              f"±{o['mean_score_diff_se']:.2f}, win {o['win_rate']:.3f} "
              f"±{o['win_rate_se']:.3f} ({time.time() - t:.1f} s)", flush=True)
    for i in range(k):
        for j in range(k):
            if i != j and (i, j) not in results:
                results[(i, j)] = mirror(results[(j, i)])
    print(f"{len(pairs)} enfrentamientos jugados en {time.time() - t0:.0f} s", flush=True)
    report(names, results, n, a.deals, a.deal_seed, a.out, engine="jax")


if __name__ == "__main__":
    main()
