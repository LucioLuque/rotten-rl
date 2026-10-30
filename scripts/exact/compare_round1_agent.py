"""
scripts/exact/compare_round1_agent.py  —  El canto del agente en la ronda de 1 carta contra el
equilibrio (docs/primera-ronda/).

Le pregunta a la red la probabilidad de cantar 1 en cada situacion (round1.policy_tables) y,
para cada posicion de canto, mide cuanto pierde jugando su tabla en esa posicion con los demas
en equilibrio, comparado con jugar el equilibrio; lo mismo para la heuristica como referencia.
Con la red muestreando (como entrena PPO) y jugando la accion mas probable (como se evalua).

Ejemplo (en el lab):
    python scripts/exact/compare_round1_agent.py --run ~/scratch/runs/z11_ppo3p_b128_s0_<ts>
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pyspiel
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import rotten_rl.openspiel_game  # noqa: E402,F401  registra el juego "podrida"
from rotten_rl.algorithms.dqn import mlp  # noqa: E402
from rotten_rl.exact import round1  # noqa: E402
from rotten_rl.openspiel_game import info_state_tensor  # noqa: E402

_BID0, _BID1 = 52, 53


def load_policy(spec):
    """(red, config, episodio) de <corrida> o <corrida>@<ep>."""
    m = re.fullmatch(r"(.+)@(\d+)", spec)
    run_dir = os.path.expanduser(m.group(1) if m else spec).rstrip("/")
    if m:
        path = os.path.join(run_dir, "checkpoints", f"policy_ep{m.group(2)}.pt")
    else:
        snaps = glob.glob(os.path.join(run_dir, "checkpoints", "policy_ep*.pt"))
        path = max(snaps, key=lambda p: int(re.search(r"policy_ep(\d+)", p).group(1)))
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    return path, config, int(re.search(r"policy_ep(\d+)", path).group(1))


def bid1_prob(net, states):
    """Probabilidad de cantar 1 (softmax sobre cantar 0 o 1)."""
    infos = np.stack([info_state_tensor(s, s.current_player()) for s in states])
    with torch.no_grad():
        logits = net(torch.as_tensor(infos))[:, [_BID0, _BID1]]
    return torch.softmax(logits, dim=1)[:, 1].numpy()


def loss_vs_eq(n, eq, tables, deals, value_eq):
    """Por posicion: valor de jugar `tables` ahi con los demas en equilibrio, menos el del
    equilibrio (negativo = pierde)."""
    out = []
    for k in range(n):
        mixed = list(eq)
        mixed[k] = tables[k]
        out.append(round1.evaluate(n, mixed, deals)["value"][k] - value_eq[k])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="<corrida> o <corrida>@<ep>")
    p.add_argument("--deals", type=int, default=500000)
    p.add_argument("--equilibrium", default="docs/primera-ronda/equilibrio.npz")
    a = p.parse_args()

    path, config, ep = load_policy(a.run)
    n = config["players"]
    game = pyspiel.load_game("podrida", {"players": n,
                                         "infer_trump_void": config.get("infer_trump_void", False)})
    net = mlp(game.information_state_tensor_size(),
              [int(x) for x in config["hidden_layers_sizes"]], game.num_distinct_actions())
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()

    saved = np.load(a.equilibrium)
    eq = [saved[f"n{n}_pos{k}"].astype(np.float64) for k in range(n)]
    sample = round1.policy_tables(game, lambda states: bid1_prob(net, states))
    greedy = [(t > 0.5).astype(np.float64) for t in sample]
    heur = round1.heuristic_tables(n)

    deals = round1.sample_deals(n, a.deals, np.random.RandomState(2000 + n))
    value_eq = round1.evaluate(n, eq, deals)["value"]
    rows = [("Agente muestreando", loss_vs_eq(n, eq, sample, deals, value_eq)),
            ("Agente, acción más probable", loss_vs_eq(n, eq, greedy, deals, value_eq)),
            ("Heurística", loss_vs_eq(n, eq, heur, deals, value_eq))]
    gain = round1.evaluate(n, sample, deals)["gain_best"]

    print(f"{os.path.basename(os.path.dirname(os.path.dirname(path)))} @ {ep}, {n} jugadores, "
          f"{a.deals:,} repartos\n")
    print("Pérdida contra el equilibrio por posición de canto (puntos por ronda; los demás "
          "en equilibrio):\n")
    print("| | " + " | ".join(str(k + 1) for k in range(n)) + " | promedio |")
    print("|---|" + "---|" * (n + 1))
    for name, v in rows:
        print(f"| {name} | " + " | ".join(f"{x:+.3f}" for x in v) + f" | {np.mean(v):+.3f} |")
    print("\nGanancia por desviarse si todos cantan como el agente (muestreando): "
          + ", ".join(f"{x:.3f}" for x in gain))


if __name__ == "__main__":
    main()
