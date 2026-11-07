"""
scripts/eval/tournament.py  —  Torneo entre agentes con evaluacion duplicada.

Enfrenta a cada par de agentes (A en un asiento, B en todos los demas, y al
reves) sobre los mismos repartos, con A en cada asiento de cada reparto (ver
`deal_seed` en rotten_rl/evaluation.py). Asi se cancelan la suerte de las cartas
y la del asiento, y se puede comparar directo dos variantes, una politica contra
sus versiones anteriores, o todo contra la heuristica.

Agentes (separados por coma):
  heuristic | random
  <corrida>             la ultima politica guardada en <corrida>/checkpoints/
  <corrida>@<ep>        la de ese episodio (policy_ep<ep>.pt)
  <archivo>.pt          una politica suelta (en <corrida>/checkpoints/)
  Con el sufijo :sample, la red muestrea de su softmax en vez de jugar la
  accion mas probable (como entrena PPO).
Solo politicas de una red (dqn y ppo); la arquitectura sale del config.json de
la corrida.

Salida: diferencia de puntaje por partida (y win rate) del agente de la fila,
sentado en un asiento, contra el de la columna en los demas, con su error
estandar; y un ranking por diferencia promedio contra el resto. Si dos agentes
juegan igual, la diferencia da exactamente 0.

Los enfrentamientos se reparten entre `--workers` procesos (dan lo mismo que de a uno: los
repartos salen de `deal_seed`). Con 2 jugadores, "B contra A" son las mismas partidas que "A contra
B" (cada reparto ya se juega con A en los dos asientos), así que no se juega: se anota con el signo
cambiado, salvo que uno de los dos tenga azar (random, :sample o un perfil).

Ejemplo (en el lab):
    python scripts/eval/tournament.py --players 3 --deals 1000 --device cuda \\
        --agents heuristic,~/scratch/runs/x21_ppo_3p_roll128_<ts>,~/scratch/runs/w10_ppo_3p_s0_<ts>@100000 \\
        --out ~/scratch/torneo_3p.md
"""

import argparse
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import pyspiel
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import rotten_rl.openspiel_game  # noqa: E402,F401  registra el juego "podrida"
from rotten_rl.algorithms.dqn import greedy_batch, mlp  # noqa: E402
from rotten_rl.evaluation import evaluate, heuristic_opponent, random_opponent  # noqa: E402
from rotten_rl.openspiel_game import info_state_tensor  # noqa: E402
from rotten_rl.profiles.rules import PROFILES  # noqa: E402
from rotten_rl.profiles.spots import LearnedProfile, load_answers  # noqa: E402
from rotten_rl.tournament_report import mirror, policy_file, report  # noqa: E402

_ILLEGAL = -1e9


_PEOPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "profiles")


def _network_agent(spec, game, device, rng):
    sample = spec.endswith(":sample")
    spec = spec[:-len(":sample")] if sample else spec
    path, run_dir, name = policy_file(spec)
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    if config.get("algo") not in ("dqn", "ppo"):
        raise ValueError(f"{run_dir}: solo politicas de una red (dqn, ppo)")
    if config["players"] != game.num_players():
        raise ValueError(f"{run_dir} es de {config['players']} jugadores, no {game.num_players()}")
    num_actions = game.num_distinct_actions()
    net = mlp(game.information_state_tensor_size(),
              [int(x) for x in config["hidden_layers_sizes"]], num_actions)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.to(device).eval()
    # Cada red juega con la observacion con la que se entreno.
    infer_trump_void = config.get("infer_trump_void", False)

    if not sample:
        def batch(states, legals):
            return greedy_batch(net, states, legals, num_actions, device, infer_trump_void)
    else:
        name += ":sample"
        gen = torch.Generator(device=device)
        gen.manual_seed(int(rng.randint(2 ** 31)))

        def batch(states, legals):
            infos = np.stack([info_state_tensor(s, s.current_player(), infer_trump_void)
                              for s in states])
            mask = np.zeros((len(states), num_actions), dtype=bool)
            for i, legal in enumerate(legals):
                mask[i, legal] = True
            with torch.no_grad():
                logits = net(torch.as_tensor(infos, device=device))
                logits = logits.masked_fill(~torch.as_tensor(mask, device=device), _ILLEGAL)
                probs = torch.softmax(logits, dim=1)
                return torch.multinomial(probs, 1, generator=gen)[:, 0].cpu().numpy()

    def act(state, legal):
        return int(batch([state], [legal])[0])
    act.batch = batch
    return name, act


def make_agent(spec, game, device, rng):
    if spec == "heuristic":
        return "heuristic", heuristic_opponent(rng)
    if spec == "random":
        return "random", random_opponent(rng)
    if spec in PROFILES:
        return spec, PROFILES[spec](rng)
    if os.sep not in spec and os.path.isdir(os.path.join(_PEOPLE_DIR, spec)):   # una persona
        return spec, LearnedProfile(load_answers(game, os.path.join(_PEOPLE_DIR, spec)), rng)
    return _network_agent(spec, game, device, rng)


# Filled by main() before forking the workers (they inherit the agents).
_GAME = _AGENTS = _ARGS = None


def _play_pair(pair):
    """One matchup: agent i in each seat of each deal, agent j in the others."""
    i, j = pair
    t = time.time()
    rng = np.random.RandomState([_ARGS.seed, i, j])
    o = evaluate(_GAME, _AGENTS[i][1], _AGENTS[j][1], _ARGS.players, _ARGS.deals * _ARGS.players,
                 rng, parallel=True, deal_seed=_ARGS.deal_seed)["overall"]
    return i, j, o, time.time() - t


def _deterministic(spec):
    spec = spec.strip()
    return spec == "heuristic" or not (spec == "random" or spec.endswith(":sample")
                                       or spec in PROFILES or os.sep not in spec
                                       and os.path.isdir(os.path.join(_PEOPLE_DIR, spec)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--agents", required=True, help="agentes separados por coma (ver arriba)")
    p.add_argument("--players", type=int, required=True)
    p.add_argument("--deals", type=int, default=500,
                   help="repartos por enfrentamiento (cada uno se juega una vez por asiento)")
    p.add_argument("--deal_seed", type=int, default=2026)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0, help="azar de random y de :sample")
    p.add_argument("--out", help="guardar el resultado en markdown (y .json al lado)")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                   help="procesos que juegan enfrentamientos a la vez")
    a = p.parse_args()

    game = pyspiel.load_game("podrida", {"players": a.players})
    n = a.players
    rng = np.random.RandomState(a.seed)
    agents = [make_agent(s.strip(), game, a.device, rng) for s in a.agents.split(",")]
    names = [name for name, _ in agents]
    if len(set(names)) != len(names):
        raise ValueError(f"Agentes repetidos: {names}")

    global _GAME, _AGENTS, _ARGS
    _GAME, _AGENTS, _ARGS = game, agents, a
    torch.set_num_threads(1)
    specs = [s.strip() for s in a.agents.split(",")]
    k = len(agents)
    skip_mirror = n == 2
    pairs = [(i, j) for i in range(k) for j in range(k) if i != j and not (
        skip_mirror and i > j and _deterministic(specs[i]) and _deterministic(specs[j]))]
    results = {}
    t0 = time.time()

    def show(i, j, o, dt):
        results[(i, j)] = o
        print(f"{names[i]} vs {names[j]}: diff {o['mean_score_diff']:+.2f} "
              f"±{o['mean_score_diff_se']:.2f}, win {o['win_rate']:.3f} "
              f"±{o['win_rate_se']:.3f} ({dt:.0f} s)", flush=True)

    if a.workers > 1 and a.device == "cpu":         # CUDA does not survive a fork
        with multiprocessing.get_context("fork").Pool(a.workers) as pool:
            for i, j, o, dt in pool.imap_unordered(_play_pair, pairs):
                show(i, j, o, dt)
    else:
        for pair in pairs:
            show(*_play_pair(pair))
    for i in range(k):
        for j in range(k):
            if i != j and (i, j) not in results:
                results[(i, j)] = mirror(results[(j, i)])
    print(f"{len(pairs)} enfrentamientos jugados en {time.time() - t0:.0f} s "
          f"({a.workers} procesos)", flush=True)

    report(names, results, n, a.deals, a.deal_seed, a.out)

if __name__ == "__main__":
    main()
