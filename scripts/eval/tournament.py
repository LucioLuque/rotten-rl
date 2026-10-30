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

Ejemplo (en el lab):
    python scripts/eval/tournament.py --players 3 --deals 1000 --device cuda \\
        --agents heuristic,~/scratch/runs/x21_ppo_3p_roll128_<ts>,~/scratch/runs/w10_ppo_3p_s0_<ts>@100000 \\
        --out ~/scratch/torneo_3p.md
"""

import argparse
import glob
import json
import os
import re
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
from rotten_rl.perfiles.reglas import PROFILES  # noqa: E402
from rotten_rl.perfiles.situaciones import LearnedProfile, load_answers  # noqa: E402

_ILLEGAL = -1e9


def _policy_file(spec):
    """(archivo .pt, directorio de la corrida, nombre corto) de un agente de red."""
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
    # Nombre corto: el prefijo de la corrida (p.ej. x21) y el episodio (en miles si es redondo).
    short = os.path.basename(run_dir).split("_")[0]
    return path, run_dir, f"{short}@{ep // 1000}k" if ep and ep % 1000 == 0 else f"{short}@{ep}"


_PEOPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "perfiles")


def _network_agent(spec, game, device, rng):
    sample = spec.endswith(":sample")
    spec = spec[:-len(":sample")] if sample else spec
    path, run_dir, name = _policy_file(spec)
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
    if os.path.isdir(os.path.join(_PEOPLE_DIR, spec)):     # lo aprendido de una persona
        return spec, LearnedProfile(load_answers(game, os.path.join(_PEOPLE_DIR, spec)), rng)
    return _network_agent(spec, game, device, rng)


def _table(names, cell):
    width = max(len(n) for n in names) + 2
    lines = ["| fila vs columna | " + " | ".join(names) + " |",
             "|---|" + "---|" * len(names)]
    for i, a in enumerate(names):
        lines.append(f"| {a:{width}s} | " + " | ".join(
            "—" if i == j else cell(i, j) for j in range(len(names))) + " |")
    return lines


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
    a = p.parse_args()

    game = pyspiel.load_game("podrida", {"players": a.players})
    n = a.players
    rng = np.random.RandomState(a.seed)
    agents = [make_agent(s.strip(), game, a.device, rng) for s in a.agents.split(",")]
    names = [name for name, _ in agents]
    if len(set(names)) != len(names):
        raise ValueError(f"Agentes repetidos: {names}")

    results = {}
    for i, (name_a, act_a) in enumerate(agents):
        for j, (name_b, act_b) in enumerate(agents):
            if i == j:
                continue
            t = time.time()
            o = evaluate(game, act_a, act_b, n, a.deals * n, rng, parallel=True,
                         deal_seed=a.deal_seed)["overall"]
            results[(i, j)] = o
            print(f"{name_a} vs {name_b}: diff {o['mean_score_diff']:+.2f} "
                  f"±{o['mean_score_diff_se']:.2f}, win {o['win_rate']:.3f} "
                  f"±{o['win_rate_se']:.3f} ({time.time() - t:.0f} s)", flush=True)

    k = len(agents)
    lines = [f"# Torneo, {n} jugadores", "",
             f"{a.deals} repartos por enfrentamiento (semilla {a.deal_seed}), cada uno con el "
             f"agente de la fila en cada asiento y el de la columna en los otros {n - 1}. "
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
    if a.out:
        out = os.path.expanduser(a.out)
        with open(out, "w") as f:
            f.write(text)
        with open(os.path.splitext(out)[0] + ".json", "w") as f:
            json.dump({"players": n, "deals": a.deals, "deal_seed": a.deal_seed,
                       "agents": names,
                       "results": {f"{names[i]} vs {names[j]}": o
                                   for (i, j), o in results.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
