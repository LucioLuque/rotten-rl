"""
scripts/exact/compare_round2_agent.py  —  El agente en la ronda de 2 cartas contra la estrategia
calculada con MCCFR (docs/segunda-ronda/).

Para cada posicion de canto k y cada reparto: se juega la ronda 2 de una partida de verdad (la
ronda 1 la juega el agente en todos los asientos, asi su observacion es la de siempre) con la
posicion k jugando de una forma y las demas con la tabla (`round2 serve`). Se compara contra la
tabla jugando tambien en k, con el mismo reparto y los mismos sorteos para los demas: la
diferencia es cuanto pierde (o gana) esa forma de jugar en esa posicion contra el equilibrio.
Formas: el agente con la accion mas probable (como se evalua), el agente muestreando (como
entrena PPO) y la heuristica, como referencia.

Ademas, en las decisiones del agente (accion mas probable): con que frecuencia elige lo mismo que
la accion mas probable de la tabla, en el canto y en la jugada.

Ejemplo (en el lab):
    python scripts/exact/compare_round2_agent.py --run ~/scratch/runs/c10_ppo3p_<ts> \\
        --table ~/scratch/round2/n3_b.bin --deals 20000
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pyspiel
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import rotten_rl.openspiel_game  # noqa: E402,F401  registra el juego "podrida"
from rotten_rl.algorithms.dqn import mlp  # noqa: E402
from rotten_rl.evaluation import heuristic_opponent  # noqa: E402
from rotten_rl.exact import round2  # noqa: E402
from rotten_rl.openspiel_game import info_state_tensor  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from compare_round1_agent import load_policy  # noqa: E402

_BID = 52
_POLICIES = ("tabla", "agente", "agente_muestreando", "heuristica")


class Table:
    """La estrategia calculada, consultada a `round2 serve` (posiciones en orden de canto)."""

    def __init__(self, n, path, abstraction):
        self.n = n
        self.proc = subprocess.Popen(
            [round2.binary(), "serve", "--n", str(n), "--load", path, "--abstraction", abstraction],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lookups = self.misses = 0

    def _ask(self, line):
        self.proc.stdin.write(line + "\n")
        r = self.proc.stdout.readline().split()
        self.lookups += 1
        self.misses += r[0] == "0"
        return [(int(r[i]), float(r[i + 1])) for i in range(1, len(r), 2)]

    def bid(self, trump, hands, k, bids):
        cards = " ".join(f"{a} {b}" for a, b in hands)
        return self._ask(f"bid {trump} {cards} {k} " + " ".join(map(str, bids[:k])))

    def play(self, trump, hands, k, bids, plays):
        cards = " ".join(f"{a} {b}" for a, b in hands)
        return self._ask(f"play {trump} {cards} {k} " + " ".join(map(str, bids))
                         + " " + " ".join(map(str, plays[:k])))


def logits(net, state):
    x = torch.as_tensor(info_state_tensor(state, state.current_player()))[None]
    with torch.no_grad():
        return net(x)[0].numpy()


def round2_start(game, net, deal_rng, trump, hands):
    """Estado al empezar a cantar la ronda 2, con la ronda 1 jugada por el agente (accion mas
    probable) y el reparto de la ronda 2 dado: `hands[k]` es la mano de la posicion k."""
    s = game.new_initial_state()
    while s._round_ptr == 0:
        if s.is_chance_node():
            outcomes, probs = zip(*s.chance_outcomes())
            s.apply_action(int(deal_rng.choice(outcomes, p=probs)))
        else:
            legal = s.legal_actions()
            s.apply_action(legal[int(np.argmax(logits(net, s)[legal]))])
    n = game.num_players()
    first = (s._engine.dealer_index + 1) % n
    for p in range(n):
        for c in hands[(p - first) % n]:
            s.apply_action(c)
    s.apply_action(trump)
    assert s._engine.first_player_index == first and s._engine.phase == "bidding"
    return s, first


def play_round2(state, first, k, policy, table, net, trump, hands, rng, agent_rng, heur, stats):
    """Juega la ronda 2: la posicion k con `policy`, las demas con la tabla. Devuelve el puntaje
    de k menos el promedio de los demas (el reward relativo, sin normalizar)."""
    n = len(hands)
    bids, plays = [], []
    eng = state._engine
    while eng.phase != "done":
        p = state.current_player()
        pos = (p - first) % n
        legal = state.legal_actions()
        bidding = eng.phase == "bidding"
        if len(legal) == 1:
            a = legal[0]
        elif pos != k or policy == "tabla":
            opts = (table.bid(trump, hands, pos, bids) if bidding
                    else table.play(trump, hands, pos, bids, plays))
            u, acc = rng.uniform(), 0.0
            a = opts[-1][0]
            for act, pr in opts:
                acc += pr
                if acc > u:
                    a = act
                    break
            if bidding:
                a += _BID
        elif policy == "heuristica":
            a = heur(state, legal)
        else:
            lg = logits(net, state)[legal]
            if policy == "agente":
                a = legal[int(np.argmax(lg))]
                opts = (table.bid(trump, hands, pos, bids) if bidding
                        else table.play(trump, hands, pos, bids, plays))
                best = max(opts, key=lambda o: o[1])[0] + (_BID if bidding else 0)
                key = "canto" if bidding else "jugada"
                stats[key][0] += 1
                stats[key][1] += a == best
            else:
                pr = np.exp(lg - lg.max())
                a = legal[int(agent_rng.choice(len(legal), p=pr / pr.sum()))]
        state.apply_action(a)
        if bidding:
            bids.append(a - _BID)
        elif len(plays) < n:                   # las primeras n cartas son la primera baza
            plays.append(a)
    raw = [pl.last_round_score for pl in eng.players]
    me = raw[(first + k) % n]
    return me - (sum(raw) - me) / (n - 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="<corrida> o <corrida>@<ep>")
    p.add_argument("--table", required=True, help="tabla de `round2 solve` con la misma n")
    p.add_argument("--abstraction", default="b")
    p.add_argument("--deals", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    path, config, ep = load_policy(a.run)
    n = config["players"]
    game = pyspiel.load_game("podrida", {"players": n,
                                         "infer_trump_void": config.get("infer_trump_void", False)})
    assert game.ladder[1] == 2
    net = mlp(game.information_state_tensor_size(),
              [int(x) for x in config["hidden_layers_sizes"]], game.num_distinct_actions())
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    table = Table(n, os.path.expanduser(a.table), a.abstraction)
    heur = heuristic_opponent(np.random.RandomState(a.seed))

    vals = {pol: np.zeros((a.deals, n)) for pol in _POLICIES}
    stats = {"canto": [0, 0], "jugada": [0, 0]}
    t0 = time.time()
    for g in range(a.deals):
        rs = np.random.RandomState([a.seed, g])
        cards = rs.permutation(52)[:2 * n + 1]
        trump = int(cards[0])
        hands = [tuple(sorted(int(c) for c in cards[1 + 2 * j:3 + 2 * j])) for j in range(n)]
        start, first = round2_start(game, net, rs, trump, hands)
        for k in range(n):
            for pol in _POLICIES:
                vals[pol][g, k] = play_round2(
                    start.clone(), first, k, pol, table, net, trump, hands,
                    np.random.RandomState([a.seed, g, k, 1]),
                    np.random.RandomState([a.seed, g, k, 2]), heur, stats)
        if (g + 1) % 1000 == 0:
            print(f"  {g + 1} repartos, {time.time() - t0:.0f} s", file=sys.stderr)

    name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    print(f"{name} @ {ep}, {n} jugadores, {a.deals:,} repartos, tabla {a.table}\n")
    print("Valor de la tabla en cada posición, todos con la tabla (puntos por ronda): "
          + " / ".join(f"{x:+.2f}" for x in vals["tabla"].mean(0)) + "\n")
    print("Diferencia contra la tabla jugando en esa posición (los demás con la tabla; puntos "
          "por ronda, ± error estándar):\n")
    print("| | " + " | ".join(str(k + 1) for k in range(n)) + " | promedio |")
    print("|---|" + "---|" * (n + 1))
    labels = {"agente": "Agente, acción más probable", "agente_muestreando": "Agente muestreando",
              "heuristica": "Heurística"}
    for pol, label in labels.items():
        d = vals[pol] - vals["tabla"]
        cells = [f"{d[:, k].mean():+.3f} ± {d[:, k].std() / np.sqrt(a.deals):.3f}"
                 for k in range(n)]
        print(f"| {label} | " + " | ".join(cells) + f" | {d.mean():+.3f} |")
    print()
    for key, (tot, same) in stats.items():
        if tot:
            print(f"Agente (acción más probable): elige lo mismo que la tabla en el {100 * same / tot:.1f}% "
                  f"de sus decisiones de {key} con más de una opción ({tot:,}).")
    print(f"Situaciones que no están en la tabla: {100 * table.misses / max(1, table.lookups):.2f}%")


if __name__ == "__main__":
    main()
