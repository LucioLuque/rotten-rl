"""
rotten_rl/exact/round1.py

La ronda de 1 carta, resuelta de forma (casi) exacta. Con una carta la jugada
es forzada, asi que la unica decision es el canto (0 o 1) y quien gana la baza
depende solo de las cartas. Cada situacion de canto queda definida por:

  - el numero de la carta de triunfo dada vuelta (13),
  - la carta propia: triunfo o no, y su numero (26 tipos; los tres palos que
    no son triunfo son intercambiables, asi que el palo no importa),
  - la posicion en el orden de canto (k = 0 es el primero, que tambien sale)
    y lo que cantaron los anteriores (2^k combinaciones).

Son 13 * 26 * (2^n - 1) situaciones (7 jugadores: ~41 mil). Las estrategias
son tablas por posicion con la probabilidad de cantar 1 en cada situacion.

El reward es el mismo que entrena el agente (`reward_scheme=relative`): el
puntaje de la ronda (10 + 2*bazas si cumple, si no bazas) menos el promedio de
los rivales. Los valores esperados se estiman con repartos al azar.

  - `solve(n)`: equilibrio con CFR+ (muestreando repartos en cada iteracion).
  - `evaluate(n, tablas, deals)`: valor de cada posicion y cuanto ganaria quien
    se desvie de su tabla (con toda la informacion, o mirando solo su carta).
  - `heuristic_tables(n)`: el canto de evaluation.heuristic_opponent.
  - `situation_state(game, ...)`: un estado de pyspiel en esa situacion, para
    preguntarle al agente que cantaria (`policy_tables`).
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from ..engine import RANKS, SUITS, Card

NUM_RANKS = 13
NUM_TYPES = 2 * NUM_RANKS          # 0..12: no triunfo de ese numero; 13..25: triunfo


def num_situations(n: int) -> int:
    return NUM_RANKS * NUM_TYPES * (2 ** n - 1)


def sample_deals(n: int, num: int, rng: np.random.RandomState) -> Dict[str, np.ndarray]:
    """`num` repartos al azar de una carta por jugador mas el triunfo. Las
    posiciones van en orden de canto (la 0 sale). Devuelve el numero del
    triunfo (num,), el tipo de carta de cada posicion (num, n) y quien gana la
    baza (num,)."""
    order = rng.random_sample((num, 52)).argsort(axis=1)[:, :n + 1]
    trump, cards = order[:, 0], order[:, 1:]
    suit, rank = cards // NUM_RANKS, cards % NUM_RANKS
    is_trump = suit == (trump // NUM_RANKS)[:, None]
    is_lead = suit == suit[:, :1]
    # Gana el triunfo mas alto; si no hay, la mas alta del palo de salida.
    key = np.where(is_trump, 200 + rank, np.where(is_lead, 100 + rank, -1))
    return {"trump_rank": trump % NUM_RANKS, "types": rank + NUM_RANKS * is_trump,
            "winner": key.argmax(axis=1), "trump": trump, "cards": cards}


def _forced(n: int, k: int) -> np.ndarray:
    """Para cada combinacion de cantos anteriores (bit j = canto de la posicion
    j): -1 si se elige, 0 o 1 si el canto es forzado. Solo el ultimo tiene la
    restriccion: la suma de cantos no puede dar 1 (las cartas de la ronda)."""
    out = np.full(2 ** k, -1)
    if k == n - 1:
        ones = np.array([bin(p).count("1") for p in range(2 ** k)])
        out[ones == 0] = 0                 # cantar 1 haria que sumen 1
        out[ones == 1] = 1                 # cantar 0 dejaria la suma en 1
    return out


def _utilities(n: int, winner: np.ndarray) -> np.ndarray:
    """(num, 2^n, n): reward relativo de cada jugador para cada combinacion de
    cantos (bit j = canto de la posicion j)."""
    bids = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1                   # (2^n, n)
    tricks = (winner[:, None] == np.arange(n)).astype(np.float32)[:, None, :]  # (num, 1, n)
    score = np.where(tricks == bids, 10 + 2 * tricks, tricks)
    total = score.sum(axis=2, keepdims=True)
    return (score - (total - score) / (n - 1)).astype(np.float32)


def _sweep(n: int, tables: List[np.ndarray], deals: Dict[str, np.ndarray]):
    """Recorre todas las combinaciones de cantos de cada reparto con las tablas
    dadas. Para cada posicion k devuelve (indices de situacion, alcance por los
    cantos de los demas, valor para k de cantar 0, de cantar 1, prob. de cantar 1
    usada, mascara de canto elegible), todos (num, 2^k)."""
    num = len(deals["winner"])
    reach, sig, idx = [np.ones((num, 1), np.float32)], [], []
    for k in range(n):
        p = np.arange(2 ** k)
        base = deals["trump_rank"] * NUM_TYPES + deals["types"][:, k]
        ix = base[:, None] * (2 ** k) + p[None, :]
        s = tables[k][ix].astype(np.float32)
        forced = _forced(n, k)
        s = np.where(forced < 0, s, forced.astype(np.float32))
        idx.append(ix)
        sig.append(s)
        reach.append(np.concatenate([reach[k] * (1 - s), reach[k] * s], axis=1))
    out = [None] * n
    value = _utilities(n, deals["winner"])
    for k in reversed(range(n)):
        half = 2 ** k
        v0, v1 = value[:, :half], value[:, half:]
        out[k] = (idx[k], reach[k], v0[:, :, k], v1[:, :, k], sig[k],
                  np.broadcast_to(_forced(n, k) < 0, (num, half)))
        s = sig[k][:, :, None]
        value = (1 - s) * v0 + s * v1
    return out


def uniform_tables(n: int) -> List[np.ndarray]:
    return [np.full(NUM_RANKS * NUM_TYPES * 2 ** k, 0.5) for k in range(n)]


def solve(n: int, iterations: int = 2000, deals_per_iteration: int = 20000,
          rng: np.random.RandomState = None, log: Callable[[int, List[np.ndarray]], None] = None):
    """Equilibrio aproximado con CFR+ muestreando repartos: en cada iteracion se
    sortean repartos nuevos, se recorren todas las combinaciones de cantos y se
    acumula el arrepentimiento de cada situacion (sin negativos). La estrategia
    promedio, pesada por iteracion, es la que converge. Devuelve las tablas
    promedio (probabilidad de cantar 1)."""
    rng = rng or np.random.RandomState(0)
    regret = [np.zeros((len(t), 2)) for t in uniform_tables(n)]
    avg = [np.zeros(len(t)) for t in regret]
    weight = 0.0
    for it in range(1, iterations + 1):
        current = []
        for r in regret:
            pos = np.maximum(r, 0)
            tot = pos.sum(axis=1)
            current.append(np.where(tot > 0, pos[:, 1] / np.where(tot > 0, tot, 1), 0.5))
        deals = sample_deals(n, deals_per_iteration, rng)
        for k, (ix, reach, v0, v1, s, free) in enumerate(_sweep(n, current, deals)):
            delta = (v1 - v0) * reach * free
            size = len(regret[k])
            regret[k][:, 1] += np.bincount(ix.ravel(), (delta * (1 - s)).ravel(), size)
            regret[k][:, 0] -= np.bincount(ix.ravel(), (delta * s).ravel(), size)
            np.maximum(regret[k], 0, out=regret[k])
        for a, c in zip(avg, current):
            a += it * c
        weight += it
        if log is not None:
            log(it, [a / weight for a in avg])
    return [a / weight for a in avg]


def evaluate(n: int, tables: List[np.ndarray], deals: Dict[str, np.ndarray]) -> Dict:
    """Con todos jugando sus tablas (tables[k] para la posicion k), por
    posicion: valor esperado, cuanto ganaria quien se desvie eligiendo lo
    mejor en cada situacion (`gain_best`, >= 0), y cuanto si solo puede mirar
    su carta y no los cantos anteriores (`gain_card_only`; puede ser negativo
    si las tablas ya aprovechan bien los cantos de los demas). La diferencia
    entre las dos es lo que vale escuchar a los demas. Todo en puntos por ronda."""
    num = len(deals["winner"])
    out = {"value": [], "gain_best": [], "gain_card_only": []}
    for k, (ix, reach, v0, v1, s, free) in enumerate(_sweep(n, tables, deals)):
        size = len(tables[k])
        q0 = np.bincount(ix.ravel(), (reach * v0).ravel(), size)
        q1 = np.bincount(ix.ravel(), (reach * v1).ravel(), size)
        cur = np.bincount(ix.ravel(), (reach * ((1 - s) * v0 + s * v1)).ravel(), size)
        # Elegible: hay al menos un camino en el que el canto no es forzado.
        free_ix = np.unique(ix[free])
        best = cur.copy()
        best[free_ix] = np.maximum(q0, q1)[free_ix]
        # Solo mirando la carta: un mismo canto para todos los cantos anteriores
        # (en los que puede elegir).
        prefix = 2 ** k
        fq0 = np.bincount(ix[free] // prefix, (reach * v0)[free], size // prefix)
        fq1 = np.bincount(ix[free] // prefix, (reach * v1)[free], size // prefix)
        fcur = np.bincount(ix[free] // prefix, (reach * ((1 - s) * v0 + s * v1))[free],
                           size // prefix)
        out["value"].append(cur.sum() / num)
        out["gain_best"].append((best.sum() - cur.sum()) / num)
        out["gain_card_only"].append((np.maximum(fq0, fq1).sum() - fcur.sum()) / num)
    return out


def heuristic_tables(n: int) -> List[np.ndarray]:
    """El canto de evaluation.heuristic_opponent con una carta: cantar 1 si el
    peso de la carta (bazas esperadas) redondea a 1."""
    from ..evaluation import _card_bid_weight
    bid1 = np.zeros((NUM_RANKS, NUM_TYPES))
    for t in range(NUM_TYPES):
        card = Card(RANKS[t % NUM_RANKS], SUITS[0] if t >= NUM_RANKS else SUITS[1])
        bid1[:, t] = round(_card_bid_weight(card, SUITS[0], n)) >= 1
    return [np.repeat(bid1.ravel(), 2 ** k) for k in range(n)]


def situation_state(game, k: int, trump_rank: int, card_type: int, prefix: int):
    """Un estado de pyspiel en la primera ronda, con el jugador en la posicion
    de canto k a punto de cantar, en esa situacion. Las cartas de los demas
    son cualquiera (no las ve). El primero en cantar es el siguiente al que
    reparte (el jugador 1 en la primera ronda)."""
    n = game.num_players()
    trump = trump_rank                                       # palo 0 = triunfo
    rank = card_type % NUM_RANKS
    mine = rank if card_type >= NUM_RANKS else NUM_RANKS + rank   # palo 1 si no es triunfo
    others = [c for c in range(2 * NUM_RANKS, 52)][:n - 1]
    me = (1 + k) % n
    hands = {}
    for p in range(n):
        hands[p] = mine if p == me else others.pop()
    state = game.new_initial_state()
    for p in range(n):
        state.apply_action(hands[p])
    state.apply_action(trump)
    for j in range(k):
        state.apply_action(52 + ((prefix >> j) & 1))
    assert state.current_player() == me
    return state


def policy_tables(game, prob_bid1: Callable[[list], np.ndarray]) -> List[np.ndarray]:
    """Las tablas de un agente: `prob_bid1(states)` da la probabilidad de
    cantar 1 en cada estado. Las situaciones imposibles (triunfo del mismo
    numero que el dado vuelta) quedan en 0."""
    n = game.num_players()
    tables = []
    for k in range(n):
        table = np.zeros(NUM_RANKS * NUM_TYPES * 2 ** k)
        states, where = [], []
        for tr in range(NUM_RANKS):
            for t in range(NUM_TYPES):
                if t == NUM_RANKS + tr:
                    continue
                for p in range(2 ** k):
                    if k == n - 1 and _forced(n, k)[p] >= 0:
                        continue
                    states.append(situation_state(game, k, tr, t, p))
                    where.append((tr * NUM_TYPES + t) * 2 ** k + p)
        if states:
            table[where] = prob_bid1(states)
        tables.append(table)
    return tables
