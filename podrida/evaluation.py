"""
evaluation.py

Evaluador reusable para agentes de La Podrida sobre el wrapper de OpenSpiel.

No depende de torch: recibe una funcion `act_fn(state, legal) -> action_id`
para el asiento evaluado, y una `opponent_fn(state, legal) -> action_id` para
el resto. Asi sirve igual para evaluar NFSP, PSRO, una heuristica o random.

Metricas (todas del documento del proyecto), por partida y por ronda:
  - win_rate                : fraccion de partidas ganadas por el asiento (con
                              reparto justo de empates)
  - mean_score_diff         : E[S_i - mean(S_opp)]  sobre puntaje CRUDO de partida
  - mean_return             : retorno del wrapper (segun reward_scheme)
  - norm_round_score        : E[S_ronda / (10 + 2h)]
  - exact_bid_rate          : # rondas con tricks == bid / # rondas
  - mae_bid                 : E[|bid - tricks|]
  - bid_bias                : E[bid - tricks]  (con signo)
  - calibration[b]          : E[tricks | bid = b]
Todas las de nivel-ronda se reportan ademas DESGLOSADAS POR h.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict

import numpy as np

from podrida import RANKS, SUITS, Card


ActFn = Callable[[object, list], int]


# ----------------------------------------------------------------------
# Oponentes de referencia
# ----------------------------------------------------------------------

def random_opponent(rng: np.random.RandomState) -> ActFn:
    def act(state, legal):
        return int(rng.choice(legal))
    return act


def _beats(candidate: Card, best: Card, trump) -> bool:
    if trump is not None and candidate.suit == trump and best.suit != trump:
        return True
    if best.suit == trump and candidate.suit != trump:
        return False
    if candidate.suit == best.suit:
        return candidate.value > best.value
    return False


def _current_best(eng) -> Card | None:
    if not eng.current_trick:
        return None
    best = eng.current_trick[0][1]
    for _, card in eng.current_trick[1:]:
        if _beats(card, best, eng.trump_suit):
            best = card
    return best


def heuristic_opponent(rng: np.random.RandomState) -> ActFn:
    """Heuristica simple, la que sugiere el documento: cantar segun cartas altas,
    y jugar para ganar la baza si todavia falta cumplir el canto."""
    def act(state, legal):
        eng = state._engine
        idx = eng.current_player_index
        trump = eng.trump_suit

        if eng.phase == "bidding":
            hand = eng.players[idx].hand
            est = 0.0
            for c in hand:
                if c.suit == trump:
                    est += 0.85 if c.value >= 11 else 0.45
                elif c.rank == "A":
                    est += 0.80
                elif c.rank == "K":
                    est += 0.50
                elif c.rank == "Q":
                    est += 0.25
            target = int(round(est))
            bids = [a - 52 for a in legal]           # legal son ids 52+bid
            best_bid = min(bids, key=lambda b: (abs(b - target), b))
            return 52 + best_bid

        # fase de juego: legal son ids de carta
        cards = [(a, Card(RANKS[a % len(RANKS)], SUITS[a // len(RANKS)])) for a in legal]
        player = eng.players[idx]
        wants = player.tricks_won < (player.bid or 0)

        if not eng.current_trick:
            key = (lambda ac: ac[1].value)
            return (max(cards, key=key) if wants else min(cards, key=key))[0]

        best = _current_best(eng)
        winners = [(a, c) for a, c in cards if best is not None and _beats(c, best, trump)]
        if wants and winners:
            return min(winners, key=lambda ac: (ac[1].suit == trump, ac[1].value))[0]
        losers = [(a, c) for a, c in cards if (a, c) not in winners]
        pool = losers or cards
        return min(pool, key=lambda ac: (ac[1].suit == trump, ac[1].value))[0]

    return act


# ----------------------------------------------------------------------
# Evaluacion
# ----------------------------------------------------------------------

def evaluate(
    game,
    act_fn: ActFn,
    opponent_fn: ActFn,
    num_players: int,
    num_games: int,
    rng: np.random.RandomState,
    seat: int = 0,
) -> Dict:
    """Juega `num_games` partidas con `act_fn` en `seat` y `opponent_fn` en el
    resto. Devuelve un dict anidado con metricas overall, por-h y calibracion."""
    n = num_players

    wins = 0.0
    score_diff_sum = 0.0
    return_sum = 0.0
    norm_round_sum = 0.0
    norm_round_cnt = 0

    # acumuladores de nivel-ronda (overall y por h)
    def _acc():
        return {"n": 0, "exact": 0, "abs": 0.0, "signed": 0.0}
    overall = _acc()
    by_h: Dict[int, dict] = defaultdict(_acc)
    calib: Dict[int, list] = defaultdict(lambda: [0, 0])   # bid -> [sum_tricks, count]
    calib_by_h: Dict[tuple, list] = defaultdict(lambda: [0, 0])

    ladder = game.new_initial_state()._ladder  # escalera fija de esta config

    for _ in range(num_games):
        state = game.new_initial_state()
        prev_ptr = 0
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(outcomes, p=probs)))
                continue
            cur = state.current_player()
            legal = state.legal_actions(cur)
            action = act_fn(state, legal) if cur == seat else opponent_fn(state, legal)
            state.apply_action(action)

            if state._round_ptr != prev_ptr:
                h = ladder[prev_ptr]
                b = state._last_closed_bid[seat]
                t = state._last_closed_tricks[seat]
                e = abs(b - t)
                overall["n"] += 1
                overall["exact"] += int(e == 0)
                overall["abs"] += e
                overall["signed"] += (b - t)
                a = by_h[h]
                a["n"] += 1
                a["exact"] += int(e == 0)
                a["abs"] += e
                a["signed"] += (b - t)
                calib[b][0] += t
                calib[b][1] += 1
                calib_by_h[(h, b)][0] += t
                calib_by_h[(h, b)][1] += 1
                # puntaje de ronda normalizado por (10 + 2h)
                s_round = (10 + 2 * t) if t == b else t
                norm_round_sum += s_round / (10 + 2 * h)
                norm_round_cnt += 1
                prev_ptr = state._round_ptr

        # fin de partida: puntajes CRUDOS acumulados en el motor
        scores = [state._engine.players[i].score for i in range(n)]
        my = scores[seat]
        top = max(scores)
        if my == top:
            n_top = sum(1 for s in scores if s == top)
            wins += 1.0 / n_top                      # reparto justo de empates
        others = [scores[i] for i in range(n) if i != seat]
        score_diff_sum += my - (sum(others) / len(others))
        return_sum += state.returns()[seat]

    def _pack(acc):
        m = max(acc["n"], 1)
        return {
            "exact_bid_rate": acc["exact"] / m,
            "mae_bid": acc["abs"] / m,
            "bid_bias": acc["signed"] / m,
            "rounds": acc["n"],
        }

    result = {
        "overall": {
            "win_rate": wins / num_games,
            "mean_score_diff": score_diff_sum / num_games,
            "mean_return": return_sum / num_games,
            "norm_round_score": norm_round_sum / max(norm_round_cnt, 1),
            **_pack(overall),
        },
        "by_h": {h: _pack(acc) for h, acc in sorted(by_h.items())},
        "calibration": {
            b: (s / c) for b, (s, c) in sorted(calib.items()) if c > 0
        },
        "calibration_by_h": {
            f"h{h}_b{b}": (s / c) for (h, b), (s, c) in sorted(calib_by_h.items()) if c > 0
        },
    }
    return result
