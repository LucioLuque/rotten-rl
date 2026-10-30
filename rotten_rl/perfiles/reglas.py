"""
rotten_rl/perfiles/reglas.py  —  Perfiles armados a mano.

Juegan las cartas como la heurística (buscan bazas hasta cumplir el canto) y cambian cómo cantan:
como la jugada depende del canto, cantar distinto ya da otra forma de jugar.
"""

import math

import numpy as np

from ..evaluation import _card_bid_weight, heuristic_opponent

_BID = 52


def _bid_profile(choose_bid):
    """Perfil que canta con `choose_bid(estimación de bazas, cantos legales)`."""
    def make(rng: np.random.RandomState):
        heuristic = heuristic_opponent(rng)

        def act(state, legal):
            eng = state._engine
            if eng.phase != "bidding":
                return heuristic(state, legal)
            hand = eng.players[eng.current_player_index].hand
            est = sum(_card_bid_weight(c, eng.trump_suit, eng.num_players) for c in hand)
            bids = [a - _BID for a in legal]
            return _BID + choose_bid(est, bids)
        return act
    return make


def _closest(target, bids):
    return min(bids, key=lambda b: (abs(b - target), b))


PROFILES = {
    "conservador": _bid_profile(lambda est, bids: _closest(math.floor(est * 0.7), bids)),
    "arriesgado": _bid_profile(lambda est, bids: _closest(math.ceil(est * 1.3), bids)),
    "a_perder": _bid_profile(lambda est, bids: min(bids)),
}
