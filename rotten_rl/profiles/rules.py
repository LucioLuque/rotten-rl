"""
rotten_rl/profiles/rules.py  —  Hand-written profiles.

They play cards like the heuristic (chasing tricks until the bid is met) and only change how they
bid: since play depends on the bid, bidding differently already gives another way of playing.
"""

import math

import numpy as np

from ..evaluation import _card_bid_weight, heuristic_opponent

_BID = 52


def _bid_profile(choose_bid):
    """Profile that bids with `choose_bid(estimated tricks, legal bids)`."""
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
    "conservative": _bid_profile(lambda est, bids: _closest(math.floor(est * 0.7), bids)),
    "risky": _bid_profile(lambda est, bids: _closest(math.ceil(est * 1.3), bids)),
    "play_to_lose": _bid_profile(lambda est, bids: min(bids)),
}
