"""
rotten_rl/jax_engine/evaluate.py  —  The evaluation of rotten_rl/evaluation.py, on the device.

`evaluate(policy, opponent, ...)` plays a duplicate evaluation (as evaluation.evaluate with
`deal_seed`): num_games // n deals, each played n times with the evaluated policy in every seat and
the opponent in the others, all matches at once. The evaluated policy plays greedily (the most
probable legal action, like train.py's greedy_batch). Opponents:
  - "random":    a uniformly random legal action
  - "heuristic": `heuristic_action`, the same rules as evaluation.heuristic_opponent
  - "past":      another policy network, greedy (pass its params as `opponent_params`)

It returns the same dict as evaluation.evaluate (overall with standard errors over deals, by_h,
win_rate_by_seat, calibration, calibration_by_h, per_deal), so RunLogger.log_eval writes the same
CSVs. The deals are drawn by JAX from `deal_seed`, not by numpy as in evaluation.py: two agents
evaluated here with the same seed play the same cards, but the cards are not those of the pyspiel
evaluation, so the two agree only within their standard errors.
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import game as jg
from .ppo import ILLEGAL, match_length, mlp

# Expected tricks per card by number of players (evaluation._BID_WEIGHTS):
#        trump_high  trump_low    A     K     Q     J   others
BID_WEIGHTS = {
    2: (1.00, 0.95, 0.69, 0.58, 0.57, 0.50, 0.23),
    3: (1.00, 0.77, 0.56, 0.45, 0.37, 0.24, 0.06),
    4: (1.00, 0.64, 0.49, 0.31, 0.22, 0.16, 0.00),
    5: (0.99, 0.53, 0.38, 0.22, 0.15, 0.08, 0.00),
    6: (0.95, 0.47, 0.34, 0.14, 0.08, 0.03, 0.00),
    7: (0.92, 0.40, 0.27, 0.12, 0.05, 0.01, 0.00),
}
RANK_J, RANK_Q, RANK_K, RANK_A = 9, 10, 11, 12                 # rank index in engine.RANKS


def heuristic_action(state):
    """evaluation.heuristic_opponent for one state. Bid the rounded sum of the cards' expected
    tricks (the legal bid closest to it, the lower one on ties); when playing, lead the highest card
    if still short of the bid, else the lowest; follow with the cheapest card that wins the trick if
    short of the bid, else with the cheapest that does not (trumps cost more than any other card).
    Ties go to the lowest action id, as Python's min/max over the sorted legal actions."""
    n = state.hands.shape[0]
    mask = jg.legal_mask(state)
    p = state.current
    trump = jg.SUIT_OF[state.trump_card]
    is_trump = jg.SUIT_OF == trump
    rank = jg.RANK_OF

    w = jnp.asarray(BID_WEIGHTS[min(max(n, 2), 7)], jnp.float32)
    honor = jnp.select([rank == RANK_A, rank == RANK_K, rank == RANK_Q, rank == RANK_J],
                       [w[2], w[3], w[4], w[5]], w[6])
    card_w = jnp.where(is_trump, jnp.where(rank >= RANK_J, w[0], w[1]), honor)
    target = jnp.round(jnp.where(state.hands[p], card_w, 0.0).sum())    # half to even, like round()
    b = jnp.arange(8)
    bid_key = jnp.where(mask[jg.BID_OFFSET:], jnp.abs(b - target) * 16 + b, 1e9)
    bid = jg.BID_OFFSET + jnp.argmin(bid_key)

    card_mask = mask[:jg.BID_OFFSET]
    wants = state.tricks_won[p] < jnp.maximum(state.bids[p], 0)
    lead_high = jnp.argmax(jnp.where(card_mask, rank, -1))
    lead_low = jnp.argmin(jnp.where(card_mask, rank, 99))
    power = jnp.where(is_trump, 200 + rank, jnp.where(jg.SUIT_OF == state.lead_suit, 100 + rank, 0))
    best = state.trick[jg._winner(state.trick, state.lead_suit, trump)]
    beats = power > power[jnp.maximum(best, 0)]
    winners = card_mask & beats
    losers = card_mask & ~beats
    cost = is_trump * 16 + rank
    cheapest_winner = jnp.argmin(jnp.where(winners, cost, 99))
    pool = jnp.where(losers.any(), losers, card_mask)
    cheapest_other = jnp.argmin(jnp.where(pool, cost, 99))
    follow = jnp.where(wants & winners.any(), cheapest_winner, cheapest_other)
    play = jnp.where(state.lead_suit < 0, jnp.where(wants, lead_high, lead_low), follow)
    return jnp.where(state.phase == jg.BIDDING, bid, play).astype(jnp.int32)


def _greedy(params, obs, mask):
    return jnp.argmax(jnp.where(mask, mlp(params, obs), ILLEGAL), axis=-1).astype(jnp.int32)


@partial(jax.jit, static_argnames=("num_players", "num_deals", "opponent", "reward_scheme"))
def _play(policy, opponent_params, key, deal_key, num_players, num_deals, opponent, reward_scheme):
    """Plays every evaluation match. Returns, per match: final scores [G, n], the evaluated seat's
    return, and its bid and points in each round [13, G]."""
    n = num_players
    G = num_deals * n
    deal = jnp.arange(G) // n
    seat = jnp.arange(G) % n
    deal_keys = jax.random.split(deal_key, num_deals)[deal]
    states = jax.vmap(lambda k: jg.random_game(k, n))(deal_keys)
    step = jax.vmap(lambda s, a: jg.step(s, a, reward_scheme))
    idx = jnp.arange(G)

    def body(carry, _):
        states, key, ret = carry
        key, sub = jax.random.split(key)
        obs = jax.vmap(jg.observation)(states, states.current)
        mask = jax.vmap(jg.legal_mask)(states)
        mine = _greedy(policy, obs, mask)
        if opponent == "random":
            theirs = jax.random.categorical(sub, jnp.where(mask, 0.0, -jnp.inf)).astype(jnp.int32)
        elif opponent == "heuristic":
            theirs = jax.vmap(heuristic_action)(states)
        else:
            theirs = _greedy(opponent_params, obs, mask)
        action = jnp.where(states.current == seat, mine, theirs)
        bid = states.bids[idx, seat]
        new, rewards = step(states, action)
        points = new.scores[idx, seat] - states.scores[idx, seat]
        return (new, key, ret + rewards[idx, seat]), (bid, points)

    T = match_length(n)
    (states, _, ret), (bids, points) = jax.lax.scan(body, (states, key, jnp.zeros(G)), None, length=T)
    ends = np.cumsum([n + n * h for h in np.asarray(jg.LADDER)]) - 1       # moves that close a round
    return states.scores, ret, bids[ends], points[ends]


def evaluate(policy, opponent, num_players, num_games, deal_seed=1234, opponent_params=None,
             key=None, reward_scheme="relative"):
    """Duplicate evaluation of the greedy `policy` (JAX params) against `opponent` ("random",
    "heuristic" or "past" with `opponent_params`). Same result dict as evaluation.evaluate."""
    n = num_players
    num_deals = max(1, num_games // n)
    G = num_deals * n
    key = jax.random.PRNGKey(0) if key is None else key
    scores, ret, bids, points = jax.device_get(_play(
        policy, opponent_params if opponent == "past" else policy, key,
        jax.random.PRNGKey(deal_seed), n, num_deals, opponent, reward_scheme))
    return summarize(np.asarray(scores), np.asarray(ret), np.asarray(bids), np.asarray(points), n)


def summarize(scores, ret, bids, points, n):
    """The metrics of evaluation.evaluate from the played matches (match g: deal g // n, evaluated
    seat g % n; bids and points [13, G] of the evaluated seat)."""
    G = scores.shape[0]
    seat = np.arange(G) % n
    deal = np.arange(G) // n
    my = scores[np.arange(G), seat]
    top = scores.max(axis=1)
    n_top = (scores == top[:, None]).sum(axis=1)
    win = np.where(my == top, 1.0 / n_top, 0.0)
    diff = my - (scores.sum(axis=1) - my) / (n - 1)
    tricks = np.where(points >= 10, (points - 10) // 2, points)        # made: 10 + 2t, else t
    ladder = np.asarray(jg.LADDER)[:, None] * np.ones((1, G), int)

    def pack(sel):
        e = (bids - tricks)[sel]
        m = max(e.size, 1)
        return {"exact_bid_rate": float((e == 0).sum() / m), "mae_bid": float(np.abs(e).sum() / m),
                "bid_bias": float(e.sum() / m), "rounds": int(e.size)}

    def per_deal(x):
        return np.bincount(deal, weights=x) / n

    def se(x):
        d = per_deal(x)
        return float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")

    calib, calib_h = defaultdict(list), defaultdict(list)
    for h, b, t in zip(ladder.ravel(), bids.ravel(), tricks.ravel()):
        calib[int(b)].append(t)
        calib_h[(int(h), int(b))].append(t)
    return {
        "overall": {
            "win_rate": float(win.mean()), "win_rate_se": se(win),
            "mean_score_diff": float(diff.mean()), "mean_score_diff_se": se(diff),
            "mean_return": float(ret.mean()),
            "norm_round_score": float((points / (10 + 2 * ladder)).mean()),
            **pack(np.ones_like(bids, bool)),
        },
        "by_h": {int(h): pack(ladder == h) for h in sorted(set(ladder[:, 0].tolist()))},
        "win_rate_by_seat": {s: float(win[seat == s].mean()) for s in range(n)},
        "calibration": {b: float(np.mean(v)) for b, v in sorted(calib.items())},
        "calibration_by_h": {f"h{h}_b{b}": float(np.mean(v)) for (h, b), v in sorted(calib_h.items())},
        "per_deal": {"win": per_deal(win).tolist(), "diff": per_deal(diff).tolist()},
    }
