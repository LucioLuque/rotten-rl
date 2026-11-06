"""
rotten_rl/jax_engine/game.py  —  A whole match of La Podrida in JAX (stage 2).

The same game as `rotten_rl/openspiel_game.py` (pyspiel "podrida"): the 13-round ladder
1..7..1, the dealer rotating each round, bidding with the last-bidder restriction, tricks with the
follow-suit and must-trump rules, scoring (10 + 2*tricks if you make your bid, else tricks), the
per-round reward ("raw" or "relative", divided by 24) and the 808-value information state of each
player. Fixed shapes and no Python control flow per move: `jax.jit` + `jax.vmap` run thousands of
matches at once.

Actions are the pyspiel ones: 0..51 play that card (suit * 13 + rank), 52 + b bid b.

Deals: `new_game(orders)` takes, for each round, an order of the 52 cards. As in the pyspiel
wrapper, the first h go to player 0, the next h to player 1, ..., and the next one is the trump card.
`random_game(key, n)` draws them. Giving the orders makes a match reproducible and lets the tests
replay a pyspiel match exactly.

Not here: `infer_trump_void` (off by default; it did not help) and the hidden hands of the central
critic.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

MAX_PLAYERS = 7
NUM_ACTIONS = 60                                   # 52 cards + bids 0..7
BID_OFFSET = 52
LADDER = jnp.array([1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1], jnp.int32)
NUM_ROUNDS = 13
SCORE_CEILING = 228.0                              # sum of 10 + 2h over the ladder
ROUND_SCORE_NORM = 24.0
OBS_SIZE = 808

SUIT_OF = jnp.arange(52) // 13
RANK_OF = jnp.arange(52) % 13
SUIT_MASKS = jnp.stack([SUIT_OF == s for s in range(4)])          # bool[4, 52]

BIDDING, PLAYING, TERMINAL = 0, 1, 2


class GameState(NamedTuple):
    orders: jnp.ndarray       # int[13, 52]  card order of each round's deal
    round_ptr: jnp.ndarray    # int          round in the ladder (13 when the match is over)
    phase: jnp.ndarray        # int          BIDDING, PLAYING or TERMINAL
    dealer: jnp.ndarray       # int
    current: jnp.ndarray      # int          player to move
    hands: jnp.ndarray        # bool[n, 52]
    trump_card: jnp.ndarray   # int
    bids: jnp.ndarray         # int[n]       -1 = not bid yet
    tricks_won: jnp.ndarray   # int[n]
    scores: jnp.ndarray       # int[n]       accumulated over the match
    lead_suit: jnp.ndarray    # int          -1 = nobody played in this trick yet
    trick: jnp.ndarray        # int[n]       card each player put in the current trick, -1 if not
    trick_len: jnp.ndarray    # int
    tricks_done: jnp.ndarray  # int
    played_by: jnp.ndarray    # int[52]      who played each card this round, -1 if nobody
    voids: jnp.ndarray        # bool[n, 4]   known not to hold that suit (did not follow it)


def _round_start(orders, round_ptr, dealer, n):
    """Hands, trump card and first player of round `round_ptr` (dealer already rotated)."""
    h = LADDER[jnp.minimum(round_ptr, NUM_ROUNDS - 1)]
    order = orders[jnp.minimum(round_ptr, NUM_ROUNDS - 1)]
    pos = jnp.arange(52)
    owner = jnp.where(pos < n * h, pos // h, -1)             # position in the order -> player
    owner_of_card = jnp.full((52,), -1, jnp.int32).at[order].set(owner)
    hands = owner_of_card[None, :] == jnp.arange(n)[:, None]
    trump_card = order[n * h]
    return hands, trump_card, (dealer + 1) % n


def new_game(orders, num_players):
    """A match at its first decision (player after the dealer bids). `orders` int[13, 52]."""
    n = num_players
    orders = jnp.asarray(orders, jnp.int32)
    hands, trump_card, first = _round_start(orders, 0, 0, n)
    return GameState(
        orders=orders, round_ptr=jnp.int32(0), phase=jnp.int32(BIDDING), dealer=jnp.int32(0),
        current=jnp.int32(first), hands=hands, trump_card=trump_card.astype(jnp.int32),
        bids=jnp.full((n,), -1, jnp.int32), tricks_won=jnp.zeros((n,), jnp.int32),
        scores=jnp.zeros((n,), jnp.int32), lead_suit=jnp.int32(-1),
        trick=jnp.full((n,), -1, jnp.int32), trick_len=jnp.int32(0), tricks_done=jnp.int32(0),
        played_by=jnp.full((52,), -1, jnp.int32), voids=jnp.zeros((n, 4), bool))


def random_game(key, num_players):
    orders = jax.vmap(lambda k: jax.random.permutation(k, 52))(jax.random.split(key, NUM_ROUNDS))
    return new_game(orders, num_players)


def hand_size(state):
    return LADDER[jnp.minimum(state.round_ptr, NUM_ROUNDS - 1)]


def is_terminal(state):
    return state.phase == TERMINAL


def legal_mask(state):
    """bool[60]: legal actions of the current player (all False at the end)."""
    n = state.hands.shape[0]
    h = hand_size(state)
    # Bidding: 0..h, and the last bidder may not make the bids add up to h.
    bids_made = (state.bids >= 0).sum()
    forbidden = h - jnp.where(state.bids >= 0, state.bids, 0).sum()
    b = jnp.arange(8)
    bid_ok = (b <= h) & ~((bids_made == n - 1) & (b == forbidden))
    # Playing: follow suit; if void, trump if you can; else anything.
    hand = state.hands[state.current]
    follow = hand & SUIT_MASKS[jnp.maximum(state.lead_suit, 0)]
    trumps = hand & SUIT_MASKS[SUIT_OF[state.trump_card]]
    void_choice = jnp.where(trumps.any(), trumps, hand)
    card_ok = jnp.where(state.lead_suit < 0, hand,
                        jnp.where(follow.any(), follow, void_choice))
    bidding = state.phase == BIDDING
    playing = state.phase == PLAYING
    return jnp.concatenate([card_ok & playing, bid_ok & bidding])


def _winner(trick, lead_suit, trump_suit):
    """Who is winning a (possibly unfinished) trick: highest trump, else highest of the lead suit."""
    c = jnp.maximum(trick, 0)
    power = jnp.where(SUIT_OF[c] == trump_suit, 200 + RANK_OF[c],
                      jnp.where(SUIT_OF[c] == lead_suit, 100 + RANK_OF[c], 0))
    return jnp.argmax(jnp.where(trick >= 0, power, -1)).astype(jnp.int32)


def _round_rewards(points, n, scheme):
    points = points.astype(jnp.float32)
    if scheme == "relative":
        others = (points.sum() - points) / (n - 1)
        return (points - others) / ROUND_SCORE_NORM
    return points / ROUND_SCORE_NORM


def step(state, action, reward_scheme="relative"):
    """Applies `action` for the current player. Returns (state, rewards float[n]): the rewards are
    nonzero only on the move that closes a round. A no-op at the end of the match."""
    n = state.hands.shape[0]
    p = state.current
    h = hand_size(state)
    zeros = jnp.zeros((n,), jnp.float32)

    # --- bid ---
    bids = state.bids.at[p].set(action - BID_OFFSET)
    all_bid = (bids >= 0).sum() == n
    after_bid = state._replace(
        bids=bids,
        phase=jnp.where(all_bid, PLAYING, BIDDING).astype(jnp.int32),
        current=jnp.where(all_bid, (state.dealer + 1) % n, (p + 1) % n).astype(jnp.int32))

    # --- play a card ---
    card = jnp.clip(action, 0, 51)
    suit = SUIT_OF[card]
    leading = state.lead_suit < 0
    lead_suit = jnp.where(leading, suit, state.lead_suit)
    voids = state.voids.at[p, jnp.maximum(state.lead_suit, 0)].set(
        state.voids[p, jnp.maximum(state.lead_suit, 0)] | (~leading & (suit != state.lead_suit)))
    trick = state.trick.at[p].set(card)
    trick_len = state.trick_len + 1
    full = trick_len == n
    winner = _winner(trick, lead_suit, SUIT_OF[state.trump_card])
    tricks_won = state.tricks_won.at[winner].add(full.astype(jnp.int32))
    tricks_done = state.tricks_done + full.astype(jnp.int32)
    after_play = state._replace(
        hands=state.hands.at[p, card].set(False),
        lead_suit=jnp.where(full, -1, lead_suit).astype(jnp.int32),
        voids=voids,
        trick=jnp.where(full, -1, trick),
        trick_len=jnp.where(full, 0, trick_len).astype(jnp.int32),
        tricks_done=tricks_done,
        tricks_won=tricks_won,
        played_by=state.played_by.at[card].set(p),
        current=jnp.where(full, winner, (p + 1) % n).astype(jnp.int32))

    # --- the round closes: score, reward, next round (or the end) ---
    round_over = full & (tricks_done == h)
    made = tricks_won == state.bids
    points = jnp.where(made, 10 + 2 * tricks_won, tricks_won)
    rewards = _round_rewards(points, n, reward_scheme)
    next_ptr = state.round_ptr + 1
    next_dealer = (state.dealer + 1) % n
    hands, trump_card, first = _round_start(state.orders, next_ptr, next_dealer, n)
    over = next_ptr >= NUM_ROUNDS
    next_round = after_play._replace(
        round_ptr=next_ptr, dealer=next_dealer,
        phase=jnp.where(over, TERMINAL, BIDDING).astype(jnp.int32),
        current=jnp.where(over, 0, first).astype(jnp.int32),
        hands=hands, trump_card=trump_card.astype(jnp.int32),
        bids=jnp.full((n,), -1, jnp.int32), tricks_won=jnp.zeros((n,), jnp.int32),
        scores=state.scores + points, lead_suit=jnp.int32(-1),
        trick=jnp.full((n,), -1, jnp.int32), trick_len=jnp.int32(0), tricks_done=jnp.int32(0),
        played_by=jnp.full((52,), -1, jnp.int32), voids=jnp.zeros((n, 4), bool))

    def pick(cond, a, b):
        return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)

    played = pick(round_over, next_round, after_play)
    new = pick(state.phase == BIDDING, after_bid, played)
    new = pick(is_terminal(state), state, new)
    rewards = jnp.where((state.phase == PLAYING) & round_over, rewards, zeros)
    return new, rewards


def observation(state, player):
    """float32[808]: the information state of `player`, the same as
    `rotten_rl.openspiel_game.info_state_tensor` (zeros at the end of the match)."""
    n = state.hands.shape[0]
    h = hand_size(state)
    hf = h.astype(jnp.float32)
    rel_order = (player + jnp.arange(n)) % n          # relative position r -> absolute player
    pad = MAX_PLAYERS - n

    def per_rel(x):                                    # [n, ...] by absolute -> [7, ...] by relative
        x = x[rel_order]
        return jnp.concatenate([x, jnp.zeros((pad,) + x.shape[1:], x.dtype)])

    def one_hot(i, k, on=True):
        return jnp.where(on, jax.nn.one_hot(i, k, dtype=jnp.float32), 0.0)

    def rel(j):
        return (j - player) % n

    trick_on = state.trick_len > 0
    trump_suit = SUIT_OF[state.trump_card]
    bids_made = (state.bids >= 0).sum()
    played = state.played_by >= 0
    played_per_suit = (SUIT_MASKS & played[None, :]).sum(axis=1)
    my_hand = state.hands[player]

    loc = jnp.where(played, 3 + rel(state.played_by), 0)
    loc = loc.at[state.trump_card].set(2)
    loc = jnp.where(my_hand, 1, loc)

    seen = played_per_suit + (SUIT_MASKS & my_hand[None, :]).sum(axis=1)
    seen = seen.at[trump_suit].add(1)

    winner = _winner(state.trick, jnp.maximum(state.lead_suit, 0), trump_suit)
    bid_slot = state.bids + 1                           # -1 (not bid) -> 0, b -> b + 1
    trick_flags = jnp.zeros((52,), jnp.float32).at[jnp.maximum(state.trick, 0)].max(
        (state.trick >= 0).astype(jnp.float32))

    pieces = [
        one_hot(h - 1, 7),                                                   # hand_size
        one_hot(state.round_ptr, 13),                                        # round_index
        per_rel(state.hands.sum(axis=1).astype(jnp.float32) / hf),           # cards_remaining
        one_hot(state.phase, 2),                                             # phase
        one_hot(trump_suit, 5),                                              # trump
        one_hot(jnp.where(state.lead_suit < 0, 4, state.lead_suit), 5),      # lead_suit
        (jnp.arange(MAX_PLAYERS) < n).astype(jnp.float32),                   # active_players
        one_hot(rel(state.dealer), 7),                                       # dealer_relative
        jnp.where((state.phase == BIDDING) & (state.current == player) & (bids_made == n - 1),
                  1.0, 0.0)[None],                                           # is_last_bidder
        per_rel(jax.nn.one_hot(bid_slot, 9, dtype=jnp.float32)).reshape(-1),  # bids
        per_rel(state.tricks_won.astype(jnp.float32) / hf),                  # tricks_won
        per_rel(state.voids.astype(jnp.float32)).reshape(-1),                # void_suits
        per_rel(state.scores.astype(jnp.float32) / SCORE_CEILING),           # scores
        per_rel((jnp.maximum(state.bids, 0) - state.tricks_won).astype(jnp.float32) / hf),
        jax.nn.one_hot(loc, 10, dtype=jnp.float32).reshape(-1),              # card_info
        trick_flags,                                                         # trick_cards
        one_hot(rel(winner), 7, trick_on),                                   # winning_relative
        played_per_suit.astype(jnp.float32) / 13.0,                          # suits_played
        jax.nn.one_hot(jnp.minimum(seen, 12), 13, dtype=jnp.float32).reshape(-1),  # seen_per_suit
        one_hot(jnp.clip(n - 1 - state.trick_len, 0, 6), 7, state.phase == PLAYING),  # players_left
    ]
    obs = jnp.concatenate(pieces)
    return jnp.where(is_terminal(state), 0.0, obs)
