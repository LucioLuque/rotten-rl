"""
rotten_rl/jax_engine/tricks.py  —  Trick-taking phase of La Podrida in JAX (stage 1 prototype).

Same rules as `rotten_rl/engine.py` for the playing phase: follow the lead suit if you can; if you
cannot and you hold trumps you must trump (must_trump_when_void); otherwise play anything. The
highest trump wins the trick, else the highest card of the lead suit. The winner leads the next one.

Everything is arrays with fixed shapes and no Python control flow per move, so `jax.jit` compiles it
and `jax.vmap` runs thousands of games at once. One game's state:

    hands       bool[n, 52]   cards still held (card id = suit * 13 + rank, as in engine.py)
    trump       int           trump suit (0..3)
    current     int           player to move
    lead_suit   int           suit led in the current trick, -1 if nobody played yet
    trick       int[n]        card each player put in the current trick, -1 if not yet
    in_trick    int           cards in the current trick
    tricks_done int           completed tricks
    tricks_won  int[n]        tricks won by each player
    hand_size   int           cards dealt per player this round (the round ends after that many tricks)

Bidding, scoring, the round ladder and the observation are not here yet (stage 2).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

NUM_CARDS = 52
RANKS = 13
SUIT_OF = jnp.arange(NUM_CARDS) // RANKS          # suit of each card id
RANK_OF = jnp.arange(NUM_CARDS) % RANKS           # 0 = "2" ... 12 = "A"
SUIT_MASKS = jnp.stack([SUIT_OF == s for s in range(4)])   # bool[4, 52]


class TrickState(NamedTuple):
    hands: jnp.ndarray
    trump: jnp.ndarray
    current: jnp.ndarray
    lead_suit: jnp.ndarray
    trick: jnp.ndarray
    in_trick: jnp.ndarray
    tricks_done: jnp.ndarray
    tricks_won: jnp.ndarray
    hand_size: jnp.ndarray


def new_round(hands, trump, first_player, hand_size):
    """State at the start of the playing phase. `hands` bool[n, 52]."""
    n = hands.shape[0]
    return TrickState(
        hands=hands.astype(bool),
        trump=jnp.asarray(trump, jnp.int32),
        current=jnp.asarray(first_player, jnp.int32),
        lead_suit=jnp.asarray(-1, jnp.int32),
        trick=jnp.full((n,), -1, jnp.int32),
        in_trick=jnp.asarray(0, jnp.int32),
        tricks_done=jnp.asarray(0, jnp.int32),
        tricks_won=jnp.zeros((n,), jnp.int32),
        hand_size=jnp.asarray(hand_size, jnp.int32),
    )


def deal(key, num_players, hand_size, first_player=0):
    """Random deal: `hand_size` cards each and the next card turned up as trump."""
    order = jax.random.permutation(key, NUM_CARDS)
    owner = jnp.full((NUM_CARDS,), -1, jnp.int32)
    dealt = jnp.arange(NUM_CARDS)
    # Card in position i of the shuffled deck goes to player (first_player + i) % n while i < n*h.
    who = jnp.where(dealt < num_players * hand_size, (first_player + dealt) % num_players, -1)
    owner = owner.at[order].set(who)
    hands = owner[None, :] == jnp.arange(num_players)[:, None]
    trump = SUIT_OF[order[num_players * hand_size]]
    return new_round(hands, trump, first_player, hand_size)


def is_done(state):
    return state.tricks_done >= state.hand_size


def legal_mask(state):
    """bool[52]: cards the current player may play (all False once the round is over)."""
    hand = state.hands[state.current]
    leading = state.lead_suit < 0
    follow = hand & SUIT_MASKS[jnp.maximum(state.lead_suit, 0)]
    trumps = hand & SUIT_MASKS[state.trump]
    void_choice = jnp.where(trumps.any(), trumps, hand)
    mask = jnp.where(leading, hand, jnp.where(follow.any(), follow, void_choice))
    return mask & ~is_done(state)


def _trick_winner(trick, lead_suit, trump):
    suit = SUIT_OF[jnp.maximum(trick, 0)]
    rank = RANK_OF[jnp.maximum(trick, 0)]
    power = jnp.where(suit == trump, 200 + rank, jnp.where(suit == lead_suit, 100 + rank, 0))
    return jnp.argmax(jnp.where(trick >= 0, power, -1)).astype(jnp.int32)


def step(state, card):
    """Plays `card` for the current player. A no-op if the round is already over."""
    n = state.hands.shape[0]
    live = ~is_done(state)
    p = state.current
    hands = state.hands.at[p, card].set(jnp.where(live, False, state.hands[p, card]))
    lead_suit = jnp.where(state.lead_suit < 0, SUIT_OF[card], state.lead_suit)
    trick = state.trick.at[p].set(card)
    in_trick = state.in_trick + 1
    full = in_trick == n
    winner = _trick_winner(trick, lead_suit, state.trump)
    won = state.tricks_won.at[winner].add(full.astype(jnp.int32))
    new = TrickState(
        hands=hands,
        trump=state.trump,
        current=jnp.where(full, winner, (p + 1) % n),
        lead_suit=jnp.where(full, -1, lead_suit),
        trick=jnp.where(full, jnp.full((n,), -1, jnp.int32), trick),
        in_trick=jnp.where(full, 0, in_trick),
        tricks_done=state.tricks_done + full.astype(jnp.int32),
        tricks_won=won,
        hand_size=state.hand_size,
    )
    return jax.tree_util.tree_map(lambda a, b: jnp.where(live, a, b), new, state)


def random_card(key, state):
    """A uniformly random legal card (for benchmarks and tests)."""
    mask = legal_mask(state)
    logits = jnp.where(mask, 0.0, -jnp.inf)
    return jax.random.categorical(key, logits).astype(jnp.int32)


def play_random_rounds(key, num_games, num_players, hand_size):
    """Deals `num_games` rounds and plays them to the end with random legal cards, all at once.
    Returns tricks won, int[num_games, num_players]."""
    k_deal, k_play = jax.random.split(key)
    states = jax.vmap(lambda k: deal(k, num_players, hand_size))(jax.random.split(k_deal, num_games))

    def body(i, carry):
        states, key = carry
        key, sub = jax.random.split(key)
        cards = jax.vmap(random_card)(jax.random.split(sub, num_games), states)
        return jax.vmap(step)(states, cards), key

    states, _ = jax.lax.fori_loop(0, num_players * hand_size, body, (states, k_play))
    return states.tricks_won
