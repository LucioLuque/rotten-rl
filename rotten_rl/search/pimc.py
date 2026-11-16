"""
rotten_rl/search/pimc.py  —  PIMC (Perfect Information Monte Carlo) on the JAX engine.

At a decision of player p, PIMC does not know the other hands. It draws `num_samples` deals of the
cards p cannot see that agree with what p knows (its own hand, the cards already played, the trump
card, how many cards each player still holds and the suits each one is known to lack), and for each
deal and each legal action plays the rest of the ROUND with a policy network in every seat (all
players see the sampled deal through their own observation, as in a real match). The value of an
action is p's mean round reward over the samples; PIMC plays the best one.

Used as a stronger rival than the heuristic and to analyse the agent's decisions (how much it loses
against the best action PIMC finds). Known limits of PIMC: it ignores what the bids say about the
hands (no inference beyond the known voids), and inside each sampled deal the rollout policy plays
as if that deal were the real one ("strategy fusion").

Only legal actions are rolled out: at most MAX_LEGAL = 8 per decision (bids 0..7, or the cards of a
hand of at most 7).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rotten_rl.jax_engine import game as jg
from rotten_rl.jax_engine.ppo import ILLEGAL, mlp

MAX_LEGAL = 8
SAMPLE_TRIES = 32          # random deals tried per sample to respect the known voids


def _unseen(state, player):
    """bool[52]: cards `player` cannot see (in other hands): not in its hand, not played this
    round, not the trump card."""
    return (~state.hands[player] & (state.played_by < 0)
            & (jnp.arange(52) != state.trump_card))


def sample_hands(key, state, player):
    """A deal of the unseen cards to the other players: bool[n, 52], `player`'s hand unchanged.
    Each other player gets as many cards as it holds now; deals that give a player a suit it is
    known to lack are rejected (up to SAMPLE_TRIES; if none is valid, the last one is used).
    Returns (hands, valid)."""
    n = state.hands.shape[0]
    unseen = _unseen(state, player)
    counts = jnp.where(jnp.arange(n) == player, 0, state.hands.sum(axis=1))      # int[n]
    ends = jnp.cumsum(counts)
    starts = ends - counts

    def one(k):
        # Random order of the unseen cards first (the seen ones are pushed to the end).
        r = jax.random.uniform(k, (52,)) + jnp.where(unseen, 0.0, 2.0)
        order = jnp.argsort(r)                                   # unseen cards in random order
        pos = jnp.zeros((52,), jnp.int32).at[order].set(jnp.arange(52))  # card -> slot
        slot_of = (pos[:, None] >= starts[None, :]) & (pos[:, None] < ends[None, :])  # [52, n]
        dealt = unseen & slot_of.any(axis=1)
        q_of = jnp.argmax(slot_of, axis=1)                       # the q with starts[q] <= pos < ends[q]
        hands = (q_of[None, :] == jnp.arange(n)[:, None]) & dealt[None, :]
        lacks = state.voids[:, jg.SUIT_OF]                       # bool[n, 52]: q lacks that card's suit
        ok = ~(hands & lacks).any()
        return hands, ok

    hands_all, ok_all = jax.vmap(one)(jax.random.split(key, SAMPLE_TRIES))
    first = jnp.argmax(ok_all)                                   # first valid try (0 if none)
    others = hands_all[first]
    hands = others.at[player].set(state.hands[player])
    return hands, ok_all.any()


def _policy_action(params, state, key, sample, dt=None):
    obs = jg.observation(state, state.current)
    logits = jnp.where(jg.legal_mask(state), mlp(params, obs[None, :], dt)[0], ILLEGAL)
    return jnp.where(sample, jax.random.categorical(key, logits), jnp.argmax(logits)).astype(jnp.int32)


def rollout_round(params, state, key, sample=False, reward_scheme="relative"):
    """Plays from `state` to the end of its round with `params` in every seat. Returns the round
    rewards float[n] (zeros if the state is already terminal)."""
    n = state.hands.shape[0]
    start = state.round_ptr
    steps = n * (1 + 7)                                          # bids + plays of the largest round

    def body(carry, k):
        s, total, done = carry
        a = _policy_action(params, s, k, sample)
        s2, r = jg.step(s, a, reward_scheme)
        live = ~done
        s = jax.tree_util.tree_map(lambda x, y: jnp.where(live, y, x), s, s2)
        total = total + jnp.where(live, r, 0.0)
        done = done | (s.round_ptr != start) | jg.is_terminal(s)
        return (s, total, done), None

    init = (state, jnp.zeros((n,), jnp.float32), jg.is_terminal(state))
    (_, total, _), _ = jax.lax.scan(body, init, jax.random.split(key, steps))
    return total


def legal_actions(state):
    """(actions int[MAX_LEGAL], valid bool[MAX_LEGAL]): the legal actions of the current player,
    padded."""
    mask = jg.legal_mask(state)
    idx = jnp.nonzero(mask, size=MAX_LEGAL, fill_value=0)[0].astype(jnp.int32)
    valid = jnp.arange(MAX_LEGAL) < mask.sum()
    return idx, valid


def action_values(params, state, key, num_samples=32, sample_rollouts=False,
                  reward_scheme="relative"):
    """PIMC values of the current player's legal actions.
    Returns (actions int[8], values float[8] (-inf where not valid), valid bool[8], sampled_ok):
    values = mean round reward of the current player over `num_samples` sampled deals."""
    p = state.current
    actions, valid = legal_actions(state)
    k_deal, k_roll = jax.random.split(key)

    def per_sample(kd, kr):
        hands, ok = sample_hands(kd, state, p)
        s = state._replace(hands=hands)

        def per_action(a, k):
            s1, r = jg.step(s, a, reward_scheme)
            # The action may close the round itself (last card): then r already holds it.
            closed = s1.round_ptr != s.round_ptr
            rest = rollout_round(params, s1, k, sample_rollouts, reward_scheme)
            return jnp.where(closed, r, rest)[p]

        return jax.vmap(per_action)(actions, jax.random.split(kr, MAX_LEGAL)), ok

    vals, oks = jax.vmap(per_sample)(jax.random.split(k_deal, num_samples),
                                     jax.random.split(k_roll, num_samples))
    values = jnp.where(valid, vals.mean(axis=0), -jnp.inf)
    return actions, values, valid, oks.mean()


def pimc_action(params, state, key, num_samples=32, sample_rollouts=False):
    """The action PIMC plays: the legal one with the highest value."""
    actions, values, _, _ = action_values(params, state, key, num_samples, sample_rollouts)
    return actions[jnp.argmax(values)]
