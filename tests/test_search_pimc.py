"""PIMC on the JAX engine (rotten_rl/search/pimc.py): the sampled deals agree with what the player
knows, the rollout plays the round to its end like a plain loop, and the values cover exactly the
legal actions."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optax")

import jax.numpy as jnp  # noqa: E402

from rotten_rl.jax_engine import game as jg  # noqa: E402
from rotten_rl.jax_engine import ppo  # noqa: E402
from rotten_rl.search import pimc  # noqa: E402


def _midround_state(n, round_ptr, plays, seed=0):
    """A random match advanced (random legal moves) to round `round_ptr`, then `plays` more moves."""
    rng = np.random.RandomState(seed)
    s = jg.random_game(jax.random.PRNGKey(seed), n)
    step = jax.jit(jg.step)
    while int(s.round_ptr) < round_ptr:
        s, _ = step(s, int(rng.choice(np.flatnonzero(np.asarray(jg.legal_mask(s))))))
    for _ in range(plays):
        s, _ = step(s, int(rng.choice(np.flatnonzero(np.asarray(jg.legal_mask(s))))))
    return s


@pytest.mark.parametrize("n,plays", [(2, 3), (3, 7), (4, 9)])
def test_sample_hands_agree_with_what_the_player_knows(n, plays):
    s = _midround_state(n, round_ptr=4, plays=plays, seed=n)    # hand of 5
    p = int(s.current)
    unseen = np.asarray(pimc._unseen(s, p))
    true = np.asarray(s.hands)
    for i in range(20):
        hands, ok = pimc.sample_hands(jax.random.PRNGKey(i), s, p)
        hands = np.asarray(hands)
        np.testing.assert_array_equal(hands[p], true[p])
        np.testing.assert_array_equal(hands.sum(axis=1), true.sum(axis=1))
        assert hands.sum(axis=0).max() <= 1                      # no card twice
        others = np.delete(hands, p, axis=0).any(axis=0)
        assert not (others & ~unseen).any()                      # only unseen cards are dealt
        if bool(ok):
            lacks = np.asarray(s.voids)[:, np.asarray(jg.SUIT_OF)]
            assert not (hands & lacks).any()


def test_rollout_round_matches_a_plain_loop():
    n = 3
    s = _midround_state(n, round_ptr=3, plays=4, seed=7)
    cfg = ppo.Config(num_players=n, hidden=(32,))
    params = ppo.init(jax.random.PRNGKey(1), cfg).policy
    got = np.asarray(pimc.rollout_round(params, s, jax.random.PRNGKey(0), sample=False))
    start, total = int(s.round_ptr), np.zeros(n)
    while int(s.round_ptr) == start:
        obs = jg.observation(s, s.current)
        logits = jnp.where(jg.legal_mask(s), ppo.mlp(params, obs[None, :])[0], ppo.ILLEGAL)
        s, r = jg.step(s, int(jnp.argmax(logits)))
        total += np.asarray(r)
    np.testing.assert_allclose(got, total, atol=1e-6)
    assert abs(total).sum() > 0


@pytest.mark.parametrize("n", [2, 3])
def test_action_values_cover_the_legal_actions(n):
    s = _midround_state(n, round_ptr=5, plays=n + 2, seed=3)    # hand of 6, playing
    cfg = ppo.Config(num_players=n, hidden=(32,))
    params = ppo.init(jax.random.PRNGKey(2), cfg).policy
    actions, values, valid, ok = jax.jit(
        lambda st, k: pimc.action_values(params, st, k, num_samples=8))(s, jax.random.PRNGKey(0))
    legal = np.flatnonzero(np.asarray(jg.legal_mask(s)))
    np.testing.assert_array_equal(np.asarray(actions)[np.asarray(valid)], legal)
    v = np.asarray(values)
    assert np.isfinite(v[np.asarray(valid)]).all() and np.isneginf(v[~np.asarray(valid)]).all()
    a = int(pimc.pimc_action(params, s, jax.random.PRNGKey(0), num_samples=8))
    assert a in legal
    assert 0.0 <= float(ok) <= 1.0


@pytest.mark.parametrize("n", [2, 3])
def test_bid_likelihood_with_the_true_hands_replays_the_bids(n):
    """Bids made by the network: with the true hands (mid-play), bid_log_likelihood equals the sum
    of the log-probabilities the network gave the others' bids when they made them."""
    cfg = ppo.Config(num_players=n, hidden=(32,))
    params = ppo.init(jax.random.PRNGKey(4), cfg).policy
    s = _midround_state(n, round_ptr=4, plays=0, seed=11)        # start of a round of 5, bidding
    rng = np.random.RandomState(0)
    logps = {}
    while int(s.phase) == jg.BIDDING:
        q = int(s.current)
        logits = jnp.where(jg.legal_mask(s), ppo.mlp(params, jg.observation(s, q)[None, :])[0],
                           ppo.ILLEGAL)
        lp = np.asarray(jax.nn.log_softmax(logits))
        a = int(rng.choice(np.flatnonzero(np.asarray(jg.legal_mask(s)))))
        logps[q] = lp[a]
        s, _ = jg.step(s, a)
    for _ in range(n + 1):                                       # a few cards into the play
        s, _ = jg.step(s, int(np.flatnonzero(np.asarray(jg.legal_mask(s)))[0]))
    p = int(s.current)
    got = float(pimc.bid_log_likelihood(params, s, s.hands, p))
    want = sum(v for q, v in logps.items() if q != p)
    assert abs(got - want) < 1e-4
