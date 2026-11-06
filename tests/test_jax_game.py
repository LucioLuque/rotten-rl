"""The JAX match (rotten_rl/jax_engine/game.py) must match the pyspiel "podrida" game move by move:
current player, legal actions, every player's information state and the rewards."""

import random

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pyspiel = pytest.importorskip("pyspiel")
import jax.numpy as jnp  # noqa: E402

import rotten_rl.openspiel_game as og  # noqa: E402
from rotten_rl.jax_engine import game as jg  # noqa: E402


def _play_pyspiel(num_players, scheme, seed):
    """Plays a random match; returns the deal orders and, for every decision, what JAX must match."""
    rng = random.Random(seed)
    game = pyspiel.load_game("podrida", {"players": num_players, "reward_scheme": scheme})
    state = game.new_initial_state()
    orders, dealt, snaps = [], [], []
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = [c for c, _ in state.chance_outcomes()]
            card = rng.choice(outcomes)
            dealt.append(card)
            state.apply_action(card)
            if not state.is_chance_node():                  # the deal of this round is complete
                rest = [c for c in range(52) if c not in dealt]
                orders.append(dealt + rest)
                dealt = []
            continue
        player = state.current_player()
        legal = state.legal_actions()
        obs = [og.info_state_tensor(state, p) for p in range(num_players)]
        action = rng.choice(legal)
        state.apply_action(action)
        snaps.append((player, legal, obs, action, state.rewards()))
    return orders, snaps, state.returns()


@pytest.mark.parametrize("num_players, scheme", [(2, "relative"), (3, "raw"), (4, "relative"),
                                                 (7, "relative")])
def test_matches_pyspiel_move_by_move(num_players, scheme):
    step = jax.jit(lambda s, a: jg.step(s, a, scheme))
    legal_mask = jax.jit(jg.legal_mask)
    observe = jax.jit(jax.vmap(jg.observation, in_axes=(None, 0)))
    players = jnp.arange(num_players)
    for seed in range(3):
        orders, snaps, returns = _play_pyspiel(num_players, scheme, seed)
        state = jg.new_game(np.array(orders), num_players)
        total = np.zeros(num_players)
        for i, (player, legal, obs, action, rewards) in enumerate(snaps):
            assert int(state.current) == player, f"move {i}"
            assert np.flatnonzero(np.asarray(legal_mask(state))).tolist() == legal, f"move {i}"
            np.testing.assert_allclose(np.asarray(observe(state, players)), np.stack(obs),
                                       rtol=1e-6, atol=1e-7, err_msg=f"move {i}")
            state, r = step(state, jnp.int32(action))
            np.testing.assert_allclose(np.asarray(r), rewards, rtol=1e-6, atol=1e-6,
                                       err_msg=f"move {i}")
            total += np.asarray(r)
        assert bool(jg.is_terminal(state))
        np.testing.assert_allclose(total, returns, rtol=1e-5, atol=1e-5)
        assert not np.asarray(legal_mask(state)).any()


def test_random_matches_run_to_the_end():
    n, games = 3, 64

    def play(key):
        state = jg.random_game(key, n)

        def body(i, carry):
            state, key, total = carry
            key, sub = jax.random.split(key)
            mask = jg.legal_mask(state)
            action = jax.random.categorical(sub, jnp.where(mask, 0.0, -jnp.inf)).astype(jnp.int32)
            state, r = jg.step(state, jnp.where(mask.any(), action, 0))
            return state, key, total + r

        max_len = sum(n + n * h for h in [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1])
        return jax.lax.fori_loop(0, max_len, body, (state, key, jnp.zeros(n)))

    states, _, totals = jax.jit(jax.vmap(play))(jax.random.split(jax.random.PRNGKey(0), games))
    assert np.asarray(jg.is_terminal(states)).all()
    np.testing.assert_allclose(np.asarray(totals).sum(axis=1), 0.0, atol=1e-4)   # relative: zero-sum
