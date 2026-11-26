"""Match formats in the pyspiel "podrida" game (params "format" and "no_repeat_bid") must match the
JAX engine move by move (rotten_rl/jax_engine/game.py), including the extended observation the
networks trained with --formats see (openspiel_game.ext_info_state_tensor)."""

import random

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pyspiel = pytest.importorskip("pyspiel")
import jax.numpy as jnp  # noqa: E402

import rotten_rl.openspiel_game as og  # noqa: E402
from rotten_rl.jax_engine import game as jg  # noqa: E402

FORMATS = [
    ("1,2,3,4,5,6,7,7*,6,5,4,3,2,1,7,7*", True),
    ("1,2,3,4,5,6,7,7*,6*,5*,4*,3*,2*,1*", False),
    ("1*,2,3,4,5,6,7,7,6,5,4,3,2,1", True),
    ("", False),
]


def _play_pyspiel(n, fmt, no_repeat, seed):
    rng = random.Random(seed)
    game = pyspiel.load_game("podrida", {"players": n, "reward_scheme": "relative", "format": fmt,
                                         "no_repeat_bid": no_repeat})
    state = game.new_initial_state()
    orders, dealt, snaps = [], [], []
    while not state.is_terminal():
        if state.is_chance_node():
            card = rng.choice([c for c, _ in state.chance_outcomes()])
            dealt.append(card)
            state.apply_action(card)
            if not state.is_chance_node():
                orders.append(dealt + [c for c in range(52) if c not in dealt])
                dealt = []
            continue
        player = state.current_player()
        legal = state.legal_actions()
        obs = [og.ext_info_state_tensor(state, p) for p in range(n)]
        action = rng.choice(legal)
        state.apply_action(action)
        snaps.append((player, legal, obs, action, state.rewards()))
    return game, orders, snaps


@pytest.mark.parametrize("fmt, no_repeat", FORMATS)
@pytest.mark.parametrize("n", [2, 3, 7])
def test_formats_match_jax(n, fmt, no_repeat):
    step = jax.jit(lambda s, a: jg.step(s, a, "relative"))
    legal_mask = jax.jit(jg.legal_mask)
    observe = jax.jit(jax.vmap(lambda s, p: jg.observation(s, p, True, True), in_axes=(None, 0)))
    for seed in range(2):
        game, orders, snaps = _play_pyspiel(n, fmt, no_repeat, seed)
        st = game.new_initial_state()
        ladder, nt = st._ladder, st._no_trump
        h = np.zeros(jg.MAX_ROUNDS, np.int32)
        h[:len(ladder)] = ladder
        no_trump = np.zeros(jg.MAX_ROUNDS, bool)
        no_trump[:len(nt)] = nt
        state = jg.new_game(np.array(orders), n, no_repeat, h, no_trump)
        for i, (player, legal, obs, action, rewards) in enumerate(snaps):
            assert int(state.current) == player, f"move {i}"
            assert np.flatnonzero(np.asarray(legal_mask(state))).tolist() == legal, f"move {i}"
            np.testing.assert_allclose(np.asarray(observe(state, jnp.arange(n))), np.stack(obs),
                                       rtol=1e-6, atol=1e-6, err_msg=f"move {i}")
            state, r = step(state, jnp.int32(action))
            np.testing.assert_allclose(np.asarray(r), rewards, rtol=1e-6, atol=1e-6,
                                       err_msg=f"move {i}")
        assert bool(jg.is_terminal(state))


def test_no_repeat_rule_in_pyspiel():
    """With the rule, nobody bids the same 3 rounds in a row unless it is the only legal bid."""
    for seed in range(5):
        _, _, snaps = _play_pyspiel(3, "", True, seed)
        seqs = {}
        for player, legal, _, action, _ in snaps:
            if action >= og.BID_OFFSET:
                seq = seqs.setdefault(player, [])
                b = action - og.BID_OFFSET
                if len(seq) >= 2 and seq[-1] == seq[-2] == b:
                    assert legal == [action]
                seq.append(b)
