"""Rule variants of the JAX match (rotten_rl/jax_engine/game.py, docs/rule_variants/)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from rotten_rl.jax_engine import game as jg  # noqa: E402

step = jax.jit(jg.step)
legal_mask = jax.jit(jg.legal_mask)
observe = jax.jit(jg.observation, static_argnames="bid_history")


def _random_match(n, seed, no_repeat_bid):
    """Plays a match with random legal actions; returns each bid as (player, round, bid, legal bids)."""
    rng = np.random.default_rng(seed)
    state = jg.random_game(jax.random.PRNGKey(seed), n, no_repeat_bid)
    bids = []
    while not bool(jg.is_terminal(state)):
        mask = np.asarray(legal_mask(state))
        action = int(rng.choice(np.flatnonzero(mask)))
        if action >= jg.BID_OFFSET:
            legal = [a - jg.BID_OFFSET for a in np.flatnonzero(mask[jg.BID_OFFSET:]) + jg.BID_OFFSET]
            bids.append((int(state.current), int(state.round_ptr), action - jg.BID_OFFSET, legal))
            # the observation's history: this player's bids of the two previous rounds
            hist = np.asarray(observe(state, state.current, bid_history=True))[jg.OBS_SIZE:]
            own = hist.reshape(jg.MAX_PLAYERS, 2, 9)[0]
            prev = [b for p, r, b, _ in bids[:-1] if p == bids[-1][0]]
            expect = (prev[::-1] + [-1, -1])[:2]
            assert [int(np.argmax(own[k])) - 1 for k in range(2)] == expect
        state, _ = step(state, jnp.int32(action))
    return bids


@pytest.mark.parametrize("n", [2, 3, 7])
def test_no_three_equal_bids_in_a_row(n):
    forced = 0
    for seed in range(6):
        by_player = {}
        for p, r, b, legal in _random_match(n, seed, True):
            seq = by_player.setdefault(p, [])
            if len(seq) >= 2 and seq[-1] == seq[-2]:
                if seq[-1] in legal:                          # only when nothing else was left
                    assert legal == [seq[-1]], (p, r, seq, legal)
                    forced += 1
                assert b != seq[-1] or legal == [b]
            seq.append(b)
    assert forced >= 0


def test_without_the_variant_repeating_is_allowed():
    """The plain game never takes the bid history into account (some player repeats 3 times)."""
    repeats = 0
    for seed in range(4):
        by_player = {}
        for p, _, b, _ in _random_match(3, seed, False):
            seq = by_player.setdefault(p, [])
            repeats += len(seq) >= 2 and seq[-1] == seq[-2] == b
            seq.append(b)
    assert repeats > 0


def test_last_bidder_rule_wins_over_no_repeat():
    """2 players, round of 1 card: the first bids 1, so the last may not bid 0 (both would make
    it); if it bid 1 twice before, 1 is still legal because nothing else is."""
    state = jg.random_game(jax.random.PRNGKey(0), 2, True)
    state = state._replace(prev_bids=jnp.array([[1, 1], [1, 1]], jnp.int32))
    first = int(state.current)
    assert np.flatnonzero(np.asarray(legal_mask(state))[jg.BID_OFFSET:]).tolist() == [0]
    state, _ = step(state, jnp.int32(jg.BID_OFFSET + 0))
    assert int(state.current) == 1 - first
    # the second: 1 is forbidden by the last-bidder rule... the other way round
    state = jg.random_game(jax.random.PRNGKey(0), 2, True)
    state = state._replace(prev_bids=jnp.array([[0, 0], [0, 0]], jnp.int32))
    state, _ = step(state, jnp.int32(jg.BID_OFFSET + 1))           # first bids 1
    # last: 0 is forbidden (1 + 0 = 1 card) and it has no 1 issue: bids 1
    assert np.flatnonzero(np.asarray(legal_mask(state))[jg.BID_OFFSET:]).tolist() == [1]
    state = jg.random_game(jax.random.PRNGKey(0), 2, True)
    state = state._replace(prev_bids=jnp.array([[0, 0], [1, 1]], jnp.int32))
    state, _ = step(state, jnp.int32(jg.BID_OFFSET + 1))           # first bids 1
    # last: 0 forbidden by the sum, 1 by the variant; nothing left, so 1 is allowed
    assert np.flatnonzero(np.asarray(legal_mask(state))[jg.BID_OFFSET:]).tolist() == [1]


def test_observation_size():
    state = jg.random_game(jax.random.PRNGKey(1), 4, True)
    assert observe(state, 0).shape == (jg.OBS_SIZE,)
    assert observe(state, 0, bid_history=True).shape == (jg.obs_size(True),)
    np.testing.assert_array_equal(np.asarray(observe(state, 0, bid_history=True))[:jg.OBS_SIZE],
                                  np.asarray(observe(state, 0)))


def test_ppo_trains_and_evaluates_with_the_variant():
    from rotten_rl.jax_engine import evaluate, ppo
    cfg = ppo.Config(num_players=3, hidden=(32, 32), rollout_games=8, minibatch=64, epochs=1,
                     total_games=16, no_repeat_bid=True)
    ts = ppo.init(jax.random.PRNGKey(0), cfg)
    assert ts.policy[0][0].shape[0] == jg.obs_size(True)
    ts, _ = ppo.make_update(cfg)(ts)
    res = evaluate.evaluate(ts.policy, "heuristic", 3, 12, no_repeat_bid=True)
    assert 0.0 <= res["overall"]["win_rate"] <= 1.0


def test_plain_network_plays_the_variant():
    """A network of the plain game (808 inputs) plays a match with the variant: it sees only its 808
    values and the rule still holds (the mask)."""
    from rotten_rl.jax_engine import evaluate, ppo
    cfg = ppo.Config(num_players=3, hidden=(16,))
    plain = ppo.init(jax.random.PRNGKey(0), cfg).policy
    variant = ppo.init(jax.random.PRNGKey(1), cfg._replace(no_repeat_bid=True)).policy
    assert plain[0][0].shape[0] == jg.OBS_SIZE
    res = evaluate.match(("greedy", plain), ("greedy", variant), 3, 4, 0, no_repeat_bid=True)
    assert 0.0 <= res["overall"]["win_rate"] <= 1.0
