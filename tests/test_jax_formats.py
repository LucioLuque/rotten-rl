"""Match formats (rotten_rl/jax_engine/formats.py and game.format_game, docs/rule_variants/)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from rotten_rl.jax_engine import formats  # noqa: E402
from rotten_rl.jax_engine import game as jg  # noqa: E402

step = jax.jit(jg.step)
legal_mask = jax.jit(jg.legal_mask)
observe = jax.jit(jg.observation, static_argnames=("bid_history", "format_info"))


def _winner(cards, lead, trump):
    """Python reference: highest trump, else highest of the suit led (trump None = no trump)."""
    def power(c):
        s, r = c // 13, c % 13
        return 200 + r if s == trump else (100 + r if s == lead else 0)
    return max(range(len(cards)), key=lambda i: power(cards[i]))


def _play(state, seed):
    """Plays a match with random legal actions checking the rules; returns the hand size and
    trump (None = no trump) of each round played."""
    rng = np.random.default_rng(seed)
    n = state.hands.shape[0]
    rounds = []
    while not bool(jg.is_terminal(state)):
        r = int(state.round_ptr)
        h = int(state.round_h[r])
        nt = bool(state.no_trump[r])
        trump = None if nt else int(state.trump_card) // 13
        assert int(jg.trump_suit(state)) == (4 if nt else trump)
        if not rounds or rounds[-1][0] != r:
            assert int(state.hands.sum()) == n * h            # everyone holds h cards
            rounds.append((r, h, trump))
        mask = np.asarray(legal_mask(state))
        if state.phase == jg.PLAYING:
            hand = np.flatnonzero(np.asarray(state.hands[state.current]))
            lead = int(state.lead_suit)
            if lead >= 0 and not any(c // 13 == lead for c in hand):
                if nt or not any(c // 13 == trump for c in hand):
                    assert sorted(np.flatnonzero(mask[:52])) == sorted(hand)   # anything goes
            trick_before = np.asarray(state.trick).copy()
        action = int(rng.choice(np.flatnonzero(mask)))
        playing = bool(state.phase == jg.PLAYING)
        p = int(state.current)
        tl = int(state.trick_len)
        lead_before = int(state.lead_suit)
        won_before = np.asarray(state.tricks_won).copy()
        state, _ = step(state, jnp.int32(action))
        if playing and tl == n - 1:                           # the trick just closed
            cards = trick_before.copy()
            cards[p] = action
            lead = lead_before
            w = _winner(list(cards), lead, trump)
            if int(state.round_ptr) == r:
                assert int(np.asarray(state.tricks_won)[w]) == won_before[w] + 1
    return [(h, t) for _, h, t in rounds]


def test_the_classic_ladder_is_the_default():
    s = jg.random_game(jax.random.PRNGKey(0), 3)
    assert np.asarray(s.round_h).tolist()[:13] == [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
    assert int(jg.num_rounds(s)) == 13 and not np.asarray(s.no_trump).any()


@pytest.mark.parametrize("n", [2, 3, 7])
def test_formats_play_by_the_rules(n):
    for seed in range(5):
        state = jg.format_game(jax.random.PRNGKey(100 + seed), n)
        h = np.asarray(state.round_h)
        nt = np.asarray(state.no_trump)
        played = _play(state, seed)
        assert [x for x, _ in played] == h[h > 0].tolist()
        assert [t is None for _, t in played] == nt[h > 0].tolist()


def test_a_named_format_with_rounds_without_trump():
    h = formats._LADDERS[3]
    nt = np.zeros(formats.MAX_ROUNDS, bool)
    nt[[7, 15]] = True                                   # 1..7,7*,6..1,7,7*
    orders = jax.vmap(lambda k: jax.random.permutation(k, 52))(jax.random.split(jax.random.PRNGKey(5), 16))
    state = jg.new_game(orders, 4, False, h, nt)
    played = _play(state, 0)
    assert [x for x, _ in played] == [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 7, 7]
    assert [i for i, (_, t) in enumerate(played) if t is None] == [7, 15]


def test_format_observation():
    state = jg.format_game(jax.random.PRNGKey(3), 3)
    obs = np.asarray(observe(state, 0, bid_history=True, format_info=True))
    assert obs.shape == (jg.obs_size(True, True),)
    fmt = obs[jg.OBS_SIZE + jg.HISTORY_SIZE:]
    ahead = fmt[:-1].reshape(jg.MAX_ROUNDS, 9)
    h = np.asarray(state.round_h)
    nt = np.asarray(state.no_trump)
    assert np.argmax(ahead[:, :8], axis=1).tolist() == h.tolist()
    assert ahead[:, 8].astype(bool).tolist() == (nt & (h > 0)).tolist()
    assert fmt[-1] == float(state.no_repeat_bid)
    # the 808 part says "no trump" in a round without trump
    state = jg.new_game(jax.vmap(lambda k: jax.random.permutation(k, 52))(
        jax.random.split(jax.random.PRNGKey(1), 16)), 3, False, formats._LADDERS[1],
        np.ones(formats.MAX_ROUNDS, bool))
    obs = np.asarray(observe(state, 0))
    off = 7 + 13 + 7 + 2                                # hand_size, round_index, cards_remaining, phase
    assert obs[off:off + 5].tolist() == [0, 0, 0, 0, 1]


def test_sample_format_frequencies():
    keys = jax.random.split(jax.random.PRNGKey(0), 20000)
    h, nt, rep = (np.asarray(x) for x in jax.vmap(formats.sample_format)(keys))
    usual = (h == formats._LADDERS[1]).all(1) & ~nt.any(1)
    assert 0.28 < usual.mean() < 0.36                    # 20 % named + the random draws of it
    assert 0.47 < rep.mean() < 0.53
    assert set((h > 0).sum(1).tolist()) == {13, 14, 15, 16}
    assert not (nt & (h == 0)).any()


def test_ppo_trains_and_evaluates_with_formats():
    from rotten_rl.jax_engine import evaluate, ppo
    cfg = ppo.Config(num_players=3, hidden=(32, 32), rollout_games=8, minibatch=64, epochs=1,
                     total_games=16, formats=True)
    ts = ppo.init(jax.random.PRNGKey(0), cfg)
    assert ts.policy[0][0].shape[0] == jg.obs_size(True, True)
    ts, stats = ppo.make_update(cfg)(ts)
    assert all(np.isfinite(float(v)) for v in stats.values())
    res = evaluate.evaluate(ts.policy, "heuristic", 3, 12, formats=True)
    assert 0.0 <= res["overall"]["win_rate"] <= 1.0
    plain = ppo.init(jax.random.PRNGKey(1), ppo.Config(num_players=3, hidden=(16,))).policy
    res = evaluate.match(("greedy", plain), ("greedy", ts.policy), 3, 4, 0, formats=True)
    assert 0.0 <= res["overall"]["win_rate"] <= 1.0
