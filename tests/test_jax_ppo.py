"""JAX PPO (rotten_rl/jax_engine/ppo.py): the advantages must be the ones of the PyTorch PPO with
--round_episodes (one trajectory per seat, each round an episode, the round reward on each
player's last decision of the round), and a short training must run and save a loadable policy."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optax")

from rotten_rl.jax_engine import ppo  # noqa: E402


def _reference_advantages(traj, n, gamma, lam, match=False, bonus=None):
    """Like rotten_rl/algorithms/ppo.py: per (match, seat) stream, rewards and dones as VectorRunner
    gives them with round_episodes, then SharedPPO._advantages."""
    player = np.asarray(traj["player"])
    value = np.asarray(traj["value"])
    rewards = np.asarray(traj["rewards"])
    T, B = player.shape
    ends = set((np.cumsum([n + n * h for h in [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]]) - 1).tolist())
    adv = np.zeros((T, B), np.float32)
    for g in range(B):
        streams = {p: [] for p in range(n)}          # list of [t, value, reward, done]
        for t in range(T):
            p = player[t, g]
            streams[p].append([t, value[t, g], 0.0, False])
            if t in ends:
                for q in range(n):
                    streams[q][-1][2] += rewards[t, g, q]
                    streams[q][-1][3] = not match or t == T - 1
            if match and t == T - 1:
                for q in range(n):
                    streams[q][-1][2] += bonus[g, q]
        for steps in streams.values():
            gae = 0.0
            for i in reversed(range(len(steps))):
                t, v, r, done = steps[i]
                next_v = 0.0 if done or i + 1 == len(steps) else steps[i + 1][1]
                delta = r + gamma * next_v - v
                gae = delta + gamma * lam * (0.0 if done else gae)
                adv[t, g] = gae
    return adv


@pytest.mark.parametrize("n", [2, 3])
def test_advantages_match_pytorch_ppo(n):
    cfg = ppo.Config(num_players=n, hidden=(32,), rollout_games=4)
    ts = ppo.init(jax.random.PRNGKey(n), cfg)
    traj = jax.jit(lambda k: ppo._rollout(ts.policy, ts.value, k, cfg))(jax.random.PRNGKey(7))
    adv, ret = ppo._advantages(traj, cfg)
    ref = _reference_advantages(traj, n, cfg.gamma, cfg.gae_lambda)
    np.testing.assert_allclose(np.asarray(adv), ref, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ret), ref + np.asarray(traj["value"]), rtol=1e-4, atol=1e-5)
    assert np.abs(np.asarray(traj["rewards"])).sum() > 0


@pytest.mark.parametrize("n", [2, 3])
def test_match_episode_advantages(n):
    """episode="match": one episode per match (rounds don't cut GAE) and the final bonus on each
    player's last decision."""
    from rotten_rl.jax_engine import game as jg
    cfg = ppo.Config(num_players=n, hidden=(32,), rollout_games=4, episode="match", match_bonus=0.7)
    ts = ppo.init(jax.random.PRNGKey(n), cfg)
    traj = jax.jit(lambda k: ppo._rollout(ts.policy, ts.value, k, cfg))(jax.random.PRNGKey(7))
    adv, ret = ppo._advantages(traj, cfg)
    bonus = 0.7 * np.asarray(jax.vmap(lambda s: jg.final_rewards(s, cfg.rank_weight))(
        traj["final_scores"]))
    ref = _reference_advantages(traj, n, cfg.gamma, cfg.gae_lambda, match=True, bonus=bonus)
    np.testing.assert_allclose(np.asarray(adv), ref, rtol=1e-4, atol=1e-5)
    # With lambda = 1 the return of the first decision is everything the seat gets in the match.
    cfg1 = cfg._replace(gae_lambda=1.0)
    _, ret1 = ppo._advantages(traj, cfg1)
    player = np.asarray(traj["player"])
    total = np.asarray(traj["rewards"]).sum(axis=0) + bonus           # [B, n]
    for g in range(4):
        for q in range(n):
            t0 = int(np.argmax(player[:, g] == q))
            np.testing.assert_allclose(float(ret1[t0, g]), total[g, q], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n", [2, 3])
def test_win_prob_shaping(n):
    """episode="round" with round_var: each round adds lambda * the change in the probability of
    winning; the changes of a match add up to the final win share - 1/n."""
    var = tuple(float(v) for v in np.linspace(40, 60, 13))
    cfg = ppo.Config(num_players=n, hidden=(32,), rollout_games=4, match_bonus=0.8, round_var=var)
    ts = ppo.init(jax.random.PRNGKey(n), cfg)
    traj = jax.jit(lambda k: ppo._rollout(ts.policy, ts.value, k, cfg))(jax.random.PRNGKey(7))
    shaping = np.asarray(ppo._win_prob_shaping(traj, cfg))                 # [T, B, n]
    final = np.asarray(traj["final_scores"]).astype(float)
    top = final == final.max(axis=1, keepdims=True)
    share = top / top.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(shaping.sum(axis=0), share - 1.0 / n, atol=1e-5)
    adv, _ = ppo._advantages(traj, cfg)
    shaped = dict(traj, rewards=np.asarray(traj["rewards"]) + 0.8 * shaping)
    ref = _reference_advantages(shaped, n, cfg.gamma, cfg.gae_lambda)
    np.testing.assert_allclose(np.asarray(adv), ref, rtol=1e-4, atol=1e-5)
    # Without round_var the bonus is ignored in round episodes (train_ppo refuses that case).
    adv0, _ = ppo._advantages(traj, cfg._replace(round_var=()))
    np.testing.assert_allclose(np.asarray(adv0), _reference_advantages(traj, n, cfg.gamma,
                                                                       cfg.gae_lambda), rtol=1e-4,
                               atol=1e-5)


def test_final_rewards():
    from rotten_rl.jax_engine import game as jg
    r = np.asarray(jg.final_rewards(jax.numpy.array([50, 80, 20]), rank_weight=0.25))
    np.testing.assert_allclose(r, [-1 / 3, 1 - 1 / 3 + 0.125, -1 / 3 - 0.125], atol=1e-6)
    tie = np.asarray(jg.final_rewards(jax.numpy.array([80, 80, 20]), rank_weight=0.25))
    np.testing.assert_allclose(tie, [0.5 - 1 / 3 + 0.0625, 0.5 - 1 / 3 + 0.0625, -1 / 3 - 0.125],
                               atol=1e-6)
    for s in ([50, 80, 20], [80, 80, 20], [10, 10], [3, 9, 9, 1, 0, 7, 7]):
        assert abs(float(jg.final_rewards(jax.numpy.array(s)).sum())) < 1e-6


def test_update_runs_and_exports():
    torch = pytest.importorskip("torch")
    from rotten_rl.algorithms.dqn import mlp
    cfg = ppo.Config(num_players=2, hidden=(32, 32), rollout_games=8, minibatch=64, epochs=2,
                     total_games=64)
    ts = ppo.init(jax.random.PRNGKey(0), cfg)
    update = ppo.make_update(cfg)
    ts, stats = update(ts)
    ts, stats = update(ts)
    assert int(ts.games_done) == 16
    assert all(np.isfinite(float(v)) for v in stats.values())
    net = mlp(808, [32, 32], 60)
    net.load_state_dict(ppo.to_torch_state_dict(ts.policy))
    x = np.random.RandomState(0).rand(3, 808).astype(np.float32)
    np.testing.assert_allclose(net(torch.as_tensor(x)).detach().numpy(),
                               np.asarray(ppo.mlp(ts.policy, x)), rtol=1e-4, atol=1e-5)


def test_multi_seed_update_matches_single_seeds():
    """Training seeds 0 and 1 stacked in one vmapped step gives what each seed gives alone."""
    cfg = ppo.Config(num_players=2, hidden=(32,), rollout_games=4, minibatch=62, total_games=100)
    single = ppo.make_update(cfg)
    multi = ppo.make_update(cfg, multi_seed=True)
    stacked = ppo.init_seeds([0, 1], cfg)
    alone = [ppo.init(jax.random.PRNGKey(s), cfg) for s in (0, 1)]
    for _ in range(2):
        stacked, stats = multi(stacked)
        alone = [single(ts)[0] for ts in alone]
    assert stats["entropy"].shape == (2,)
    for i, ts in enumerate(alone):
        got = ppo.unstack(stacked, i)
        assert int(got.games_done) == int(ts.games_done) == 8
        for a, b in zip(jax.tree_util.tree_leaves(got.policy), jax.tree_util.tree_leaves(ts.policy)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)
    leaves0 = jax.tree_util.tree_leaves(ppo.unstack(stacked, 0).policy)
    leaves1 = jax.tree_util.tree_leaves(ppo.unstack(stacked, 1).policy)
    assert not np.allclose(np.asarray(leaves0[0]), np.asarray(leaves1[0]))


def test_members_with_their_own_hyperparameters():
    """Members stacked with their own learning rate and entropy bonus (hp) train as each would
    alone with those values in cfg; members with the same seed start equal and then differ."""
    base = dict(num_players=2, hidden=(32,), rollout_games=4, minibatch=62, total_games=100)
    values = [(1e-4, 0.01), (1e-3, 0.0), (3e-4, 0.05)]
    multi = ppo.make_update(ppo.Config(**base), multi_seed=True)
    stacked = ppo.init_seeds([0, 0, 0], ppo.Config(**base))
    hp = {"learning_rate": np.array([v[0] for v in values], np.float32),
          "ent_coef": np.array([v[1] for v in values], np.float32)}
    alone = []
    for lr, ent in values:
        cfg = ppo.Config(**base, learning_rate=lr, ent_coef=ent)
        ts, update = ppo.init(jax.random.PRNGKey(0), cfg), ppo.make_update(cfg)
        for _ in range(2):
            ts = update(ts)[0]
        alone.append(ts)
    for _ in range(2):
        stacked, stats = multi(stacked, hp)
    np.testing.assert_allclose(stats["learning_rate"], hp["learning_rate"] * (1 - 4 / 100), rtol=1e-6)
    for i, ts in enumerate(alone):
        got = ppo.unstack(stacked, i)
        for a, b in zip(jax.tree_util.tree_leaves(got.policy), jax.tree_util.tree_leaves(ts.policy)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)
    w = [np.asarray(jax.tree_util.tree_leaves(ppo.unstack(stacked, i).policy)[0]) for i in range(3)]
    assert not np.allclose(w[0], w[1])


def test_bfloat16_matmuls():
    """With dtype bfloat16 the network gives about the float32 output, the weights stay float32
    and a short training runs."""
    params = ppo.init_mlp(jax.random.PRNGKey(1), [808, 64, 64, 60])
    x = (np.random.RandomState(0).rand(16, 808) < 0.1).astype(np.float32)
    full = np.asarray(ppo.mlp(params, x))
    half = ppo.mlp(params, x, jax.numpy.bfloat16)
    assert half.dtype == np.float32
    np.testing.assert_allclose(np.asarray(half), full, atol=0.02 * np.abs(full).max())
    cfg = ppo.Config(num_players=2, hidden=(32, 32), rollout_games=8, minibatch=64, epochs=2,
                     total_games=64, dtype="bfloat16")
    ts = ppo.init(jax.random.PRNGKey(0), cfg)
    ts, stats = ppo.make_update(cfg)(ts)
    assert all(np.isfinite(float(v)) for v in stats.values())
    assert all(leaf.dtype == np.float32 for leaf in jax.tree_util.tree_leaves(ts.policy))
