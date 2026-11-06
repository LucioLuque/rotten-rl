"""JAX PPO (rotten_rl/jax_engine/ppo.py): the advantages must be the ones of the PyTorch PPO with
--round_episodes (one trajectory per seat, each round an episode, the round reward on each
player's last decision of the round), and a short training must run and save a loadable policy."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optax")

from rotten_rl.jax_engine import ppo  # noqa: E402


def _reference_advantages(traj, n, gamma, lam):
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
                    streams[q][-1][2] = rewards[t, g, q]
                    streams[q][-1][3] = True
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
