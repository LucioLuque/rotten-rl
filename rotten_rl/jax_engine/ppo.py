"""
rotten_rl/jax_engine/ppo.py  —  PPO for La Podrida entirely in JAX (stage 3).

The same algorithm as `rotten_rl/algorithms/ppo.py` with --round_episodes (the recipe of the
current runs): one policy and one value network shared by every seat, self-play against the current
policy, illegal actions masked in the softmax, GAE(lambda) per seat with each round an episode (the
round reward goes to each player's last decision of that round), advantages normalized over the
batch, clipped policy loss, unclipped value loss, entropy bonus, gradient clipping, Adam with the
learning rate annealed linearly to 0, PyTorch's default init. Networks 808 -> hidden -> 60 / 1.

What changes is how it runs: a batch of `rollout_games` matches is played all at once on the device
(every match with n players has exactly the same number of decisions, so the batch ends together),
and the rollout, the advantages and every minibatch update are one compiled function. Python only
loops over batches.

Policies are saved as PyTorch state_dicts (`to_torch_state_dict`) so the evaluation, tournaments and
play.py load them like any other run.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import game as jg

ILLEGAL = -1e9


# ---------------------------------------------------------------------------------------------
# Networks: an MLP as a list of (W, b), W [in, out]; PyTorch nn.Linear default init.

def init_mlp(key, sizes):
    params = []
    for k, (i, o) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        kw, kb = jax.random.split(k)
        bound = 1.0 / np.sqrt(i)
        params.append((jax.random.uniform(kw, (i, o), minval=-bound, maxval=bound),
                       jax.random.uniform(kb, (o,), minval=-bound, maxval=bound)))
    return params


def mlp(params, x):
    for w, b in params[:-1]:
        x = jax.nn.relu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def to_torch_state_dict(params):
    """Policy params -> state_dict of rotten_rl.algorithms.dqn.mlp (nn.Sequential with ReLUs)."""
    import torch
    out = {}
    for i, (w, b) in enumerate(params):
        out[f"{2 * i}.weight"] = torch.as_tensor(np.asarray(w).T.copy())
        out[f"{2 * i}.bias"] = torch.as_tensor(np.asarray(b).copy())
    return out


# ---------------------------------------------------------------------------------------------

class Config(NamedTuple):
    num_players: int = 2
    hidden: tuple = (512, 512)
    learning_rate: float = 3e-4
    rollout_games: int = 512        # matches per update (--ppo_rollout_episodes)
    epochs: int = 5
    minibatch: int = 512
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    gae_lambda: float = 0.95
    gamma: float = 1.0
    max_grad_norm: float = 0.5
    total_games: int = 1_000_000    # the learning rate reaches 0 here (--ppo_anneal_lr)
    anneal: bool = True
    reward_scheme: str = "relative"


class TrainState(NamedTuple):
    policy: list
    value: list
    opt_state: tuple
    games_done: jnp.ndarray
    key: jnp.ndarray


def match_length(n):
    return int(sum(n + n * h for h in [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]))


def make_optimizer(cfg):
    """Clip the global gradient norm, then Adam (PyTorch's defaults); the learning rate is applied
    in `make_update`, so it can be annealed by games played."""
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                       optax.scale_by_adam(eps=1e-8))


def init(key, cfg):
    k_pi, k_v, key = jax.random.split(key, 3)
    policy = init_mlp(k_pi, [jg.OBS_SIZE, *cfg.hidden, jg.NUM_ACTIONS])
    value = init_mlp(k_v, [jg.OBS_SIZE, *cfg.hidden, 1])
    opt_state = make_optimizer(cfg).init((policy, value))
    return TrainState(policy, value, opt_state, jnp.int32(0), key)


def _rollout(policy, value, key, cfg):
    """Plays `rollout_games` whole matches. Returns per-step arrays [T, B]."""
    n, B, T = cfg.num_players, cfg.rollout_games, match_length(cfg.num_players)
    k_games, key = jax.random.split(key)
    states = jax.vmap(lambda k: jg.random_game(k, n))(jax.random.split(k_games, B))
    step = jax.vmap(lambda s, a: jg.step(s, a, cfg.reward_scheme))

    def body(carry, _):
        states, key = carry
        key, sub = jax.random.split(key)
        player = states.current
        obs = jax.vmap(jg.observation)(states, player)
        mask = jax.vmap(jg.legal_mask)(states)
        logits = jnp.where(mask, mlp(policy, obs), ILLEGAL)
        action = jax.random.categorical(sub, logits).astype(jnp.int32)
        logp = jax.nn.log_softmax(logits)[jnp.arange(B), action]
        v = mlp(value, obs)[:, 0]
        states, rewards = step(states, action)
        out = dict(obs=obs, mask=mask, action=action, logp=logp, value=v, player=player,
                   rewards=rewards)
        return (states, key), out

    (_, _), traj = jax.lax.scan(body, (states, key), None, length=T)
    return traj


def _round_closed(traj, n):
    """bool[T, B]: the move at t closed a round. Rounds close after a fixed number of decisions
    (n bids + n*h plays), so it is known without looking at the rewards (which can be all zero)."""
    ends = np.cumsum([n + n * h for h in [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]]) - 1
    closed = np.zeros(match_length(n), bool)
    closed[ends] = True
    return jnp.asarray(closed)[:, None] & jnp.ones_like(traj["player"], bool)


def _advantages(traj, cfg):
    """GAE per (match, seat), each round an episode: the round reward goes to every player's last
    decision of the round (done there); the other decisions get 0. Backward scan over time."""
    n = cfg.num_players
    closed = _round_closed(traj, n)
    B = traj["player"].shape[1]

    def body(carry, x):
        next_v, gae, pend, pend_r = carry          # [B, n] each: per seat
        player, v, r, cl = x
        # A round closed at this move: each seat's latest decision (this one or earlier) is its
        # last of the round.
        pend = jnp.where(cl[:, None], True, pend)
        pend_r = jnp.where(cl[:, None], r, pend_r)
        idx = jnp.arange(B)
        done = pend[idx, player]
        rew = jnp.where(done, pend_r[idx, player], 0.0)
        nv = jnp.where(done, 0.0, next_v[idx, player])
        g_next = jnp.where(done, 0.0, gae[idx, player])
        delta = rew + cfg.gamma * nv - v
        a = delta + cfg.gamma * cfg.gae_lambda * g_next
        next_v = next_v.at[idx, player].set(v)
        gae = gae.at[idx, player].set(a)
        pend = pend.at[idx, player].set(False)
        return (next_v, gae, pend, pend_r), a

    zeros = jnp.zeros((B, n))
    init = (zeros, zeros, jnp.zeros((B, n), bool), zeros)
    xs = (traj["player"], traj["value"], traj["rewards"], closed)
    _, adv = jax.lax.scan(body, init, xs, reverse=True)
    return adv, adv + traj["value"]


def _loss(params, batch, cfg):
    policy, value = params
    logits = jnp.where(batch["mask"], mlp(policy, batch["obs"]), ILLEGAL)
    logp_all = jax.nn.log_softmax(logits)
    logp = jnp.take_along_axis(logp_all, batch["action"][:, None], axis=1)[:, 0]
    ratio = jnp.exp(logp - batch["logp"])
    a = batch["adv"]
    loss_pi = -jnp.minimum(ratio * a, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
    loss_v = ((mlp(value, batch["obs"])[:, 0] - batch["ret"]) ** 2).mean()
    probs = jnp.exp(logp_all)
    entropy = -(jnp.where(batch["mask"], probs * logp_all, 0.0)).sum(axis=1).mean()
    loss = loss_pi + cfg.vf_coef * loss_v - cfg.ent_coef * entropy
    stats = dict(loss_policy=loss_pi, loss_value=loss_v, entropy=entropy,
                 clip_frac=(jnp.abs(ratio - 1) > cfg.clip).mean(),
                 approx_kl=(batch["logp"] - logp).mean())
    return loss, stats


def make_update(cfg):
    """One compiled step: play a batch of matches and run every PPO minibatch update on it."""
    opt = make_optimizer(cfg)
    T, B = match_length(cfg.num_players), cfg.rollout_games
    size = T * B
    num_mb = size // cfg.minibatch                     # the leftover transitions of each epoch are skipped

    def update(ts: TrainState):
        key, k_roll, k_perm = jax.random.split(ts.key, 3)
        traj = _rollout(ts.policy, ts.value, k_roll, cfg)
        adv, ret = _advantages(traj, cfg)
        flat = dict(obs=traj["obs"].reshape(size, -1), mask=traj["mask"].reshape(size, -1),
                    action=traj["action"].reshape(size), logp=traj["logp"].reshape(size),
                    adv=adv.reshape(size), ret=ret.reshape(size))
        explained = 1 - jnp.var(flat["ret"] - traj["value"].reshape(size)) / (jnp.var(flat["ret"]) + 1e-8)
        flat["adv"] = (flat["adv"] - flat["adv"].mean()) / (flat["adv"].std() + 1e-8)
        frac = jnp.maximum(0.0, 1.0 - ts.games_done / cfg.total_games) if cfg.anneal else 1.0
        lr = cfg.learning_rate * frac

        def epoch(carry, k):
            params, opt_state = carry
            perm = jax.random.permutation(k, size)[:num_mb * cfg.minibatch].reshape(num_mb, -1)

            def minibatch(carry, idx):
                params, opt_state = carry
                batch = jax.tree_util.tree_map(lambda x: x[idx], flat)
                (loss, stats), grads = jax.value_and_grad(_loss, has_aux=True)(params, batch, cfg)
                stats["grad_norm"] = optax.tree.norm(grads)
                updates, opt_state = opt.update(grads, opt_state, params)
                updates = jax.tree_util.tree_map(lambda u: -lr * u, updates)
                return (optax.apply_updates(params, updates), opt_state), stats

            return jax.lax.scan(minibatch, (params, opt_state), perm)

        (params, opt_state), stats = jax.lax.scan(
            epoch, ((ts.policy, ts.value), ts.opt_state), jax.random.split(k_perm, cfg.epochs))
        stats = jax.tree_util.tree_map(jnp.mean, stats)
        stats["explained_variance"] = explained
        stats["learning_rate"] = lr
        stats["reward_seat0"] = traj["rewards"][:, :, 0].sum(axis=0).mean()
        new = TrainState(params[0], params[1], opt_state, ts.games_done + B, key)
        return new, stats

    return jax.jit(update)
