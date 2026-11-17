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


def mlp_features(params, x, dtype=None):
    """The last hidden layer of the MLP `params` (everything but its output layer), as `mlp`."""
    for w, b in params[:-1]:
        if dtype is None:
            x = jax.nn.relu(x @ w + b)
        else:
            x = jax.nn.relu(jnp.dot(x.astype(dtype), w.astype(dtype),
                                    preferred_element_type=jnp.float32) + b)
    return x


def mlp(params, x, dtype=None):
    """With `dtype` (e.g. bfloat16), every matmul takes its inputs and weights in that type and
    accumulates in float32; biases, activations' sums and the output stay float32. The weights
    themselves (and Adam) are always float32."""
    if dtype is None:
        for w, b in params[:-1]:
            x = jax.nn.relu(x @ w + b)
        w, b = params[-1]
        return x @ w + b
    for i, (w, b) in enumerate(params):
        x = jnp.dot(x.astype(dtype), w.astype(dtype), preferred_element_type=jnp.float32) + b
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x


def compute_dtype(cfg):
    """None (plain float32) or the dtype of the matmuls (Config.dtype)."""
    return None if cfg.dtype == "float32" else jnp.dtype(cfg.dtype)


def to_torch_state_dict(params):
    """Policy params -> state_dict of rotten_rl.algorithms.dqn.mlp (nn.Sequential with ReLUs)."""
    import torch
    out = {}
    for i, (w, b) in enumerate(params):
        out[f"{2 * i}.weight"] = torch.as_tensor(np.asarray(w).T.copy())
        out[f"{2 * i}.bias"] = torch.as_tensor(np.asarray(b).copy())
    return out


def from_torch_state_dict(state_dict):
    """The inverse of `to_torch_state_dict`: a saved policy back to JAX params."""
    n = len(state_dict) // 2
    return [(jnp.asarray(state_dict[f"{2 * i}.weight"].numpy().T),
             jnp.asarray(state_dict[f"{2 * i}.bias"].numpy())) for i in range(n)]


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
    dtype: str = "float32"          # "bfloat16": matmuls (and stored observations) in bf16
    episode: str = "round"          # "match": GAE runs over the whole match (rounds don't cut it)
    match_bonus: float = 0.0        # lambda: weight of game.final_rewards at the end of the match
    rank_weight: float = 0.25       # share of the place (vs winning) in game.final_rewards
    # With episode == "round" and match_bonus != 0: each round also gets lambda * (change in the
    # probability of winning the match), rotten_rl/match_value/win_prob.py with these per-round
    # variances (13 floats, from scripts/match_value/fit_win_prob.py).
    round_var: tuple = ()
    # Auxiliary task (--aux_hands): a head on the policy's last hidden layer predicts which unseen
    # cards each other player holds (sigmoid per relative seat and card); its loss, times this, is
    # added to PPO's. 0 = off (no head).
    aux_coef: float = 0.0


AUX_SEATS = 6                       # other players, by relative position 1..6 (padded)


class TrainState(NamedTuple):
    policy: list
    value: list
    opt_state: tuple
    games_done: jnp.ndarray
    key: jnp.ndarray
    aux: list = None                # the auxiliary head [(W, b)] when cfg.aux_coef > 0


def trainable(ts):
    """The parameters the optimizer updates: (policy, value) or (policy, value, aux)."""
    return (ts.policy, ts.value) if ts.aux is None else (ts.policy, ts.value, ts.aux)


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
    aux = (init_mlp(jax.random.fold_in(k_pi, 1), [cfg.hidden[-1], AUX_SEATS * 52])
           if cfg.aux_coef else None)
    ts = TrainState(policy, value, None, jnp.int32(0), key, aux)
    return ts._replace(opt_state=make_optimizer(cfg).init(trainable(ts)))


def _opponent_logits(opponent, obs, dt, groups):
    """Logits of the frozen opponent(s): one network, or with groups > 0 a stack of `groups`
    networks, each playing a consecutive block of B / groups matches."""
    if not groups:
        return mlp(opponent, obs, dt)
    B = obs.shape[0]
    x = obs.reshape(groups, B // groups, -1)
    return jax.vmap(lambda p, o: mlp(p, o, dt))(opponent, x).reshape(B, -1)


def _rollout(policy, value, key, cfg, opponent=None, groups=0, seats=1):
    """Plays `rollout_games` whole matches. Returns per-step arrays [T, B].
    With `opponent` (the policy params of a frozen network, any hidden sizes), each match has one
    seat at random for `policy` (sampling) and the others for `opponent`, greedy as in the
    evaluations and tournaments; traj["learner"] [T, B] marks the learner's decisions and
    traj["learner_seat"] [B] its seat. With groups > 0, `opponent` is a stack of that many
    networks (leading axis) and each plays a block of B / groups consecutive matches. With seats = m
    > 1, the learner takes m seats at random in each match (the opponents the other n - m) and
    traj["learner_seats"] bool[B, n] marks them."""
    n, B, T = cfg.num_players, cfg.rollout_games, match_length(cfg.num_players)
    k_games, k_seat, key = jax.random.split(key, 3)
    states = jax.vmap(lambda k: jg.random_game(k, n))(jax.random.split(k_games, B))
    learner_seat = jax.random.randint(k_seat, (B,), 0, n)
    if seats == 1:
        learner_seats = jnp.arange(n)[None, :] == learner_seat[:, None]
    else:                                              # m seats at random per match
        rank = jnp.argsort(jnp.argsort(jax.random.uniform(k_seat, (B, n)), axis=1), axis=1)
        learner_seats = rank < seats
    step = jax.vmap(lambda s, a: jg.step(s, a, cfg.reward_scheme))
    dt = compute_dtype(cfg)

    def body(carry, _):
        states, key = carry
        key, sub = jax.random.split(key)
        player = states.current
        obs = jax.vmap(jg.observation)(states, player)
        if dt is not None:      # the first matmul would round it anyway; halves the stored batch
            obs = obs.astype(dt)
        mask = jax.vmap(jg.legal_mask)(states)
        logits = jnp.where(mask, mlp(policy, obs, dt), ILLEGAL)
        action = jax.random.categorical(sub, logits).astype(jnp.int32)
        logp = jax.nn.log_softmax(logits)[jnp.arange(B), action]
        v = mlp(value, obs, dt)[:, 0]
        out = dict(obs=obs, mask=mask, action=action, logp=logp, value=v, player=player)
        if cfg.aux_coef:                               # targets of the auxiliary task
            out.update(_aux_targets(states, player))
        if opponent is not None:
            learner = learner_seats[jnp.arange(B), player]
            theirs = jnp.argmax(jnp.where(mask, _opponent_logits(opponent, obs, dt, groups), ILLEGAL),
                                axis=1)
            action = jnp.where(learner, action, theirs.astype(jnp.int32))
            out.update(action=action, learner=learner)
        states, rewards = step(states, action)
        out.update(rewards=rewards, scores=states.scores)
        return (states, key), out

    (states, _), traj = jax.lax.scan(body, (states, key), None, length=T)
    traj["final_scores"] = states.scores                 # [B, n]: every match ends at T
    if opponent is not None:
        traj["learner_seat"] = learner_seat
        traj["learner_seats"] = learner_seats
    return traj


def _aux_targets(states, player):
    """Targets of the auxiliary task at a decision of `player` (batched over matches): aux_hands
    bool[B, AUX_SEATS, 52], the hand of the player at each relative seat 1..6 (False past n), and
    aux_unseen bool[B, 52], the cards `player` cannot see (only those count in the loss)."""
    B, n = states.hands.shape[:2]
    rel = jnp.arange(1, AUX_SEATS + 1)
    seat = (player[:, None] + rel[None, :]) % n                   # [B, 6]
    hands = jnp.take_along_axis(states.hands, seat[:, :, None], axis=1)
    hands = hands & (rel < n)[None, :, None]
    mine = states.hands[jnp.arange(B), player]
    unseen = (~mine & (states.played_by < 0)
              & (jnp.arange(52)[None, :] != states.trump_card[:, None]))
    return dict(aux_hands=hands, aux_unseen=unseen)


def _round_ends(n):
    """The moves that close each of the 13 rounds (the same for every match)."""
    return np.cumsum([n + n * h for h in [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]]) - 1


def _win_prob_shaping(traj, cfg):
    """float[T, B, n]: at each move that closes a round, the change in each player's probability
    of winning the match (0 elsewhere). Summed over the match: final win share - 1/n."""
    from rotten_rl.match_value import win_prob
    n = cfg.num_players
    ends = _round_ends(n)
    scores = traj["scores"][ends]                                            # [13, B, n]
    var = win_prob.var_left(np.asarray(cfg.round_var), np.arange(1, 14))     # [13]
    var = jnp.broadcast_to(var[:, None], scores.shape[:2])
    after = win_prob.normal_win_prob(scores, var)                            # [13, B, n]
    before = jnp.concatenate([jnp.full((1,) + after.shape[1:], 1.0 / n), after[:-1]])
    return jnp.zeros(traj["rewards"].shape).at[ends].set(after - before)


def _round_closed(traj, n):
    """bool[T, B]: the move at t closed a round. Rounds close after a fixed number of decisions
    (n bids + n*h plays), so it is known without looking at the rewards (which can be all zero)."""
    ends = _round_ends(n)
    closed = np.zeros(match_length(n), bool)
    closed[ends] = True
    return jnp.asarray(closed)[:, None] & jnp.ones_like(traj["player"], bool)


def _advantages(traj, cfg, match_bonus=None):
    """GAE per (match, seat). Each round's reward goes to every player's last decision of that
    round; the other decisions get 0. With cfg.episode == "round" each round is an episode (done
    at those decisions); with "match" only the end of the match is, and the last decision of each
    player also gets `match_bonus` * game.final_rewards. With "round", `match_bonus` != 0 and
    cfg.round_var, each round's reward adds `match_bonus` * the change in the probability of
    winning the match (_win_prob_shaping). Backward scan over time."""
    n = cfg.num_players
    closed = _round_closed(traj, n)
    T, B = traj["player"].shape
    if match_bonus is None:
        match_bonus = cfg.match_bonus
    if cfg.episode == "match":
        ends = jnp.zeros((T,), bool).at[T - 1].set(True)[:, None] & closed
        bonus = match_bonus * jax.vmap(lambda s: jg.final_rewards(s, cfg.rank_weight))(
            traj["final_scores"])                                    # [B, n]
    elif cfg.episode == "round":
        ends = closed
        bonus = jnp.zeros((B, n))
    else:
        raise ValueError(f"episode must be 'round' or 'match', not {cfg.episode!r}")
    last = jnp.zeros((T,), bool).at[T - 1].set(True)
    if cfg.episode == "round" and cfg.round_var:
        shaping = match_bonus * _win_prob_shaping(traj, cfg)
    else:
        shaping = jnp.zeros(traj["rewards"].shape)

    def body(carry, x):
        next_v, gae, pend_r, pend_done = carry     # [B, n] each: per seat
        player, v, r, cl, end, is_last, shape = x
        # The move at t closed a round (or the match): each seat's latest decision (this one or
        # earlier) is its last one before that, and gets the reward.
        pend_r = (pend_r + jnp.where(cl[:, None], r + shape, 0.0)
                  + jnp.where(is_last, bonus, 0.0))
        pend_done = pend_done | end[:, None]
        idx = jnp.arange(B)
        done = pend_done[idx, player]
        rew = pend_r[idx, player]
        nv = jnp.where(done, 0.0, next_v[idx, player])
        g_next = jnp.where(done, 0.0, gae[idx, player])
        delta = rew + cfg.gamma * nv - v
        a = delta + cfg.gamma * cfg.gae_lambda * g_next
        next_v = next_v.at[idx, player].set(v)
        gae = gae.at[idx, player].set(a)
        pend_r = pend_r.at[idx, player].set(0.0)
        pend_done = pend_done.at[idx, player].set(False)
        return (next_v, gae, pend_r, pend_done), a

    zeros = jnp.zeros((B, n))
    init = (zeros, zeros, zeros, jnp.zeros((B, n), bool))
    xs = (traj["player"], traj["value"], traj["rewards"], closed, ends, last, shaping)
    _, adv = jax.lax.scan(body, init, xs, reverse=True)
    return adv, adv + traj["value"]


def _loss(params, batch, cfg, ent_coef):
    policy, value = params[0], params[1]
    dt = compute_dtype(cfg)
    if cfg.aux_coef:
        feats = mlp_features(policy, batch["obs"], dt)
        w, b = policy[-1]
        raw = (feats @ w + b if dt is None else
               jnp.dot(feats.astype(dt), w.astype(dt), preferred_element_type=jnp.float32) + b)
        logits = jnp.where(batch["mask"], raw, ILLEGAL)
    else:
        logits = jnp.where(batch["mask"], mlp(policy, batch["obs"], dt), ILLEGAL)
    logp_all = jax.nn.log_softmax(logits)
    logp = jnp.take_along_axis(logp_all, batch["action"][:, None], axis=1)[:, 0]
    ratio = jnp.exp(logp - batch["logp"])
    a = batch["adv"]
    loss_pi = -jnp.minimum(ratio * a, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
    loss_v = ((mlp(value, batch["obs"], dt)[:, 0] - batch["ret"]) ** 2).mean()
    probs = jnp.exp(logp_all)
    entropy = -(jnp.where(batch["mask"], probs * logp_all, 0.0)).sum(axis=1).mean()
    loss = loss_pi + cfg.vf_coef * loss_v - ent_coef * entropy
    stats = dict(loss_policy=loss_pi, loss_value=loss_v, entropy=entropy,
                 clip_frac=(jnp.abs(ratio - 1) > cfg.clip).mean(),
                 approx_kl=(batch["logp"] - logp).mean())
    if cfg.aux_coef:                                   # binary cross-entropy on the unseen cards
        aw, ab = params[2][0]
        pred = (feats @ aw + ab).reshape(-1, AUX_SEATS, 52)
        target = batch["aux_hands"].astype(jnp.float32)
        m = (batch["aux_unseen"][:, None, :] & (batch["aux_seat_on"][:, :, None])).astype(jnp.float32)
        bce = optax.sigmoid_binary_cross_entropy(pred, target)
        loss_aux = (bce * m).sum() / jnp.maximum(m.sum(), 1.0)
        acc = (((pred > 0) == batch["aux_hands"]) * m).sum() / jnp.maximum(m.sum(), 1.0)
        loss = loss + cfg.aux_coef * loss_aux
        stats.update(loss_aux=loss_aux, aux_accuracy=acc)
    return loss, stats


def init_seeds(seeds, cfg):
    """One TrainState per seed, stacked on a leading axis (for `make_update(cfg, multi_seed=True)`).
    Seed s starts exactly as `init(jax.random.PRNGKey(s), cfg)`."""
    keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])
    return jax.vmap(lambda k: init(k, cfg))(keys)


def unstack(tree, i):
    """The i-th seed of a stacked TrainState (or any stacked tree)."""
    return jax.tree_util.tree_map(lambda x: x[i], tree)


def make_update(cfg, multi_seed=False, exploit=False, league_groups=0, league_self=False,
                league_seats=1):
    """One compiled step: play a batch of matches and run every PPO minibatch update on it.
    With exploit, the step is `update(ts, hp, opponent)`: it trains against the frozen policy
    `opponent` (with multi_seed, one per stacked state) instead of self-play. Only the learner's
    decisions are trained on: every seat makes T / n decisions per match, so the batch is
    T / n * B transitions (n times fewer minibatches per update than self-play), and the logged
    "reward_seat0" is the learner's reward per match.
    With league_groups = K > 0 (league training, implies exploit), the step is
    `update(ts, hp, pool, pool_count)`: `pool` is a stack of frozen policies (leading axis of a fixed
    size, the first `pool_count` valid); each update draws K of them at random (with replacement)
    and each plays B / K of the matches. With league_self, the first of the K is the current
    policy itself (greedy, like the others). With league_seats = m, the learner takes m seats of
    each match (m * T / n decisions trained on per match instead of T / n).
    With multi_seed, the step takes a stacked TrainState (`init_seeds`) and trains every seed at
    once, each with its own networks, optimizer, matches and random keys; stats get a seed axis.
    The GPU is latency-bound on one seed's small minibatches, so the extra seeds come almost free.
    The step is `update(ts, hp=None)`: `hp` = {"learning_rate", "ent_coef"[, "match_bonus"]}
    overrides those of cfg
    (with multi_seed, one value per stacked state, so each can train with its own)."""
    opt = make_optimizer(cfg)
    n, T, B = cfg.num_players, match_length(cfg.num_players), cfg.rollout_games
    exploit = exploit or league_groups > 0
    if league_groups and B % league_groups:
        raise ValueError(f"rollout_games ({B}) must be a multiple of league_groups ({league_groups})")
    if not 1 <= league_seats < n:
        raise ValueError(f"league_seats must be between 1 and {n - 1}")
    L = T * league_seats // n if exploit else T        # decisions trained on per match
    size = L * B
    num_mb = size // cfg.minibatch                     # the leftover transitions of each epoch are skipped

    def update(ts: TrainState, hp=None, opponent=None, pool_count=None):
        base_lr = cfg.learning_rate if hp is None else hp["learning_rate"]
        ent_coef = cfg.ent_coef if hp is None else hp["ent_coef"]
        match_bonus = cfg.match_bonus if hp is None else hp.get("match_bonus", cfg.match_bonus)
        key, k_roll, k_perm = jax.random.split(ts.key, 3)
        if league_groups:                              # draw this update's opponents from the pool
            pick_idx = jax.random.randint(jax.random.fold_in(ts.key, 7), (league_groups,), 0,
                                          pool_count)
            opponent = jax.tree_util.tree_map(lambda x: x[pick_idx], opponent)
            if league_self:
                opponent = jax.tree_util.tree_map(lambda o, p: o.at[0].set(p), opponent, ts.policy)
        traj = _rollout(ts.policy, ts.value, k_roll, cfg, opponent if exploit else None,
                        league_groups, league_seats)
        adv, ret = _advantages(traj, cfg, match_bonus)
        if exploit:                                    # keep the learner's T / n decisions per match
            idx = jax.vmap(lambda col: jnp.nonzero(col, size=L)[0], in_axes=1, out_axes=1)(
                traj["learner"])                       # [L, B]
            cols = jnp.arange(B)[None, :]
            pick = lambda x: x[idx, cols]              # noqa: E731
            totals = traj["rewards"].sum(axis=0)                     # [B, n]
            reward = (totals * traj["learner_seats"]).sum() / (B * league_seats)
        else:
            pick = lambda x: x                         # noqa: E731
            reward = traj["rewards"][:, :, 0].sum(axis=0).mean()
        flat = dict(obs=pick(traj["obs"]).reshape(size, -1), mask=pick(traj["mask"]).reshape(size, -1),
                    action=pick(traj["action"]).reshape(size), logp=pick(traj["logp"]).reshape(size),
                    adv=pick(adv).reshape(size), ret=pick(ret).reshape(size))
        if cfg.aux_coef:
            flat["aux_hands"] = pick(traj["aux_hands"]).reshape(size, AUX_SEATS, 52)
            flat["aux_unseen"] = pick(traj["aux_unseen"]).reshape(size, 52)
            flat["aux_seat_on"] = jnp.broadcast_to(jnp.arange(1, AUX_SEATS + 1) < n, (size, AUX_SEATS))
        values = pick(traj["value"]).reshape(size)
        explained = 1 - jnp.var(flat["ret"] - values) / (jnp.var(flat["ret"]) + 1e-8)
        flat["adv"] = (flat["adv"] - flat["adv"].mean()) / (flat["adv"].std() + 1e-8)
        frac = jnp.maximum(0.0, 1.0 - ts.games_done / cfg.total_games) if cfg.anneal else 1.0
        lr = base_lr * frac

        def epoch(carry, k):
            params, opt_state = carry
            perm = jax.random.permutation(k, size)[:num_mb * cfg.minibatch].reshape(num_mb, -1)

            def minibatch(carry, idx):
                params, opt_state = carry
                batch = jax.tree_util.tree_map(lambda x: x[idx], flat)
                (loss, stats), grads = jax.value_and_grad(_loss, has_aux=True)(params, batch, cfg, ent_coef)
                stats["grad_norm"] = optax.tree.norm(grads)
                updates, opt_state = opt.update(grads, opt_state, params)
                updates = jax.tree_util.tree_map(lambda u: -lr * u, updates)
                return (optax.apply_updates(params, updates), opt_state), stats

            return jax.lax.scan(minibatch, (params, opt_state), perm)

        (params, opt_state), stats = jax.lax.scan(
            epoch, (trainable(ts), ts.opt_state), jax.random.split(k_perm, cfg.epochs))
        stats = jax.tree_util.tree_map(jnp.mean, stats)
        stats["explained_variance"] = explained
        stats["learning_rate"] = lr
        stats["reward_seat0"] = reward
        new = TrainState(params[0], params[1], opt_state, ts.games_done + B, key,
                         params[2] if cfg.aux_coef else None)
        return new, stats

    return jax.jit(jax.vmap(update) if multi_seed else update)
