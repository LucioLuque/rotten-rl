"""
scripts/jax_engine/profile_ppo.py  —  Where the time goes in one JAX PPO update.

Times, separately and compiled, each part of one update (rollout_games matches):
  - game only:       play the matches with random legal actions, no observation
  - game + obs:      the same, building the 808 observation at every decision
  - rollout:         game + obs + policy and value networks (what PPO plays)
  - advantages:      GAE per seat over the rollout
  - update:          the --epochs passes of minibatch updates
  - whole update:    ppo.make_update (all of the above in one compiled function)

    python scripts/jax_engine/profile_ppo.py --players 2 --rollout_games 512
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rotten_rl.jax_engine import game as jg  # noqa: E402
from rotten_rl.jax_engine import ppo  # noqa: E402


def timeit(fn, *args, repeats=5):
    out = fn(*args)
    jax.block_until_ready(out)                       # compile + warm up
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args)
        jax.block_until_ready(out)
    return (time.perf_counter() - t0) / repeats, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--rollout_games", type=int, default=512)
    p.add_argument("--hidden", default="512,512")
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    n, B = a.players, a.rollout_games
    cfg = ppo.Config(num_players=n, hidden=tuple(int(x) for x in a.hidden.split(",")),
                     rollout_games=B)
    T = ppo.match_length(n)
    ts = ppo.init(jax.random.PRNGKey(0), cfg)
    key = jax.random.PRNGKey(1)

    def play(key, with_obs):
        states = jax.vmap(lambda k: jg.random_game(k, n))(jax.random.split(key, B))

        def body(carry, _):
            states, key, acc = carry
            key, sub = jax.random.split(key)
            mask = jax.vmap(jg.legal_mask)(states)
            if with_obs:
                acc = acc + jax.vmap(jg.observation)(states, states.current).sum()
            action = jax.random.categorical(sub, jnp.where(mask, 0.0, -jnp.inf)).astype(jnp.int32)
            states, r = jax.vmap(jg.step)(states, action)
            return (states, key, acc + r.sum()), None

        (_, _, acc), _ = jax.lax.scan(body, (states, key, 0.0), None, length=T)
        return acc

    game_only = jax.jit(lambda k: play(k, False))
    game_obs = jax.jit(lambda k: play(k, True))
    rollout = jax.jit(lambda k: ppo._rollout(ts.policy, ts.value, k, cfg))
    advantages = jax.jit(lambda traj: ppo._advantages(traj, cfg))
    whole = ppo.make_update(cfg)

    rows = []
    t_game, _ = timeit(game_only, key, repeats=a.repeats)
    rows.append(("game only (random actions)", t_game))
    t_obs, _ = timeit(game_obs, key, repeats=a.repeats)
    rows.append(("game + observation", t_obs))
    t_roll, traj = timeit(rollout, key, repeats=a.repeats)
    rows.append(("rollout: game + obs + networks", t_roll))
    t_adv, _ = timeit(advantages, traj, repeats=a.repeats)
    rows.append(("advantages (GAE)", t_adv))
    t_whole, _ = timeit(lambda s: whole(s)[1], ts, repeats=a.repeats)
    rows.append(("whole update (compiled together)", t_whole))
    rows.append(("  => minibatch updates (whole - rollout - GAE)", t_whole - t_roll - t_adv))

    print(f"JAX {jax.__version__} on {jax.devices()}")
    print(f"{n} players, {B} matches per update ({T} decisions each, {T * B:,} transitions), "
          f"net {a.hidden}, {cfg.epochs} epochs x {T * B // cfg.minibatch} minibatches of {cfg.minibatch}")
    for name, t in rows:
        print(f"  {name:48s} {t * 1000:9.1f} ms   {100 * t / t_whole:5.1f}% of the update")
    print(f"  matches per second (whole update): {B / t_whole:,.0f}")


if __name__ == "__main__":
    main()
