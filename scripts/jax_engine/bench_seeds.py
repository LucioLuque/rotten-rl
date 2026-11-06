"""
scripts/jax_engine/bench_seeds.py  —  How many seeds to train at once in one process.

Times one compiled PPO update (rollout + GAE + every minibatch update) training 1, 2, 4, ... seeds
stacked with vmap (ppo.make_update(cfg, multi_seed=True)) and prints matches per second, per seed
and in total. One seed's minibatches leave the GPU mostly idle, so the total should grow with the
seeds until the GPU is really busy.

    python scripts/jax_engine/bench_seeds.py --players 2 --rollout_games 512 --seeds 1,2,4,8,16
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax  # noqa: E402

from rotten_rl.jax_engine import ppo  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--rollout_games", type=int, default=512)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--hidden", default="512,512")
    p.add_argument("--seeds", default="1,2,4,8,16", help="numbers of seeds to try")
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    cfg = ppo.Config(num_players=a.players, hidden=tuple(int(x) for x in a.hidden.split(",")),
                     rollout_games=a.rollout_games, minibatch=a.minibatch)
    print(f"JAX {jax.__version__} on {jax.devices()}; {a.players} players, {a.rollout_games} matches "
          f"per update and seed, minibatch {a.minibatch}, net {a.hidden}", flush=True)
    update = ppo.make_update(cfg, multi_seed=True)
    base = None
    for s in [int(x) for x in a.seeds.split(",")]:
        ts = ppo.init_seeds(range(s), cfg)
        try:
            ts, _ = update(ts)
            jax.block_until_ready(ts)                    # compile + warm up
            t0 = time.perf_counter()
            for _ in range(a.repeats):
                ts, _ = update(ts)
                jax.block_until_ready(ts)
        except Exception as e:                           # out of GPU memory
            print(f"  {s:3d} seeds: failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
            break
        dt = (time.perf_counter() - t0) / a.repeats
        total = s * a.rollout_games / dt
        base = base or total
        print(f"  {s:3d} seeds: {dt * 1000:8.1f} ms per update | {total / s:7,.0f} matches/s per seed"
              f" | {total:8,.0f} in total (x{total / base:.1f})", flush=True)


if __name__ == "__main__":
    main()
