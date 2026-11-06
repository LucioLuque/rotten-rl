"""
scripts/jax_engine/bench_tricks.py  —  Speed of the trick phase: current engine vs JAX (stage 1).

Plays rounds of the playing phase with random legal cards and reports moves per second:
  - engine:  rotten_rl/engine.py, one game at a time (legal_action_ids + apply_action per move)
  - pyspiel: the OpenSpiel wrapper training uses (legal_actions + apply_action per move)
  - jax:     rotten_rl/jax_engine/tricks.py, `--batches` games at once, compiled

Only simulation: no observation and no network (those are stage 2 and 3). Run it on the lab with
the GPU free to measure the GPU; on the laptop it measures JAX on the CPU.

    python scripts/jax_engine/bench_tricks.py --players 2 --hand_size 7
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax  # noqa: E402

from rotten_rl.engine import PodridaGame  # noqa: E402
from rotten_rl.jax_engine import tricks  # noqa: E402


def bench_engine(players, hand_size, seconds):
    rng = random.Random(0)
    moves, elapsed = 0, 0.0
    game = PodridaGame(num_players=players, seed=0)
    while elapsed < seconds:
        game.start_round(hand_size)
        while game.phase == "bidding":
            game.apply_action(rng.choice(game.legal_action_ids()))
        t0 = time.perf_counter()
        while game.phase == "playing":
            game.apply_action(rng.choice(game.legal_action_ids()))
            moves += 1
        elapsed += time.perf_counter() - t0
    return moves / elapsed


def bench_pyspiel(players, hand_size, seconds):
    import pyspiel
    import rotten_rl.openspiel_game  # noqa: F401  registers "podrida"
    game = pyspiel.load_game("podrida", {"players": players})
    rng = random.Random(0)
    moves, elapsed = 0, 0.0
    while elapsed < seconds:
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(outcomes, probs)[0])
                continue
            legal = state.legal_actions()
            playing = legal[0] < 52
            t0 = time.perf_counter()
            state.apply_action(rng.choice(legal))
            if playing:
                elapsed += time.perf_counter() - t0
                moves += 1
    return moves / elapsed


def bench_jax(players, hand_size, batch, repeats):
    run = jax.jit(tricks.play_random_rounds, static_argnums=(1, 2, 3))
    run(jax.random.PRNGKey(0), batch, players, hand_size).block_until_ready()   # compile
    t0 = time.perf_counter()
    for i in range(repeats):
        run(jax.random.PRNGKey(i + 1), batch, players, hand_size).block_until_ready()
    elapsed = time.perf_counter() - t0
    return repeats * batch * players * hand_size / elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--hand_size", type=int, default=7)
    p.add_argument("--seconds", type=float, default=5.0, help="time for the Python engines")
    p.add_argument("--batches", default="32,1024,4096,16384")
    p.add_argument("--repeats", type=int, default=20)
    a = p.parse_args()
    print(f"JAX {jax.__version__} on {jax.devices()}")
    print(f"{a.players} players, {a.hand_size} cards, random legal play, moves per second:")
    base = bench_engine(a.players, a.hand_size, a.seconds)
    print(f"  engine.py              {base:>12,.0f}")
    try:
        spiel = bench_pyspiel(a.players, a.hand_size, a.seconds)
        print(f"  pyspiel wrapper        {spiel:>12,.0f}")
    except ImportError:
        spiel = None
    for batch in (int(b) for b in a.batches.split(",")):
        rate = bench_jax(a.players, a.hand_size, batch, a.repeats)
        print(f"  jax, {batch:>6} at once   {rate:>12,.0f}   x{rate / base:.1f} vs engine.py"
              + (f", x{rate / spiel:.1f} vs pyspiel" if spiel else ""))


if __name__ == "__main__":
    main()
