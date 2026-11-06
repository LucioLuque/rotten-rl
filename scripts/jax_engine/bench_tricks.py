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


def play_with_network(key, num_games, num_players, hand_size, params):
    """Like tricks.play_random_rounds, but every move first runs a policy MLP of the training size
    (808 -> 512 -> 512 -> 60) on a stand-in observation: estimates the cost of deciding on the GPU.
    The real observation is stage 2."""
    import jax.numpy as jnp
    k_deal, k_play = jax.random.split(key)
    states = jax.vmap(lambda k: tricks.deal(k, num_players, hand_size))(jax.random.split(k_deal, num_games))

    def policy(obs):
        h = jax.nn.relu(obs @ params[0] + params[1])
        h = jax.nn.relu(h @ params[2] + params[3])
        return h @ params[4] + params[5]

    def body(i, carry):
        states, key = carry
        key, sub = jax.random.split(key)
        obs = jnp.concatenate([states.hands.reshape(num_games, -1).astype(jnp.float32),
                               jnp.zeros((num_games, 808 - 52 * num_players), jnp.float32)], axis=1)
        logits = policy(obs)[:, :52]
        mask = jax.vmap(tricks.legal_mask)(states)
        cards = jax.random.categorical(sub, jnp.where(mask, logits, -jnp.inf)).astype(jnp.int32)
        return jax.vmap(tricks.step)(states, cards), key

    states, _ = jax.lax.fori_loop(0, num_players * hand_size, body, (states, k_play))
    return states.tricks_won


def bench_jax_network(players, hand_size, batch, repeats):
    import jax.numpy as jnp
    keys = jax.random.split(jax.random.PRNGKey(42), 3)
    sizes = [(808, 512), (512, 512), (512, 60)]
    params = []
    for k, (i, o) in zip(keys, sizes):
        params += [jax.random.normal(k, (i, o)) * 0.03, jnp.zeros(o)]
    run = jax.jit(play_with_network, static_argnums=(1, 2, 3))
    run(jax.random.PRNGKey(0), batch, players, hand_size, params).block_until_ready()
    t0 = time.perf_counter()
    for i in range(repeats):
        run(jax.random.PRNGKey(i + 1), batch, players, hand_size, params).block_until_ready()
    return repeats * batch * players * hand_size / (time.perf_counter() - t0)


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
    print("with a policy network of the training size deciding every move (jax):")
    for batch in (int(b) for b in a.batches.split(",")):
        rate = bench_jax_network(a.players, a.hand_size, batch, a.repeats)
        print(f"  jax+net, {batch:>6} at once {rate:>11,.0f}   x{rate / base:.1f} vs engine.py")


if __name__ == "__main__":
    main()
