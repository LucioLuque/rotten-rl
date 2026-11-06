"""
scripts/jax_engine/bench_game.py  —  Speed of whole matches: pyspiel wrapper vs JAX (stage 2).

Plays full matches (13 rounds) and reports decisions per second:
  - pyspiel:      the wrapper training uses, one match at a time: legal_actions + info_state_tensor
                  (the observation of the player to move) + apply_action per decision
  - jax:          rotten_rl/jax_engine/game.py, `--batches` matches at once, random legal actions,
                  with the observation of the player to move built at every decision
  - jax+net:      the same, but a policy MLP of the training size (808 -> 512 -> 512 -> 60) picks
                  the action from that observation: everything except the PPO update

Reference (26/10, docs/tecnico/optimizacion.md): training today, 8 runs sharing the RTX 4090, is
~220 matches of 3 players per second in total, ~40,000 decisions per second.

    python scripts/jax_engine/bench_game.py --players 3
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rotten_rl.jax_engine import game as jg  # noqa: E402


def bench_pyspiel(players, seconds):
    import pyspiel
    import rotten_rl.openspiel_game as og
    game = pyspiel.load_game("podrida", {"players": players})
    rng = random.Random(0)
    decisions, elapsed = 0, 0.0
    while elapsed < seconds:
        state = game.new_initial_state()
        t0 = time.perf_counter()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(outcomes, probs)[0])
                continue
            legal = state.legal_actions()
            og.info_state_tensor(state, state.current_player())
            state.apply_action(rng.choice(legal))
            decisions += 1
        elapsed += time.perf_counter() - t0
    return decisions / elapsed


def match_length(n):
    return int(sum(n + n * h for h in [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]))


def make_play(n, num_games, params):
    def policy(obs):
        h = jax.nn.relu(obs @ params[0] + params[1])
        h = jax.nn.relu(h @ params[2] + params[3])
        return h @ params[4] + params[5]

    def play(key):
        k_games, k_play = jax.random.split(key)
        states = jax.vmap(lambda k: jg.random_game(k, n))(jax.random.split(k_games, num_games))

        def body(i, carry):
            states, key, check = carry
            key, sub = jax.random.split(key)
            obs = jax.vmap(jg.observation)(states, states.current)
            mask = jax.vmap(jg.legal_mask)(states)
            logits = policy(obs) if params is not None else jnp.zeros(mask.shape)
            action = jax.random.categorical(sub, jnp.where(mask, logits, -jnp.inf)).astype(jnp.int32)
            states, r = jax.vmap(jg.step)(states, action)
            return states, key, check + obs.sum() + r.sum()

        _, _, check = jax.lax.fori_loop(0, match_length(n), body, (states, k_play, 0.0))
        return check

    return jax.jit(play)


def bench_jax(n, batch, repeats, with_net):
    params = None
    if with_net:
        keys = jax.random.split(jax.random.PRNGKey(42), 3)
        params = []
        for k, (i, o) in zip(keys, [(808, 512), (512, 512), (512, 60)]):
            params += [jax.random.normal(k, (i, o)) * 0.03, jnp.zeros(o)]
    play = make_play(n, batch, params)
    play(jax.random.PRNGKey(0)).block_until_ready()          # compile
    t0 = time.perf_counter()
    for i in range(repeats):
        play(jax.random.PRNGKey(i + 1)).block_until_ready()
    return repeats * batch * match_length(n) / (time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--players", type=int, default=3)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--batches", default="256,1024,4096")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--no_pyspiel", action="store_true")
    a = p.parse_args()
    print(f"JAX {jax.__version__} on {jax.devices()}")
    print(f"{a.players} players, whole matches ({match_length(a.players)} decisions each), "
          "decisions per second:")
    base = None
    if not a.no_pyspiel:
        base = bench_pyspiel(a.players, a.seconds)
        print(f"  pyspiel + observation      {base:>12,.0f}   (one match at a time, no network)")
    for with_net in (False, True):
        for batch in (int(b) for b in a.batches.split(",")):
            rate = bench_jax(a.players, batch, a.repeats, with_net)
            name = "jax+net" if with_net else "jax"
            vs = f"   x{rate / base:.1f} vs pyspiel" if base else ""
            print(f"  {name:8s} {batch:>6} at once     {rate:>12,.0f}{vs}")


if __name__ == "__main__":
    main()
