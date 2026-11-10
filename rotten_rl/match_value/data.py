"""Self-play matches of one policy in every seat, keeping each player's points in each round."""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from rotten_rl.jax_engine import game as jg
from rotten_rl.jax_engine.evaluate import _act
from rotten_rl.jax_engine.ppo import match_length


@partial(jax.jit, static_argnames=("num_players", "num_matches", "kind"))
def _play(params, key, num_players, num_matches, kind):
    n = num_players
    deal_key, key = jax.random.split(key)
    states = jax.vmap(lambda k: jg.random_game(k, n))(jax.random.split(deal_key, num_matches))
    step = jax.vmap(lambda s, a: jg.step(s, a, "relative"))

    def body(carry, _):
        states, key = carry
        key, sub = jax.random.split(key)
        obs = jax.vmap(jg.observation)(states, states.current)
        mask = jax.vmap(jg.legal_mask)(states)
        action = _act(kind, params, obs, mask, states, sub)
        new, _ = step(states, action)
        return (new, key), new.scores

    _, scores = jax.lax.scan(body, (states, key), None, length=match_length(n))
    return scores                                                  # [T, G, n]


def round_points(params, num_players, num_matches, seed=0, kind="sample", chunk=20000):
    """int[G, 13, n]: points of each player in each round of `num_matches` self-play matches.
    kind: "sample" (as PPO trains), "greedy" or "heuristic" (params None)."""
    n = num_players
    ends = np.cumsum([n + n * h for h in np.asarray(jg.LADDER)]) - 1   # moves that close a round
    out = []
    key = jax.random.PRNGKey(seed)
    for start in range(0, num_matches, chunk):
        key, sub = jax.random.split(key)
        g = min(chunk, num_matches - start)
        scores = np.asarray(jax.device_get(_play(params, sub, n, g, kind)))[ends]   # [13, g, n]
        cum = np.concatenate([np.zeros((1,) + scores.shape[1:], scores.dtype), scores])
        out.append(np.diff(cum, axis=0).transpose(1, 0, 2))
    return np.concatenate(out)
