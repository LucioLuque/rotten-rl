"""
rotten_rl/jax_engine/formats.py  —  The format of a match, drawn at random before it starts.

The table decides before the match: the sequence of hand sizes (the "ladder"), which rounds are
played without trump, and whether nobody may bid the same amount 3 rounds in a row
(docs/rule_variants/). A format is h int[MAX_ROUNDS] (cards per round, 0 past the end),
no_trump bool[MAX_ROUNDS] and no_repeat_bid bool.

`sample_format(key)` draws one (jit- and vmap-friendly). The draw gives some weight to the formats
I see most at the table and spreads the rest over variations, so the agent learns whatever the table
decides:

  - 20 %  1..7,7,6..1                 (the usual ladder, 14 rounds)
  - 10 %  1..7,7*,6..1,7,7*           (* = without trump)
  - 10 %  1..7,7*,6*..1*
  - 60 %  at random:
      ladder: 7 repeated (65 %) or once (35 %); an extra 7,7 at the end (25 %);
      rounds without trump (each mode with its probability):
        none (40 %), one round at random (25 %), each round with probability 0.2 (15 %),
        the whole way down from the (last) peak (10 %), the second 7 of each pair (10 %).
  - independently of the rest, the no-repeat-bid rule in half of the matches.
"""

import jax
import jax.numpy as jnp
import numpy as np

MAX_ROUNDS = 16

_UP = [1, 2, 3, 4, 5, 6, 7]
_DOWN = [6, 5, 4, 3, 2, 1]


def _pad(x):
    return x + [0] * (MAX_ROUNDS - len(x))


# Ladders: 0 single peak, 1 double peak, 2 single + extra 7,7, 3 double + extra 7,7.
_LADDERS = np.array([_pad(_UP + _DOWN), _pad(_UP + [7] + _DOWN),
                     _pad(_UP + _DOWN + [7, 7]), _pad(_UP + [7] + _DOWN + [7, 7])], np.int32)
_LENGTHS = np.array([13, 14, 15, 16], np.int32)

# Named formats: (ladder, rounds without trump).
_NAMED = [
    (1, []),                              # 1..7,7,6..1
    (3, [7, 15]),                         # 1..7,7*,6..1,7,7*
    (1, list(range(7, 14))),              # 1..7,7*,6*..1*
]
NAMED_WEIGHTS = [0.20, 0.10, 0.10]       # the rest (0.60) is drawn at random
_NAMED_LADDER = np.array([l for l, _ in _NAMED], np.int32)
_NAMED_NT = np.array([[r in nt for r in range(MAX_ROUNDS)] for _, nt in _NAMED], bool)

P_DOUBLE_PEAK = 0.65
P_EXTRA_PAIR = 0.25
NT_MODES = [0.40, 0.25, 0.15, 0.10, 0.10]   # none, one, each with P_NT_ROUND, way down, second 7s
P_NT_ROUND = 0.2
P_NO_REPEAT_BID = 0.5

CLASSIC = (_LADDERS[0], np.zeros(MAX_ROUNDS, bool))   # the 13-round ladder of the first agents


def sample_format(key):
    """(h int[MAX_ROUNDS], no_trump bool[MAX_ROUNDS], no_repeat_bid bool) of one match."""
    k = jax.random.split(key, 8)
    r = jnp.arange(MAX_ROUNDS)
    # The random part.
    double = jax.random.bernoulli(k[0], P_DOUBLE_PEAK)
    extra = jax.random.bernoulli(k[1], P_EXTRA_PAIR)
    ladder = double.astype(jnp.int32) + 2 * extra.astype(jnp.int32)
    length = jnp.asarray(_LENGTHS)[ladder]
    base = length - 2 * extra.astype(jnp.int32)                 # rounds before the extra pair
    real = r < length
    mode = jax.random.choice(k[2], len(NT_MODES), p=jnp.asarray(NT_MODES))
    one = r == jax.random.randint(k[3], (), 0, length)
    each = jax.random.bernoulli(k[4], P_NT_ROUND, (MAX_ROUNDS,))
    last_peak = jnp.where(double, 7, 6)
    way_down = (r >= last_peak) & (r < base)
    second_7s = (r == last_peak) | (extra & (r == base + 1))
    nt = jnp.stack([jnp.zeros(MAX_ROUNDS, bool), one, each, way_down, second_7s])[mode] & real
    # Named or random.
    probs = jnp.asarray(NAMED_WEIGHTS + [1.0 - sum(NAMED_WEIGHTS)])
    pick = jax.random.choice(k[5], len(probs), p=probs)
    named = pick < len(_NAMED)
    i = jnp.minimum(pick, len(_NAMED) - 1)
    ladder = jnp.where(named, jnp.asarray(_NAMED_LADDER)[i], ladder)
    nt = jnp.where(named, jnp.asarray(_NAMED_NT)[i], nt)
    h = jnp.asarray(_LADDERS)[ladder]
    no_repeat = jax.random.bernoulli(k[6], P_NO_REPEAT_BID)
    return h, nt, no_repeat


def format_string(h, no_trump):
    """'1,2,...,7,7*,...' (* = without trump)."""
    return ",".join(f"{int(x)}{'*' if t else ''}" for x, t in zip(h, no_trump) if x > 0)
