"""Probability of winning the match from the scores and the rounds left (the potential Φ of a
match reward, as in Suphx's global reward prediction or ICM in poker tournaments).

Model ("normal"): what each player still scores is an independent normal draw with variance
`var_left` (the sum of the per-round variances of the rounds not played yet). Then

    P(i ends on top) = E_{X_i}[ prod_{j != i} Phi((s_i + X_i - s_j) / sigma) ]

computed with Gauss-Hermite quadrature. Ties at the end share the win (as game.final_rewards). The
per-round variances come from real matches (`fit_round_variances`): half the variance of the
difference between two players' points in that round, which keeps the correlation between players
(with 2 players the tricks of one are the tricks the other did not take).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

NUM_ROUNDS = 13
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(40)


def fit_round_variances(round_points):
    """float[13]: sigma_r^2 per round from round_points [G, 13, n] (points of each player in each
    round): mean over pairs of seats of Var(p_i - p_j) / 2."""
    p = np.asarray(round_points, np.float64)
    n = p.shape[-1]
    var = np.zeros(p.shape[1])
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in pairs:
        var += np.var(p[:, :, i] - p[:, :, j], axis=0) / 2
    return var / len(pairs)


def var_left(round_var, rounds_played):
    """Variance still to come after `rounds_played` rounds (int array or scalar)."""
    tail = np.concatenate([np.cumsum(np.asarray(round_var)[::-1])[::-1], [0.0]])
    return jnp.asarray(tail, jnp.float32)[rounds_played]


def _top_share(scores):
    s = scores.astype(jnp.float32)
    top = s == s.max(axis=-1, keepdims=True)
    return top / top.sum(axis=-1, keepdims=True)


def normal_win_prob(scores, var):
    """float[..., n]: P(each player ends on top) given scores [..., n] and var [...] (variance of
    each player's remaining points). With var == 0, the share of the win now."""
    s = scores.astype(jnp.float32)
    var = jnp.asarray(var, jnp.float32)
    sigma = jnp.sqrt(jnp.maximum(var, 1e-12))[..., None, None]             # [..., 1, 1]
    x = jnp.sqrt(2.0) * jnp.asarray(_GH_X, jnp.float32)                    # [K]
    w = jnp.asarray(_GH_W, jnp.float32) / jnp.sqrt(jnp.pi)
    diff = s[..., :, None] - s[..., None, :]                               # [..., n, n] s_i - s_j
    z = (diff[..., None] / sigma[..., None]) + x                           # [..., n, n, K]
    cdf = norm.cdf(z)
    n = s.shape[-1]
    eye = jnp.eye(n, dtype=bool)[..., None]
    prod = jnp.where(eye, 1.0, cdf).prod(axis=-2)                          # [..., n, K]
    p = (prod * w).sum(axis=-1)
    p = p / p.sum(axis=-1, keepdims=True)                                  # quadrature error
    return jnp.where(var[..., None] > 1e-9, p, _top_share(s))
