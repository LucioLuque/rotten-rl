import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
from rotten_rl.match_value.win_prob import fit_round_variances, normal_win_prob, var_left  # noqa: E402


def test_two_players_closed_form():
    from scipy.stats import norm
    p = np.asarray(normal_win_prob(jnp.array([10.0, 0.0]), 100.0))
    assert p[0] == pytest.approx(norm.cdf(10 / np.sqrt(200)), abs=1e-5)
    assert p.sum() == pytest.approx(1.0, abs=1e-6)


def test_many_players_monte_carlo():
    rng = np.random.default_rng(0)
    s = np.array([12.0, 5.0, 0.0, -3.0, 1.0])
    final = s + rng.normal(size=(400000, 5)) * np.sqrt(80)
    mc = np.bincount(final.argmax(1), minlength=5) / len(final)
    assert np.asarray(normal_win_prob(jnp.array(s), 80.0)) == pytest.approx(mc, abs=0.004)


def test_no_rounds_left_is_the_win_share():
    p = np.asarray(normal_win_prob(jnp.array([[5.0, 5.0, 0.0], [3.0, 1.0, 1.0]]), jnp.zeros(2)))
    assert p == pytest.approx(np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]))


def test_round_variances_and_tail():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(20000, 13, 2)) * 3
    var = fit_round_variances(pts)
    assert var == pytest.approx(np.full(13, 9.0), rel=0.05)
    left = np.asarray(var_left(var, np.arange(14)))
    assert left[0] == pytest.approx(var.sum(), rel=1e-5) and left[13] == 0.0
