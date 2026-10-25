"""El wrapper de OpenSpiel pasa las simulaciones aleatorias de pyspiel."""
import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")

import rotten_rl.openspiel_game  # noqa: E402,F401  registra el juego "podrida"
from rotten_rl.evaluation import evaluate, heuristic_opponent, random_opponent  # noqa: E402


@pytest.mark.parametrize("players", [2, 4, 7])
def test_random_sim(players):
    game = pyspiel.load_game("podrida", {"players": players})
    pyspiel.random_sim_test(game, num_sims=3, serialize=False, verbose=False)


@pytest.mark.parametrize("players", [2, 3])
def test_evaluate_rota_asientos(players):
    game = pyspiel.load_game("podrida", {"players": players})
    rng = np.random.RandomState(0)
    m = evaluate(game, heuristic_opponent(rng), random_opponent(rng), players, 2 * players, rng)
    assert sorted(m["win_rate_by_seat"]) == list(range(players))
    assert 0.0 <= m["overall"]["win_rate"] <= 1.0
