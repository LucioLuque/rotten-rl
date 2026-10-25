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


@pytest.mark.parametrize("players", [2, 4, 7])
def test_cada_agente_recibe_todo_su_reward(players):
    """Lo que ve cada agente en sus propios turnos (con with_pending_reward)
    tiene que sumar exactamente su retorno de la partida."""
    from open_spiel.python import rl_environment
    from rotten_rl.training import with_pending_reward

    rng = np.random.RandomState(0)
    env = rl_environment.Environment("podrida", players=players)
    for _ in range(3):
        ts = env.reset()
        pending = [0.0] * players
        got = [0.0] * players
        seen = [False] * players
        while not ts.last():
            p = ts.observations["current_player"]
            visto = with_pending_reward(ts, pending, p)
            if seen[p]:                      # el primer TimeStep no cierra transicion
                got[p] += visto.rewards[p]
            seen[p] = True
            ts = env.step([int(rng.choice(ts.observations["legal_actions"][p]))])
            for i in range(players):
                pending[i] += ts.rewards[i]
        for p in range(players):
            got[p] += with_pending_reward(ts, pending, p).rewards[p]
        assert got == pytest.approx(env.get_state.returns())


@pytest.mark.parametrize("players", [2, 3])
def test_evaluate_rota_asientos(players):
    game = pyspiel.load_game("podrida", {"players": players})
    rng = np.random.RandomState(0)
    m = evaluate(game, heuristic_opponent(rng), random_opponent(rng), players, 2 * players, rng)
    assert sorted(m["win_rate_by_seat"]) == list(range(players))
    assert 0.0 <= m["overall"]["win_rate"] <= 1.0
