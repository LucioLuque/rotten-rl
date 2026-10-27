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


def _grabador(vistas, elegir):
    """Juega con `elegir` y anota las manos al empezar cada partida (primer
    canto de la primera ronda, le toque a quien le toque)."""
    def act(state, legal):
        eng = state._engine
        if state._round_ptr == 0 and not any(p.bid is not None for p in eng.players):
            vistas.append(tuple(tuple(map(str, p.hand)) for p in eng.players))
        return elegir(legal)
    return act


@pytest.mark.parametrize("players", [3, 4, 7])
def test_duplicado_da_empate_exacto_entre_iguales(players):
    """Con todos jugando la misma heuristica (sin azar), cada reparto se juega
    igual en los n asientos: el evaluado gana exactamente 1/n y la diferencia de
    puntaje es 0, en lote y de a una."""
    game = pyspiel.load_game("podrida", {"players": players})
    for parallel in (False, True):
        m = evaluate(game, heuristic_opponent(None), heuristic_opponent(None), players,
                     5 * players, np.random.RandomState(0), parallel=parallel, deal_seed=3)
        o = m["overall"]
        assert o["win_rate"] == pytest.approx(1 / players)
        assert o["mean_score_diff"] == pytest.approx(0.0, abs=1e-9)
        assert o["win_rate_se"] == pytest.approx(0.0, abs=1e-9)


def test_duplicado_mismas_cartas_en_cada_asiento_y_agente():
    """Cada reparto da las mismas manos en sus n partidas, distintos repartos
    dan manos distintas, y la misma semilla da los mismos repartos aunque se
    juegue distinto."""
    game = pyspiel.load_game("podrida", {"players": 3})
    manos = []
    evaluate(game, _grabador(manos, lambda legal: legal[0]),
             _grabador(manos, lambda legal: legal[-1]), 3, 12, np.random.RandomState(0),
             deal_seed=7)
    assert len(manos) == 12
    for d in range(4):
        assert len(set(manos[3 * d:3 * d + 3])) == 1
    assert len(set(manos)) == 4

    otras = []                                     # otros agentes, misma semilla
    evaluate(game, _grabador(otras, lambda legal: legal[-1]),
             _grabador(otras, lambda legal: legal[len(legal) // 2]), 3, 12,
             np.random.RandomState(5), deal_seed=7)
    assert otras == manos


def test_error_estandar():
    game = pyspiel.load_game("podrida", {"players": 3})
    m = evaluate(game, heuristic_opponent(None), random_opponent(np.random.RandomState(1)), 3,
                 60, np.random.RandomState(0))
    o = m["overall"]
    assert 0 < o["win_rate_se"] < 0.1 and 0 < o["mean_score_diff_se"]


def test_comparar_reparto_a_reparto():
    from rotten_rl.evaluation import paired_diff
    game = pyspiel.load_game("podrida", {"players": 3})
    a = evaluate(game, heuristic_opponent(None), random_opponent(np.random.RandomState(1)), 3, 30,
                 np.random.RandomState(0), deal_seed=4)
    b = evaluate(game, random_opponent(np.random.RandomState(2)), heuristic_opponent(None), 3, 30,
                 np.random.RandomState(0), deal_seed=4)
    assert len(a["per_deal"]["diff"]) == 10
    m, se = paired_diff(a, b)
    assert m == pytest.approx(a["overall"]["mean_score_diff"] - b["overall"]["mean_score_diff"])
    assert se > 0
    assert paired_diff(a, a) == (0.0, 0.0)
    with pytest.raises(ValueError):
        paired_diff(a, evaluate(game, heuristic_opponent(None), heuristic_opponent(None), 3, 6,
                                np.random.RandomState(0), deal_seed=4))
