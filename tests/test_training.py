"""El loop propio (EpisodeRunner) entrega los mismos TimeSteps que rl_environment."""
import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")

from open_spiel.python import rl_agent, rl_environment  # noqa: E402

import rotten_rl.openspiel_game as og  # noqa: E402
from rotten_rl.training import EpisodeRunner, with_pending_reward  # noqa: E402


class _Grabador(rl_agent.AbstractAgent):
    """Agente determinista que anota todo lo que ve."""

    def __init__(self, player_id):
        self.player_id = player_id
        self.visto = []
        self._k = 0

    def step(self, time_step, is_evaluation=False):
        p = self.player_id
        self.visto.append((
            time_step.last(),
            np.asarray(time_step.observations["info_state"][p], np.float32).copy(),
            list(time_step.observations["legal_actions"][p]),
            None if time_step.rewards is None else time_step.rewards[p],
        ))
        if time_step.last():
            return None
        legal = time_step.observations["legal_actions"][p]
        self._k += 1
        return rl_agent.StepOutput(action=legal[(self._k * 7) % len(legal)], probs=None)


def _con_rl_environment(game, seed):
    env = rl_environment.Environment(
        game, chance_event_sampler=rl_environment.ChanceEventSampler(seed=seed))
    n = game.num_players()
    agentes = [_Grabador(i) for i in range(n)]
    ts = env.reset()
    pending = [0.0] * n
    while not ts.last():
        p = ts.observations["current_player"]
        ts = env.step([agentes[p].step(with_pending_reward(ts, pending, p)).action])
        for i in range(n):
            pending[i] += ts.rewards[i]
    for p, a in enumerate(agentes):
        a.step(with_pending_reward(ts, pending, p))
    return agentes, env.get_state.returns()


@pytest.mark.parametrize("players", [2, 4, 7])
def test_runner_igual_a_rl_environment(players):
    game = pyspiel.load_game("podrida", {"players": players})
    esperado, ret_esperado = _con_rl_environment(game, seed=3)

    agentes = [_Grabador(i) for i in range(players)]
    ret = EpisodeRunner(game, np.random.RandomState(3)).play(agentes)

    assert ret == pytest.approx(ret_esperado)
    for a, e in zip(agentes, esperado):
        assert len(a.visto) == len(e.visto)
        for k, (x, y) in enumerate(zip(a.visto, e.visto)):
            assert x[0] == y[0] and x[2] == y[2], (a.player_id, k)
            np.testing.assert_array_equal(x[1], y[1])
            if k > 0:                          # el primer reward no cierra transicion
                assert x[3] == pytest.approx(y[3]), (a.player_id, k)


def test_info_state_tensor_igual_a_pyspiel():
    game = pyspiel.load_game("podrida", {"players": 3})
    rng = np.random.RandomState(0)
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            o, p = zip(*state.chance_outcomes())
            state.apply_action(int(rng.choice(o, p=p)))
            continue
        for pl in range(3):
            np.testing.assert_array_equal(
                og.info_state_tensor(state, pl), state.information_state_tensor(pl))
        state.apply_action(int(rng.choice(state.legal_actions())))
