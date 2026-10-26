"""Rondas como episodios, y DQN con red compartida (aprende y se retoma igual)."""
import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")
torch = pytest.importorskip("torch")

from open_spiel.python import rl_agent  # noqa: E402

import rotten_rl.openspiel_game  # noqa: E402,F401
from rotten_rl.algorithms import SharedDQN  # noqa: E402
from rotten_rl.buffers import CompactReplayBuffer  # noqa: E402
from rotten_rl.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint  # noqa: E402
from rotten_rl.training import EpisodeRunner  # noqa: E402


class _Grabador(rl_agent.AbstractAgent):
    """Agente determinista que anota (es_LAST, reward) de cada paso."""

    def __init__(self, player_id):
        self.player_id = player_id
        self.visto = []
        self._k = 0

    def step(self, time_step, is_evaluation=False):
        p = self.player_id
        self.visto.append((time_step.last(), time_step.first(), time_step.rewards[p]))
        if time_step.last():
            return None
        legal = time_step.observations["legal_actions"][p]
        self._k += 1
        return rl_agent.StepOutput(action=legal[(self._k * 5) % len(legal)], probs=None)


@pytest.mark.parametrize("players", [2, 4])
def test_rondas_como_episodios(players):
    game = pyspiel.load_game("podrida", {"players": players})
    rondas = len(game.ladder)
    agentes = [_Grabador(i) for i in range(players)]
    ret = EpisodeRunner(game, np.random.RandomState(0)).play(agentes, round_episodes=True)

    for a in agentes:
        lasts = [r for last, _, r in a.visto if last]
        assert len(lasts) == rondas               # un LAST por ronda
        # Cada ronda le entrega exactamente su reward (todo llega en el LAST).
        assert sum(lasts) == pytest.approx(ret[a.player_id])
    # Cada ronda arranca como episodio nuevo: un FIRST por ronda (lo recibe el
    # primero que decide, como en rl_environment).
    assert sum(first for a in agentes for _, first, _ in a.visto) == rondas


def _dqn(game, **kw):
    torch.manual_seed(0)
    args = dict(hidden_layers_sizes=[32], replay_buffer_capacity=500, min_buffer_size_to_learn=50,
                batch_size=16, learn_every=4, learning_rate=1e-3, discount_factor=1.0,
                epsilon_start=0.5, epsilon_end=0.05, epsilon_decay_episodes=20,
                target_update_every=50)
    args.update(kw)
    return SharedDQN(game, **args)


def _pesos(algo):
    return torch.cat([p.detach().flatten() for p in algo.q.parameters()])


def _entrenar(game, algo, rng, episodios, start=0):
    runner = EpisodeRunner(game, rng)
    out = []
    for ep in range(start + 1, start + episodios + 1):
        algo.begin_episode(ep)
        out.append(runner.play(algo.seats(), round_episodes=True))
    return out


def test_dqn_aprende_con_una_red_compartida():
    game = pyspiel.load_game("podrida", {"players": 3})
    np.random.seed(0)
    algo = _dqn(game)
    antes = _pesos(algo).clone()
    _entrenar(game, algo, np.random.RandomState(0), 6)
    assert isinstance(algo.buffer, CompactReplayBuffer)
    assert len(algo.buffer) > 200                 # los 3 asientos llenan el mismo buffer
    assert algo.last_loss is not None and np.isfinite(algo.last_loss)
    assert not torch.equal(antes, _pesos(algo))
    assert algo.epsilon < 0.5


def test_dqn_retomar_es_igual_que_de_corrido(tmp_path):
    game = pyspiel.load_game("podrida", {"players": 2})
    np.random.seed(1)
    rng = np.random.RandomState(1)
    corrido = _dqn(game)
    _entrenar(game, corrido, rng, 5)
    save_checkpoint(str(tmp_path), 5, corrido, rng)
    ret_corrido = _entrenar(game, corrido, rng, 5, start=5)

    np.random.seed(99)
    rng2 = np.random.RandomState(99)
    torch.manual_seed(99)
    retomado = _dqn(game)
    info = load_checkpoint(latest_checkpoint(str(tmp_path)), retomado, rng2)
    ret_retomado = _entrenar(game, retomado, rng2, 5, start=info["episode"])

    assert ret_retomado == ret_corrido
    assert torch.equal(_pesos(corrido), _pesos(retomado))
    assert retomado.steps == corrido.steps


def _ppo(game, **kw):
    torch.manual_seed(0)
    from rotten_rl.algorithms import SharedPPO
    args = dict(hidden_layers_sizes=[32], learning_rate=1e-3, discount_factor=1.0,
                rollout_episodes=3, epochs=2, minibatch=64)
    args.update(kw)
    return SharedPPO(game, **args)


def test_ppo_aprende_con_politica_compartida():
    game = pyspiel.load_game("podrida", {"players": 3})
    np.random.seed(0)
    algo = _ppo(game)
    antes = torch.cat([p.detach().flatten() for p in algo.policy.parameters()]).clone()
    _entrenar(game, algo, np.random.RandomState(0), 7)
    assert algo.updates == 2                       # cada 3 partidas
    assert np.isfinite(algo.last_stats["loss_policy"])
    assert not torch.equal(antes, torch.cat([p.detach().flatten()
                                             for p in algo.policy.parameters()]))


def test_ppo_retomar_es_igual_que_de_corrido(tmp_path):
    game = pyspiel.load_game("podrida", {"players": 2})
    np.random.seed(1)
    torch.manual_seed(1)
    rng = np.random.RandomState(1)
    corrido = _ppo(game)
    _entrenar(game, corrido, rng, 4)               # corta con un lote a medio juntar
    save_checkpoint(str(tmp_path), 4, corrido, rng)
    ret_corrido = _entrenar(game, corrido, rng, 5, start=4)

    np.random.seed(99)
    torch.manual_seed(99)
    rng2 = np.random.RandomState(99)
    retomado = _ppo(game)
    info = load_checkpoint(latest_checkpoint(str(tmp_path)), retomado, rng2)
    ret_retomado = _entrenar(game, retomado, rng2, 5, start=info["episode"])

    assert ret_retomado == ret_corrido
    assert retomado.updates == corrido.updates
    for a, b in zip(corrido.policy.parameters(), retomado.policy.parameters()):
        assert torch.equal(a, b)
