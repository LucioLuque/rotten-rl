"""Cortar y retomar desde un checkpoint da lo mismo que entrenar de corrido."""
import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")
torch = pytest.importorskip("torch")

from open_spiel.python.pytorch import nfsp  # noqa: E402

import rotten_rl.openspiel_game  # noqa: E402,F401
from rotten_rl.algorithms import NFSPAlgorithm  # noqa: E402
from rotten_rl.checkpoint import latest_checkpoint, list_checkpoints, load_checkpoint, save_checkpoint  # noqa: E402
from rotten_rl.training import EpisodeRunner  # noqa: E402


def _agentes(game):
    torch.manual_seed(0)
    return [nfsp.NFSP(i, game.information_state_tensor_size(), game.num_distinct_actions(), [32],
                      reservoir_buffer_capacity=500, anticipatory_param=0.5,
                      replay_buffer_capacity=300, min_buffer_size_to_learn=20, batch_size=16,
                      learn_every=8, optimizer_str="adam", epsilon_start=0.3, epsilon_end=0.01,
                      epsilon_decay_duration=1000)
            for i in range(game.num_players())]


def _seguir(game, agentes, rng, episodios):
    runner = EpisodeRunner(game, rng)
    return [runner.play(agentes) for _ in range(episodios)]


def _pesos(agentes):
    return [torch.cat([p.flatten() for p in a._avg_network.parameters()] +
                      [p.flatten() for p in a._rl_agent._q_network.parameters()])
            for a in agentes]


def test_retomar_es_igual_que_de_corrido(tmp_path):
    game = pyspiel.load_game("podrida", {"players": 2})

    # De corrido: 6 partidas + 6 partidas.
    np.random.seed(1)
    rng = np.random.RandomState(1)
    corrido = _agentes(game)
    _seguir(game, corrido, rng, 6)
    save_checkpoint(str(tmp_path), 6, NFSPAlgorithm(corrido), rng, extra={"hola": 1})
    ret_corrido = _seguir(game, corrido, rng, 6)

    # Cortado: agentes nuevos (otra semilla, otro estado del RNG) + restaurar + 6 partidas.
    np.random.seed(99)
    rng2 = np.random.RandomState(99)
    retomado = _agentes(game)
    info = load_checkpoint(latest_checkpoint(str(tmp_path)), NFSPAlgorithm(retomado), rng2)
    ret_retomado = _seguir(game, retomado, rng2, 6)

    assert info == {"episode": 6, "extra": {"hola": 1}}
    assert ret_retomado == ret_corrido
    for a, b in zip(_pesos(corrido), _pesos(retomado)):
        assert torch.equal(a, b)
    for a, b in zip(corrido, retomado):
        assert len(a._reservoir_buffer) == len(b._reservoir_buffer)
        assert a._rl_agent._iteration == b._rl_agent._iteration


def test_guarda_solo_los_ultimos(tmp_path):
    game = pyspiel.load_game("podrida", {"players": 2})
    agentes = _agentes(game)
    rng = np.random.RandomState(0)
    for ep in (1, 2, 3):
        save_checkpoint(str(tmp_path), ep, NFSPAlgorithm(agentes), rng, with_buffers=False, keep_last=2)
    assert [p.rsplit("_ep", 1)[1] for p in list_checkpoints(str(tmp_path))] == ["2", "3"]
