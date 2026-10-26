"""Los buffers comprimidos recuperan exactamente lo mismo y ocupan ~14 veces menos."""
import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")
torch = pytest.importorskip("torch")

from open_spiel.python.pytorch import dqn, nfsp  # noqa: E402

import rotten_rl.openspiel_game as og  # noqa: E402
from rotten_rl.buffers import (CompactReplayBuffer, CompactReservoirBuffer,  # noqa: E402
                               ENCODED_SIZE, INFO_SIZE, decode_info_states, encode_info_states,
                               use_compact_buffers)
from rotten_rl.algorithms import NFSPAlgorithm  # noqa: E402
from rotten_rl.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint  # noqa: E402
from rotten_rl.training import EpisodeRunner  # noqa: E402


def _info_states(players, partidas=3):
    game = pyspiel.load_game("podrida", {"players": players})
    rng = np.random.RandomState(players)
    out = []
    for _ in range(partidas):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                o, p = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(o, p=p)))
                continue
            for pl in range(players):
                out.append(og.info_state_tensor(state, pl))
            state.apply_action(int(rng.choice(state.legal_actions())))
    return np.stack(out)


@pytest.mark.parametrize("players", [2, 4, 7])
def test_codificar_y_decodificar_es_exacto(players):
    x = _info_states(players)
    e = encode_info_states(x)
    assert e.dtype == np.uint8 and e.shape == (len(x), ENCODED_SIZE)
    np.testing.assert_array_equal(decode_info_states(e), x)
    np.testing.assert_array_equal(decode_info_states(encode_info_states(x[0])), x[0])
    assert INFO_SIZE * 4 / ENCODED_SIZE > 14


def test_los_bloques_binarios_son_siempre_0_o_1():
    from rotten_rl.buffers import _BIN_IDX
    x = _info_states(5, partidas=5)[:, _BIN_IDX]
    assert set(np.unique(x)) <= {0.0, 1.0}


def _agentes(game, compact):
    torch.manual_seed(0)
    agentes = []
    for i in range(game.num_players()):
        extra = {"replay_buffer_class": CompactReservoirBuffer} if compact else {}
        a = nfsp.NFSP(i, game.information_state_tensor_size(), game.num_distinct_actions(), [32],
                      reservoir_buffer_capacity=500, anticipatory_param=0.5,
                      replay_buffer_capacity=300, min_buffer_size_to_learn=20, batch_size=16,
                      learn_every=8, optimizer_str="adam", epsilon_start=0.3, epsilon_end=0.01,
                      epsilon_decay_duration=1000, **extra)
        if compact:
            use_compact_buffers(a)
        agentes.append(a)
    return agentes


def _pesos(agentes):
    return [torch.cat([p.flatten() for p in a._avg_network.parameters()] +
                      [p.flatten() for p in a._rl_agent._q_network.parameters()])
            for a in agentes]


def _entrenar(compact, episodios=8):
    game = pyspiel.load_game("podrida", {"players": 2})
    np.random.seed(1)
    rng = np.random.RandomState(1)
    agentes = _agentes(game, compact)
    ret = [EpisodeRunner(game, rng).play(agentes) for _ in range(episodios)]
    return game, agentes, rng, ret


def test_entrenar_comprimido_da_lo_mismo():
    _, normal, _, ret_normal = _entrenar(False)
    _, comp, _, ret_comp = _entrenar(True)
    assert ret_comp == ret_normal
    for a, b in zip(_pesos(normal), _pesos(comp)):
        assert torch.equal(a, b)
    a, b = normal[0], comp[0]
    assert isinstance(b._reservoir_buffer, CompactReservoirBuffer)
    assert isinstance(b._rl_agent._replay_buffer, CompactReplayBuffer)
    bytes_normal = sum(v.nbytes for v in a._reservoir_buffer.experience)
    bytes_comp = sum(v.nbytes for v in b._reservoir_buffer.experience)
    assert bytes_normal / bytes_comp > 6


def test_checkpoint_con_buffers_comprimidos(tmp_path):
    game, agentes, rng, _ = _entrenar(True, episodios=6)
    save_checkpoint(str(tmp_path), 6, NFSPAlgorithm(agentes), rng)
    sigue = [EpisodeRunner(game, rng).play(agentes) for _ in range(4)]

    np.random.seed(7)
    rng2 = np.random.RandomState(7)
    retomados = _agentes(game, True)
    load_checkpoint(latest_checkpoint(str(tmp_path)), NFSPAlgorithm(retomados), rng2)
    assert [EpisodeRunner(game, rng2).play(retomados) for _ in range(4)] == sigue
    for a, b in zip(_pesos(agentes), _pesos(retomados)):
        assert torch.equal(a, b)

    # Un checkpoint comprimido no se puede cargar en agentes con buffers comunes.
    with pytest.raises(ValueError, match="comprimidos"):
        load_checkpoint(latest_checkpoint(str(tmp_path)), NFSPAlgorithm(_agentes(game, False)), rng2)
