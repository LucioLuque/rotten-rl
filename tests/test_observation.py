"""El info-state sigue siendo exactamente el mismo que el guardado en
tests/data/info_states.npz (generado con scripts/dev/make_obs_fixture.py antes de
optimizar el armado de la observacion)."""
import importlib.util
import os

import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")

import rotten_rl.openspiel_game as og  # noqa: E402

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_spec = importlib.util.spec_from_file_location(
    "make_obs_fixture", os.path.join(_ROOT, "scripts", "dev", "make_obs_fixture.py"))
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)

_DATA = np.load(os.path.join(_ROOT, "tests", "data", "info_states.npz"))


@pytest.mark.parametrize("players", range(2, 8))
def test_info_state_igual_al_guardado(players):
    tensors = []
    fixture.play(players, players, lambda s, p: tensors.append(og.info_state_tensor(s, p)))
    np.testing.assert_array_equal(np.stack(tensors), _DATA[f"p{players}"])


def test_info_state_igual_por_pyspiel():
    """El camino de pyspiel (information_state_tensor) y el rapido coinciden,
    tambien en clones (el estado incremental se copia bien)."""
    def check(state, player):
        clone = state.clone()
        expected = og.info_state_tensor(state, player)
        np.testing.assert_array_equal(np.asarray(state.information_state_tensor(player)),
                                      expected)
        np.testing.assert_array_equal(og.info_state_tensor(clone, player), expected)
    fixture.play(4, 0, check)


def _jugar_con_deduccion(players, partidas, revisar):
    """Partidas al azar con la deduccion activada, llamando a `revisar` en cada decision."""
    game = pyspiel.load_game("podrida", {"players": players, "infer_trump_void": True})
    rng = np.random.RandomState(players)
    for _ in range(partidas):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(outcomes, p=probs)))
                continue
            revisar(state)
            state.apply_action(int(rng.choice(state.legal_actions())))


@pytest.mark.parametrize("players", [3, 7])
def test_deduccion_de_triunfo_siempre_es_cierta(players):
    vistos = []

    def revisar(state):
        eng = state._engine
        for j, s in state._trump_voids:
            assert not any(og._SUIT_ID[c.suit] == s for c in eng.players[j].hand)
            assert s == og._SUIT_ID[eng.trump_suit]
            vistos.append(j)

    _jugar_con_deduccion(players, 8, revisar)
    assert vistos


def test_deduccion_solo_cambia_los_palos_vacios():
    desde = og._OFF["void_suits"]
    hasta = desde + 7 * 4
    distintas = []

    def revisar(state):
        p = state.current_player()
        con = og.info_state_tensor(state, p)
        sin = og.info_state_tensor(state, p, False)
        assert con.shape == sin.shape == (808,)
        np.testing.assert_array_equal(np.delete(con, range(desde, hasta)),
                                      np.delete(sin, range(desde, hasta)))
        assert (con[desde:hasta] >= sin[desde:hasta]).all()
        distintas.append(not np.array_equal(con, sin))

    _jugar_con_deduccion(3, 8, revisar)
    assert any(distintas)
