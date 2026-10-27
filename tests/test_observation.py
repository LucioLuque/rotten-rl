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
