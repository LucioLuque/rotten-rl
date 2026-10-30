import json

import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")

import rotten_rl.openspiel_game as og  # noqa: E402
from rotten_rl.perfiles.partida import (HumanMatch, SessionLog, next_deal, replay,  # noqa: E402
                                        sample_chance)
from rotten_rl.perfiles.reglas import PROFILES  # noqa: E402


def _primero(state, legal):
    return legal[0]


def _jugar(match, elegir):
    """Juega la partida entera, con `elegir` en los turnos del humano."""
    while not match.finished:
        if match.round_over:
            match.next_round()
        elif match.human_turn:
            assert match.apply(elegir(match.state.legal_actions()))
        else:
            assert match.step_rival()


def _manos_de_cada_ronda(game, elegir):
    match = HumanMatch(game, _primero, 0, 3, 5)
    manos = []
    while not match.finished:
        if match.round_over:
            match.next_round()
            continue
        if match.engine.phase == "bidding" and all(p.bid is None for p in match.engine.players):
            manos.append([sorted(str(c) for c in p.hand) for p in match.engine.players])
        if match.human_turn:
            match.apply(elegir(match.state.legal_actions()))
        else:
            match.step_rival()
    return manos


def test_el_reparto_no_depende_de_las_jugadas():
    game = pyspiel.load_game("podrida", {"players": 2})
    uno = _manos_de_cada_ronda(game, lambda legal: legal[0])
    otro = _manos_de_cada_ronda(game, lambda legal: legal[-1])
    assert len(uno) == 13 and uno == otro


def test_la_decision_guardada_reconstruye_el_estado(tmp_path):
    game = pyspiel.load_game("podrida", {"players": 2})
    log = SessionLog(str(tmp_path), "lucio", "prueba")
    match = HumanMatch(game, _primero, 1, 0, 0, log)
    _jugar(match, lambda legal: legal[-1])
    with open(log.path) as f:
        filas = [json.loads(line) for line in f]
    assert filas[0]["tipo"] == "partida" and filas[-1]["tipo"] == "fin"
    decisiones = [f for f in filas if f["tipo"] == "decision"]
    assert len(decisiones) == 62                 # 49 cartas y 13 cantos del humano
    for d in decisiones:
        state = replay(game, d["historia"])
        assert state.current_player() == 1
        assert state.legal_actions() == d["legales"]
        assert d["accion"] in d["legales"]
        assert len(og.info_state_tensor(state, 1)) == 808


def test_no_acepta_acciones_ilegales_y_sigue_con_el_proximo_reparto(tmp_path):
    game = pyspiel.load_game("podrida", {"players": 2})
    assert next_deal(str(tmp_path / "nada"), 0) == 0
    log = SessionLog(str(tmp_path), "lucio", "prueba")
    match = HumanMatch(game, _primero, 0, 7, 4, log)
    assert not match.apply(0 if 0 not in match.state.legal_actions() else 51 + 60)
    assert next_deal(str(tmp_path), 7) == 5
    assert next_deal(str(tmp_path), 8) == 0


def test_los_perfiles_juegan_legal_y_cantan_distinto():
    game = pyspiel.load_game("podrida", {"players": 3})
    cantos = {}
    for nombre, make in PROFILES.items():
        act = make(np.random.RandomState(0))
        rng = np.random.RandomState(1)
        total = 0
        for _ in range(5):
            state = game.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    sample_chance(state, rng)
                    continue
                legal = state.legal_actions()
                action = act(state, legal)
                assert action in legal
                total += max(action - 52, 0)
                state.apply_action(action)
        cantos[nombre] = total
    assert cantos["a_perder"] < cantos["conservador"] < cantos["arriesgado"]
