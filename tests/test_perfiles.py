import json

import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")

import rotten_rl.openspiel_game as og  # noqa: E402
from rotten_rl.perfiles.partida import (HumanMatch, SessionLog, SpotSession, next_deal,  # noqa: E402
                                        replay, sample_chance)
from rotten_rl.perfiles.reglas import PROFILES  # noqa: E402
from rotten_rl.perfiles.situaciones import (LearnedProfile, agreement, load_answers,  # noqa: E402
                                             pick_spots, play_options, spot_keys)


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


def test_elige_situaciones_con_opciones_reales_y_variadas():
    game = pyspiel.load_game("podrida", {"players": 2})
    spots = pick_spots(game, 40, games=40)
    assert len(spots) == 40
    estados = [replay(game, h) for h in spots]
    assert all(len(s.legal_actions()) > 1 for s in estados)
    assert len({s._engine.phase for s in estados}) == 2
    assert len({spot_keys(s)[1] for s in estados}) > 15


def test_una_respuesta_se_generaliza_a_manos_parecidas():
    game = pyspiel.load_game("podrida", {"players": 2})
    rng = np.random.RandomState(0)
    states = []
    while len(states) < 200:
        s = game.new_initial_state()
        while not s.is_terminal():
            if s.is_chance_node():
                sample_chance(s, rng)
                continue
            legal = s.legal_actions()
            if s._engine.phase == "playing" and s._engine.current_trick:
                states.append(s.clone())
            s.apply_action(int(rng.choice(legal)))
    # Siempre gana con la carta más baja que gana, o pierde con la más alta.
    answers = []
    for s in states[:150]:
        options = play_options(s, s.legal_actions())
        label = "gana_baja" if "gana_baja" in options else "pierde_alta"
        answers.append((*spot_keys(s), [label]))
    profile = LearnedProfile(answers)
    for s in states[150:]:
        legal = s.legal_actions()
        options = play_options(s, legal)
        expected = options.get("gana_baja", options.get("pierde_alta"))
        if profile.predict(s, legal) is not None:
            assert profile.predict(s, legal) == expected
    assert agreement(answers, LearnedProfile) > 0.95


def test_el_canto_se_aprende_como_diferencia_con_la_estimacion(tmp_path):
    game = pyspiel.load_game("podrida", {"players": 2})
    log = SessionLog(str(tmp_path), "lucio", "prueba")
    rng = np.random.RandomState(0)
    spots = pick_spots(game, 60, games=60)
    sesion = SpotSession(game, spots, log)
    while not sesion.finished:
        legal = sesion.state.legal_actions()
        accion = legal[0] if sesion.engine.phase == "playing" else min(legal)
        assert sesion.apply(accion)
    respuestas = load_answers(game, str(tmp_path))
    assert len(respuestas) == 60
    profile = LearnedProfile(respuestas)
    state = replay(game, spots[0])
    assert profile(state, state.legal_actions()) in state.legal_actions()
