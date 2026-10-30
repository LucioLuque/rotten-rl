import json

import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")

import rotten_rl.openspiel_game as og  # noqa: E402
from rotten_rl.profiles.match import (HumanMatch, SessionLog, SpotSession, next_deal,  # noqa: E402
                                        replay, sample_chance)
from rotten_rl.profiles.rules import PROFILES  # noqa: E402
from rotten_rl.profiles.spots import (LearnedProfile, agreement, answered,  # noqa: E402
                                             load_answers, pick_spots, play_options, spot_keys)


def _first(state, legal):
    return legal[0]


def _play(match, choose):
    """Plays the whole match, with `choose` on the human turns."""
    while not match.finished:
        if match.round_over:
            match.next_round()
        elif match.human_turn:
            assert match.apply(choose(match.state.legal_actions()))
        else:
            assert match.step_rival()


def _hands_per_round(game, choose):
    match = HumanMatch(game, _first, 0, 3, 5)
    hands = []
    while not match.finished:
        if match.round_over:
            match.next_round()
            continue
        if match.engine.phase == "bidding" and all(p.bid is None for p in match.engine.players):
            hands.append([sorted(str(c) for c in p.hand) for p in match.engine.players])
        if match.human_turn:
            match.apply(choose(match.state.legal_actions()))
        else:
            match.step_rival()
    return hands


def test_deal_does_not_depend_on_plays():
    game = pyspiel.load_game("podrida", {"players": 2})
    one = _hands_per_round(game, lambda legal: legal[0])
    other = _hands_per_round(game, lambda legal: legal[-1])
    assert len(one) == 13 and one == other


def test_saved_decision_rebuilds_state(tmp_path):
    game = pyspiel.load_game("podrida", {"players": 2})
    log = SessionLog(str(tmp_path), "lucio", "test")
    match = HumanMatch(game, _first, 1, 0, 0, log)
    _play(match, lambda legal: legal[-1])
    with open(log.path) as f:
        rows = [json.loads(line) for line in f]
    assert rows[0]["type"] == "match" and rows[-1]["type"] == "end"
    decisions = [f for f in rows if f["type"] == "decision"]
    assert len(decisions) == 62                 # 49 cartas y 13 bids del humano
    for d in decisions:
        state = replay(game, d["history"])
        assert state.current_player() == 1
        assert state.legal_actions() == d["legal"]
        assert d["action"] in d["legal"]
        assert len(og.info_state_tensor(state, 1)) == 808


def test_rejects_illegal_actions_and_moves_to_next_deal(tmp_path):
    game = pyspiel.load_game("podrida", {"players": 2})
    assert next_deal(str(tmp_path / "nada"), 0) == 0
    log = SessionLog(str(tmp_path), "lucio", "test")
    match = HumanMatch(game, _first, 0, 7, 4, log)
    assert not match.apply(0 if 0 not in match.state.legal_actions() else 51 + 60)
    assert next_deal(str(tmp_path), 7) == 5
    assert next_deal(str(tmp_path), 8) == 0


def test_profiles_play_legally_and_bid_differently():
    game = pyspiel.load_game("podrida", {"players": 3})
    bids = {}
    for name, make in PROFILES.items():
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
        bids[name] = total
    assert bids["play_to_lose"] < bids["conservative"] < bids["risky"]


def test_picks_varied_spots_with_real_options():
    game = pyspiel.load_game("podrida", {"players": 2})
    spots = pick_spots(game, 40, games=40)
    assert len(spots) == 40
    states = [replay(game, h) for h in spots]
    assert all(len(s.legal_actions()) > 1 for s in states)
    assert len({s._engine.phase for s in states}) == 2
    assert len({spot_keys(s)[2] for s in states}) > 15


def test_asking_for_more_spots_skips_answered_ones():
    game = pyspiel.load_game("podrida", {"players": 2})
    first_spots = pick_spots(game, 30, games=40)
    more_spots = pick_spots(game, 30, games=40, skip=first_spots)
    assert not {tuple(h) for h in first_spots} & {tuple(h) for h in more_spots}


def test_spot_type_looks_at_match_and_cards():
    game = pyspiel.load_game("podrida", {"players": 2})
    rng = np.random.RandomState(0)
    seen = set()
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            sample_chance(state, rng)
            continue
        general, detailed, full = spot_keys(state)
        assert full[:len(detailed)] == detailed and detailed[:len(general)] == general
        seen.add(full[len(detailed):])
        state.apply_action(int(rng.choice(state.legal_actions())))
    assert len(seen) > 3


def test_learned_profile_prefers_most_detailed_type():
    game = pyspiel.load_game("podrida", {"players": 2})
    state = replay(game, pick_spots(game, 1, games=20)[0])
    while state._engine.phase != "playing" or not play_options(state, state.legal_actions()):
        state.apply_action(state.legal_actions()[0])
    general, detailed, full = spot_keys(state)
    options = play_options(state, state.legal_actions())
    one, other = (list(options) * 2)[:2]
    profile = LearnedProfile([(general, ("other",), ("other", "mas"), [one]),
                              (general, detailed, full, [other])])
    assert profile.predict(state, state.legal_actions()) == options[other]


def test_una_respuesta_se_generaliza_a_hands_parecidas():
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
    # Always wins with the lowest winning card, or loses with the highest one.
    answers = []
    for s in states[:150]:
        options = play_options(s, s.legal_actions())
        label = "win_low" if "win_low" in options else "lose_high"
        answers.append((*spot_keys(s), [label]))
    profile = LearnedProfile(answers)
    for s in states[150:]:
        legal = s.legal_actions()
        options = play_options(s, legal)
        expected = options.get("win_low", options.get("lose_high"))
        if profile.predict(s, legal) is not None:
            assert profile.predict(s, legal) == expected
    assert agreement(answers, LearnedProfile) > 0.95


def test_bid_is_learned_as_offset_from_estimate(tmp_path):
    game = pyspiel.load_game("podrida", {"players": 2})
    log = SessionLog(str(tmp_path), "lucio", "test")
    rng = np.random.RandomState(0)
    spots = pick_spots(game, 60, games=60)
    session = SpotSession(game, spots, log)
    while not session.finished:
        legal = session.state.legal_actions()
        action = legal[0] if session.engine.phase == "playing" else min(legal)
        assert session.apply(action)
    answers = load_answers(game, str(tmp_path))
    assert len(answers) == 60 and answered(str(tmp_path))
    profile = LearnedProfile(answers)
    state = replay(game, spots[0])
    assert profile(state, state.legal_actions()) in state.legal_actions()
