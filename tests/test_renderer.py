import os

import pytest

pygame = pytest.importorskip("pygame")
pyspiel = pytest.importorskip("pyspiel")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import rotten_rl.openspiel_game  # noqa: E402,F401
from rotten_rl.profiles.match import HumanMatch  # noqa: E402
from rotten_rl.renderer import PodridaPygameRenderer  # noqa: E402


def _first(state, legal):
    return legal[0]


@pytest.mark.parametrize("players", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("size", [(1920, 1080), (1366, 768), (1100, 650)])
def test_trick_cards_do_not_overlap(players, size):
    game = pyspiel.load_game("podrida", {"players": players})
    match = HumanMatch(game, _first, 0, 0, 0)
    renderer = PodridaPygameRenderer(render_mode="rgb_array", width=size[0], height=size[1])
    renderer.render(match.engine, perspective_player=0)
    seats = renderer._seat_layout(match.engine, 0)
    rects = list(renderer._trick_slots(match.engine, seats).values())
    assert len(rects) == players
    for i, a in enumerate(rects):
        assert not any(a.colliderect(b) for b in rects[i + 1:])
        assert not any(a.colliderect(s.block) for s in seats if s.visible)


def test_position_counts_from_who_starts():
    game = pyspiel.load_game("podrida", {"players": 3})
    match = HumanMatch(game, _first, 0, 0, 0)
    eng = match.engine
    first = eng.first_player_index
    assert PodridaPygameRenderer._position_label(eng, first) == "1º de 3"
    assert PodridaPygameRenderer._position_label(eng, (first + 2) % 3) == "3º de 3"
