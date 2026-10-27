"""El solver en C++ de la ronda de 2 cartas (rotten_rl/exact/round2) juega con
las mismas reglas que el motor de Python: mismos cantos y cartas permitidos,
mismos puntajes, y su heuristica decide lo mismo que la de evaluation.py."""
import json
import shutil
import types

import pytest

from rotten_rl.engine import RANKS, SUITS, Card, PodridaGame
from rotten_rl.evaluation import heuristic_opponent

if not (shutil.which("g++") and shutil.which("make")):
    pytest.skip("hace falta g++ y make", allow_module_level=True)

from rotten_rl.exact import round2  # noqa: E402


def _card(c):
    return Card(RANKS[c % 13], SUITS[c // 13])


def _motor(n, trump, hands):
    """El motor en la ronda de 2 cartas: la posicion 0 canta primero y sale; la
    ultima reparte."""
    eng = PodridaGame(num_players=n, seed=0)
    for p in eng.players:
        p.reset_for_round()
    eng.cards_per_player = 2
    eng.played_cards, eng.current_trick, eng.completed_tricks = [], [], []
    eng.trick_number, eng.discard_pile, eng.deck = 0, [], []
    for p, h in enumerate(hands):
        eng.players[p].hand = [_card(c) for c in h]
    eng.trump_card = _card(trump)
    eng.trump_suit = eng.trump_card.suit
    eng.dealer_index = n - 1
    eng.first_player_index = eng.current_player_index = 0
    eng.phase = "bidding"
    return eng


@pytest.mark.parametrize("n", [2, 3, 4, 7])
def test_mismas_reglas_y_heuristica_que_el_motor(n):
    h = heuristic_opponent(None)
    for line in round2.run("trace", "--n", n, "--games", 300, "--seed", n).splitlines():
        r = json.loads(line)
        eng = _motor(n, r["trump"], r["hands"])
        estado = types.SimpleNamespace(_engine=eng)
        for k, b in enumerate(r["bids"]):
            assert eng.current_player_index == k
            legal = sorted(eng.legal_action_ids(k))
            assert legal == [52 + x for x in b["legal"]]
            assert h(estado, legal) == 52 + b["heuristic"]
            eng.apply_action(52 + b["chosen"], k)
        for k, p in enumerate(r["plays"]):
            assert eng.current_player_index == k
            legal = sorted(eng.legal_action_ids(k))
            assert legal == p["legal"]
            assert h(estado, legal) == p["heuristic"]
            eng.apply_action(p["chosen"], k)
        # La segunda baza es forzada: cada uno juega la carta que le queda.
        while eng.phase != "done":
            idx = eng.current_player_index
            eng.apply_action(eng.legal_action_ids(idx)[0], idx)
        assert [p.last_round_score for p in eng.players] == r["scores"]


def test_resolver_y_evaluar(tmp_path):
    """Una corrida corta: aprende algo que le gana a cantar y jugar al azar, y
    se puede cortar y seguir."""
    tabla = tmp_path / "t.bin"
    round2.run("solve", "--n", 3, "--iters", 20000, "--save", tabla)
    round2.run("solve", "--n", 3, "--iters", 20000, "--load", tabla, "--save", tabla)
    vs_azar = json.loads(round2.run("eval", "--n", 3, "--games", 50000, "--load", tabla,
                                    "--policies", "avg,uniform,uniform"))
    assert vs_azar["value"][0] > 5 * vs_azar["se"][0]
    assert vs_azar["miss_rate"] < 0.2


def test_mejor_respuesta_y_exportar(tmp_path):
    tabla, tsv = tmp_path / "t.bin", tmp_path / "e.tsv"
    round2.run("solve", "--n", 3, "--iters", 5000, "--save", tabla)
    br = json.loads(round2.run("br", "--n", 3, "--games", 2000, "--load", tabla))
    assert len(br["gain_in_sample"]) == len(br["gain_holdout"]) == 3
    assert all(g >= 0 for g in br["gain_in_sample"])         # elegir lo mejor nunca pierde
    round2.run("dump", "--n", 3, "--load", tabla, "--save", tsv)
    filas = [line.split("\t") for line in tsv.read_text().splitlines()]
    assert {f[0] for f in filas} == {"bid", "play"}
    for f in filas:
        probs = [float(x) for x in (f[6:9] if f[0] == "bid" else f[8:10])]
        assert sum(probs) == pytest.approx(1.0, abs=1e-3)
