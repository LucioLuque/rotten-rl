"""La ronda de 1 carta resuelta (rotten_rl/round1.py)."""
import numpy as np
import pytest

from rotten_rl.exact import round1
from rotten_rl.engine import RANKS, SUITS, Card, PodridaGame


def _card(c):
    return Card(RANKS[c % 13], SUITS[c // 13])


@pytest.mark.parametrize("n", [2, 4, 7])
def test_ganador_igual_que_el_motor(n):
    deals = round1.sample_deals(n, 300, np.random.RandomState(n))
    eng = PodridaGame(num_players=n, seed=0)
    for trump, cards, winner, types in zip(deals["trump"], deals["cards"], deals["winner"],
                                           deals["types"]):
        eng.trump_suit = SUITS[trump // 13]
        assert eng.resolve_trick([(i, _card(c)) for i, c in enumerate(cards)]) == winner
        for c, t in zip(cards, types):
            assert t == c % 13 + 13 * (c // 13 == trump // 13)


def test_dos_jugadores_el_primero_canta_1():
    """Con 2 jugadores el segundo esta obligado a copiar al primero, y al
    primero le conviene cantar 1 con cualquier carta (gana la baza con p >= 0.52)."""
    n = 2
    tables = round1.solve(n, iterations=300, deals_per_iteration=5000,
                          rng=np.random.RandomState(0))
    valid = np.ones((13, 26), bool)
    valid[np.arange(13), 13 + np.arange(13)] = False       # triunfo del numero dado vuelta
    assert (tables[0].reshape(13, 26)[valid] > 0.9).mean() > 0.95
    siempre_1 = [np.ones_like(t) for t in tables]
    deals = round1.sample_deals(n, 200000, np.random.RandomState(1))
    m = round1.evaluate(n, siempre_1, deals)
    assert m["gain_best"][0] == pytest.approx(0.0, abs=0.005)  # nada mejor que cantar 1
    assert m["gain_best"][1] == 0.0                           # el segundo no elige
    # Valor del primero: 12 si gana la baza y -12 si no (los dos cantaron 1).
    assert m["value"][0] == pytest.approx(-m["value"][1])
    assert m["value"][0] == pytest.approx(np.mean(np.where(deals["winner"] == 0, 12, -12)))


def test_cfr_se_acerca_al_equilibrio():
    n = 3
    deals = round1.sample_deals(n, 100000, np.random.RandomState(1))
    antes = round1.evaluate(n, round1.uniform_tables(n), deals)
    tables = round1.solve(n, iterations=300, deals_per_iteration=5000,
                          rng=np.random.RandomState(0))
    despues = round1.evaluate(n, tables, deals)
    assert sum(despues["gain_best"]) < 0.1 * sum(antes["gain_best"])
    assert min(despues["gain_best"]) >= 0
    assert sum(despues["value"]) == pytest.approx(0.0, abs=1e-6)   # suma cero
    for m in (antes, despues):
        for best, card in zip(m["gain_best"], m["gain_card_only"]):
            assert best >= card - 1e-9          # ver los cantos nunca es peor


@pytest.mark.parametrize("n", [3, 5])
def test_tablas_de_la_heuristica_y_de_un_agente(n):
    """situation_state arma la situacion pedida: la heuristica de evaluation.py
    canta en esos estados lo mismo que heuristic_tables, y policy_tables
    reconstruye sus tablas."""
    pyspiel = pytest.importorskip("pyspiel")
    import rotten_rl.openspiel_game  # noqa: F401
    from rotten_rl.evaluation import heuristic_opponent
    game = pyspiel.load_game("podrida", {"players": n})
    h = heuristic_opponent(None)

    def prob_bid1(states):
        out = []
        for s in states:
            eng = s._engine
            assert eng.phase == "bidding" and eng.cards_per_player == 1
            out.append(float(h(s, s.legal_actions()) == 53))
        return np.array(out)
    agente = round1.policy_tables(game, prob_bid1)
    heur = round1.heuristic_tables(n)
    for k in range(n):
        forced = round1._forced(n, k)
        for tr in range(13):
            for t in range(26):
                if t == 13 + tr:
                    continue
                for p in range(2 ** k):
                    if forced[p] < 0:
                        i = (tr * 26 + t) * 2 ** k + p
                        assert agente[k][i] == heur[k][i], (k, tr, t, p)
