"""Chequeo de la regla de seguimiento de palo sobre partidas aleatorias."""
import random

import pytest

from rotten_rl.engine import PodridaGame


def _violaciones(must_trump: bool, rondas: int = 400) -> int:
    violaciones = 0
    rng = random.Random(0)
    for r in range(rondas):
        n = rng.choice([3, 4, 5, 6])
        g = PodridaGame(num_players=n, seed=r, must_trump_when_void=must_trump)
        g.start_round(num_cards=rng.randint(2, 51 // n))
        while not g.is_round_done():
            i = g.current_player_index
            if g.phase == "playing":
                mano = list(g.players[i].hand)
                lead = g.lead_suit
                legales = set(g.legal_cards(i))
                if lead is not None:
                    del_palo = {c for c in mano if c.suit == lead}
                    triunfos = {c for c in mano if c.suit == g.trump_suit}
                    if del_palo:
                        esperado = del_palo
                    elif must_trump and triunfos:
                        esperado = triunfos
                    else:
                        esperado = set(mano)
                    if legales != esperado:
                        violaciones += 1
            g.apply_action(g.random_legal_action(i), i)
    return violaciones


@pytest.mark.parametrize("must_trump", [True, False],
                         ids=["obligacion_de_fallar", "regla_estandar"])
def test_seguir_palo(must_trump):
    assert _violaciones(must_trump) == 0


def test_fallo_obligatorio():
    """Sin palo de salida pero con triunfo -> solo triunfos son legales."""
    g = PodridaGame(num_players=4, seed=2, must_trump_when_void=True)
    g.start_round(num_cards=4)
    while g.phase == "bidding":
        g.apply_action(g.random_legal_action(), g.current_player_index)
    forzados = 0
    while not g.is_round_done():
        i = g.current_player_index
        if g.lead_suit and not any(c.suit == g.lead_suit for c in g.players[i].hand):
            if any(c.suit == g.trump_suit for c in g.players[i].hand):
                assert all(c.suit == g.trump_suit for c in g.legal_cards(i))
                forzados += 1
        g.apply_action(g.random_legal_action(i), i)
    assert forzados > 0, "la semilla ya no genera ningun fallo obligatorio"
