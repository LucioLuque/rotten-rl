"""
Guarda los info-states de partidas fijas (2 a 7 jugadores) en
tests/data/info_states.npz, para que tests/test_observation.py compruebe que
el armado de la observacion sigue dando exactamente lo mismo despues de
optimizarlo. Regenerarlo solo si cambia a proposito la observacion o el motor.

    python scripts/dev/make_obs_fixture.py
"""
import os
import sys

import numpy as np
import pyspiel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import rotten_rl.openspiel_game as og  # noqa: E402

GAMES_PER_SIZE = 2


def play(players, seed, fn):
    """Juega GAMES_PER_SIZE partidas al azar con semilla fija y llama a
    fn(state, player) en cada decision."""
    game = pyspiel.load_game("podrida", {"players": players})
    rng = np.random.RandomState(seed)
    for _ in range(GAMES_PER_SIZE):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(outcomes, p=probs)))
                continue
            player = state.current_player()
            # Tambien la vista de otro jugador (no solo la de quien mueve).
            fn(state, player)
            fn(state, (player + 1) % players)
            state.apply_action(int(rng.choice(state.legal_actions())))


def main():
    out = {}
    for players in range(2, 8):
        tensors = []
        play(players, players, lambda s, p: tensors.append(og.info_state_tensor(s, p)))
        out[f"p{players}"] = np.stack(tensors)
        print(players, out[f"p{players}"].shape)
    path = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "info_states.npz")
    np.savez_compressed(path, **out)


if __name__ == "__main__":
    main()
