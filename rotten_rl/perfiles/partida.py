"""
rotten_rl/perfiles/partida.py  —  Una partida con un jugador humano, sin nada de pantalla.

Corre sobre el estado de OpenSpiel, así el rival puede ser cualquier cosa con la forma
`act(state, legal)`: una red, la heurística o un perfil. Los repartos salen de `reparto_seed` y el
número de reparto, igual que en la evaluación duplicada (`evaluation.evaluate`) y sin depender de
las jugadas: el mismo reparto se le puede dar después a un modelo en el mismo asiento y comparar.

Cada decisión del humano se guarda en un .jsonl con la historia completa de la partida (`state.
history()`), que alcanza para reconstruir el estado y sacar la observación que haga falta.
"""

import json
import os
import time

import numpy as np

_BID = 52


def sample_chance(state, rng):
    outcomes, probs = zip(*state.chance_outcomes())
    state.apply_action(int(rng.choice(outcomes, p=probs)))


def next_deal(folder, reparto_seed):
    """El primer reparto de `reparto_seed` que todavía no se jugó en las sesiones de `folder`."""
    if not os.path.isdir(folder):
        return 0
    played = [-1]
    for fname in os.listdir(folder):
        with open(os.path.join(folder, fname)) as f:
            for line in f:
                row = json.loads(line)
                if row["tipo"] == "partida" and row["reparto_seed"] == reparto_seed:
                    played.append(row["reparto"])
    return max(played) + 1


class SessionLog:
    """Un archivo por sesión. Escribe línea por línea: si se cierra el juego no se pierde nada."""

    def __init__(self, folder, perfil, rival):
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(folder, f"{stamp}.jsonl")
        self.perfil, self.rival = perfil, rival
        self.file = open(self.path, "a")

    def write(self, tipo, **row):
        self.file.write(json.dumps({"tipo": tipo, **row}, ensure_ascii=False) + "\n")
        self.file.flush()


class HumanMatch:
    """Una partida entera. El humano mueve con `apply`; el rival, con `step_rival`. Al cerrar una
    ronda queda esperando `next_round`, para poder mostrar cómo terminó."""

    def __init__(self, game, rival, seat, reparto_seed, reparto, log=None):
        self.game, self.rival, self.seat = game, rival, seat
        self.reparto_seed, self.reparto = reparto_seed, reparto
        self.rng = np.random.RandomState([reparto_seed, reparto])
        self.log = log
        self.state = game.new_initial_state()
        if log:
            log.write("partida", perfil=log.perfil, rival=log.rival, jugadores=game.num_players(),
                      asiento=seat, reparto_seed=reparto_seed, reparto=reparto)
        self.next_round()

    @property
    def engine(self):
        return self.state._engine

    @property
    def round_index(self):
        return self.state._round_ptr

    @property
    def finished(self):
        return self.state.is_terminal()

    @property
    def round_over(self):
        return self.finished or self.state.is_chance_node()

    @property
    def human_turn(self):
        return not self.round_over and self.state.current_player() == self.seat

    def next_round(self):
        while self.state.is_chance_node():
            sample_chance(self.state, self.rng)
        self.turn_started = time.time()

    def apply(self, action):
        """Aplica la acción de quien tenga el turno. Devuelve False si no es legal."""
        state = self.state
        legal = state.legal_actions()
        if action not in legal:
            return False
        if self.log and state.current_player() == self.seat:
            self.log.write("decision", reparto=self.reparto, ronda=state._round_ptr,
                           cartas=self.engine.cards_per_player, fase=self.engine.phase,
                           historia=[int(a) for a in state.history()],
                           legales=[int(a) for a in legal], accion=int(action),
                           segundos=round(time.time() - self.turn_started, 2))
        state.apply_action(action)
        self.turn_started = time.time()
        if self.log and state.is_terminal():
            self.log.write("fin", reparto=self.reparto,
                           puntajes=[p.score for p in self.engine.players])
        return True

    def step_rival(self):
        state = self.state
        return self.apply(self.rival(state, state.legal_actions()))


class SpotSession:
    """Una serie de decisiones sueltas, ya elegidas, con la misma forma que `HumanMatch`."""

    finished_message = "No quedan situaciones. ¡Gracias!"

    def __init__(self, game, spots, log=None):
        self.game, self.spots, self.log = game, spots, log
        self.index = -1
        self.finished = False
        self.round_over = False
        self.next_spot()

    def next_spot(self):
        self.index += 1
        if self.index >= len(self.spots):
            self.finished = True
            return
        self.state = replay(self.game, self.spots[self.index])
        self.seat = self.state.current_player()
        self.turn_started = time.time()

    @property
    def engine(self):
        return self.state._engine

    @property
    def human_turn(self):
        return not self.finished

    def apply(self, action):
        legal = self.state.legal_actions()
        if action not in legal:
            return False
        if self.log:
            self.log.write("situacion", cartas=self.engine.cards_per_player, fase=self.engine.phase,
                           historia=self.spots[self.index], legales=[int(a) for a in legal],
                           accion=int(action), segundos=round(time.time() - self.turn_started, 2))
        self.next_spot()
        return True


def replay(game, historia):
    """Reconstruye el estado guardado en una decisión."""
    state = game.new_initial_state()
    for action in historia:
        state.apply_action(action)
    return state
