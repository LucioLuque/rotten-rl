"""
rotten_rl/training.py

Loop de partidas para agentes de OpenSpiel (`rl_agent`, p.ej. NFSP o DQN), sin
pasar por `rl_environment`.

`rl_environment` es generico y para este juego desperdicia mucho: en cada
decision arma el info-state de TODOS los jugadores, y cada uno pedido por
pyspiel (que copia el estado y arma el tensor dos veces). Aca se arma solo el
del jugador que decide, directo con el observer. Los TimeSteps que reciben los
agentes son los mismos que con `rl_environment` (con el reward acumulado, ver
`with_pending_reward`); lo verifica tests/test_training.py.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from open_spiel.python.rl_environment import StepType, TimeStep

from .openspiel_game import info_state_tensor


def with_pending_reward(time_step, pending: List[float], player: int):
    """TimeStep para `player` con el reward que acumulo desde su ultima decision.

    El juego da reward al cerrar cada ronda, pero cada agente solo ve los
    TimeSteps de sus propios turnos, y rl_environment no acumula: el reward de
    la ronda le llegaria solo al que juega justo despues del cierre (con 2
    jugadores se pierde ~46% del reward; con 7, ~74%). Aca se le entrega a cada
    uno todo lo acumulado y se resetea su contador.

    Uso: despues de cada env.step, sumar time_step.rewards[i] a pending[i]; y
    pasarle a cada agente with_pending_reward(time_step, pending, pid)."""
    if time_step.rewards is None:
        return time_step
    rewards = list(time_step.rewards)
    rewards[player] = pending[player]
    pending[player] = 0.0
    return time_step._replace(rewards=rewards)


# Rival que no es un rl_agent: (state, legal_actions) -> accion.
ActFn = Callable[[object, list], int]


class EpisodeRunner:
    """Juega partidas completas de `game` con un agente o una funcion por asiento.

    `seats[i]` es un `rl_agent` (recibe TimeSteps y aprende si no es evaluacion)
    o una `ActFn` (rival fijo, p.ej. la heuristica). Cada agente recibe en sus
    turnos un TimeStep con su info-state, sus acciones legales y el reward
    acumulado desde su decision anterior, y al final de la partida un TimeStep
    LAST con lo que le quedaba pendiente."""

    def __init__(self, game, rng: np.random.RandomState):
        self._game = game
        self._n = game.num_players()
        self._rng = rng
        self._terminal_info = np.zeros(game.information_state_tensor_size(), np.float32)

    def _sample_chance(self, state) -> int:
        outcomes, probs = zip(*state.chance_outcomes())
        return int(self._rng.choice(outcomes, p=probs))

    def _last_step(self, pending: List[float]) -> TimeStep:
        n = self._n
        return TimeStep(
            observations={"info_state": [self._terminal_info] * n,
                          "legal_actions": [[] for _ in range(n)],
                          "current_player": -4},
            rewards=list(pending), discounts=[0.0] * n, step_type=StepType.LAST)

    def play(self, seats: Sequence, is_evaluation: bool = False,
             on_round_reward: Optional[Callable[[List[float]], None]] = None,
             round_episodes: bool = False) -> List[float]:
        """Juega una partida y devuelve los retornos de cada asiento.
        `on_round_reward(rewards)` se llama con los rewards de cada paso que los
        tenga (para loguear, p.ej. el retorno del asiento 0).

        Con `round_episodes`, para los agentes cada RONDA es un episodio: al
        cerrarse una ronda reciben un TimeStep LAST con su reward, y la ronda
        siguiente empieza de cero (FIRST). Asi el valor de una decision solo
        incluye los puntos de su ronda y no los de las rondas siguientes, que
        no dependen de ella (como en el paper de Wizard; ver docs/bibliografia.md).
        La partida completa se juega igual."""
        n = self._n
        is_agent = [hasattr(s, "step") for s in seats]
        state = self._game.new_initial_state()
        pending = [0.0] * n
        first = True
        round_ptr = state._round_ptr

        while not state.is_terminal():
            if state.is_chance_node():
                state.apply_action(self._sample_chance(state))
                continue
            pid = state.current_player()
            legal = state.legal_actions(pid)
            if is_agent[pid]:
                info = [None] * n
                legals = [None] * n
                info[pid] = info_state_tensor(state, pid)
                legals[pid] = legal
                rewards = [0.0] * n
                rewards[pid] = pending[pid]
                pending[pid] = 0.0
                ts = TimeStep(
                    observations={"info_state": info, "legal_actions": legals,
                                  "current_player": pid},
                    rewards=rewards, discounts=[1.0] * n,
                    step_type=StepType.FIRST if first else StepType.MID)
                action = seats[pid].step(ts, is_evaluation=is_evaluation).action
            else:
                action = seats[pid](state, legal)
            first = False

            state.apply_action(action)
            step_rewards = state.rewards()
            if any(step_rewards):
                for i in range(n):
                    pending[i] += step_rewards[i]
                if on_round_reward is not None:
                    on_round_reward(step_rewards)

            if round_episodes and state._round_ptr != round_ptr and not state.is_terminal():
                round_ptr = state._round_ptr
                last = self._last_step(pending)
                for pid in range(n):
                    if is_agent[pid]:
                        seats[pid].step(last, is_evaluation=is_evaluation)
                pending = [0.0] * n
                first = True

        last = self._last_step(pending)
        for pid in range(n):
            if is_agent[pid]:
                seats[pid].step(last, is_evaluation=is_evaluation)
        return list(state.returns())
