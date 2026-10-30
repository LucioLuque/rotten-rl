"""
rotten_rl/vector.py

Juega muchas partidas de entrenamiento a la vez, para que la red decida en lote.

Con `EpisodeRunner` cada decision es una pasada de la red con un solo ejemplo
(~190 por partida con 3 jugadores), y casi todo ese tiempo es overhead de
PyTorch. Aca se juegan K partidas en paralelo y en cada paso se le piden al
algoritmo todas las decisiones pendientes juntas: una pasada de la red para K
decisiones, y recien ahi la GPU sirve.

Cada asiento de cada partida es un "stream" (stream = partida * n + asiento):
el algoritmo guarda por stream lo mismo que guardaba por asiento con
EpisodeRunner (la trayectoria en PPO, la decision anterior en DQN). Los rewards
llegan igual que con EpisodeRunner (lo acumulado desde la decision anterior del
mismo jugador) y, con `round_episodes`, cada ronda termina el episodio de todos.

El algoritmo tiene que implementar:
  - `batch_step(streams, infos, masks, legals, rewards) -> acciones`: cierra el
    paso anterior de cada stream (si hay) con su reward y elige la accion.
  - `batch_end(streams, rewards)`: fin de episodio de esos streams.
Si el algoritmo tiene `wants_hidden_hands = True` (PPO con critico
centralizado), `batch_step` recibe ademas `hidden`: las manos de los demas
jugadores de cada pedido (`hidden_hands_tensor`).
"""

from __future__ import annotations

from typing import List

import numpy as np

from .openspiel_game import hidden_hands_tensor, info_state_tensor


class VectorRunner:
    def __init__(self, game, rng: np.random.RandomState):
        self._game = game
        self._n = game.num_players()
        self._num_actions = game.num_distinct_actions()
        self._rng = rng

    def _sample_chance(self, state) -> int:
        outcomes, probs = zip(*state.chance_outcomes())
        return int(self._rng.choice(outcomes, p=probs))

    def play(self, algo, num_games: int, round_episodes: bool = False) -> List[float]:
        """Juega `num_games` partidas de self-play con `algo` en todos los
        asientos. Devuelve la suma de rewards del asiento 0 en cada partida."""
        n = self._n
        states = [self._game.new_initial_state() for _ in range(num_games)]
        pending = np.zeros((num_games, n))
        seat0 = [0.0] * num_games
        round_ptr = [s._round_ptr for s in states]
        alive = list(range(num_games))

        while alive:
            requests = []
            still = []
            for g in alive:
                state = states[g]
                while state.is_chance_node():
                    state.apply_action(self._sample_chance(state))
                if state.is_terminal():
                    algo.batch_end([g * n + p for p in range(n)], list(pending[g]))
                    pending[g] = 0.0
                    continue
                still.append(g)
                requests.append((g, state.current_player()))
            alive = still
            if not requests:
                break

            infos = np.stack([info_state_tensor(states[g], p) for g, p in requests])
            legals = [states[g].legal_actions(p) for g, p in requests]
            masks = np.zeros((len(requests), self._num_actions), dtype=bool)
            for i, legal in enumerate(legals):
                masks[i, legal] = True
            rewards = [float(pending[g, p]) for g, p in requests]
            for g, p in requests:
                pending[g, p] = 0.0
            extra = {}
            if getattr(algo, "wants_hidden_hands", False):
                extra["hidden"] = np.stack([hidden_hands_tensor(states[g], p)
                                            for g, p in requests])
            actions = algo.batch_step([g * n + p for g, p in requests], infos, masks, legals,
                                      rewards, **extra)

            for (g, _), action in zip(requests, actions):
                state = states[g]
                state.apply_action(int(action))
                step_rewards = state.rewards()
                if any(step_rewards):
                    pending[g] += step_rewards
                    seat0[g] += step_rewards[0]
                if round_episodes and state._round_ptr != round_ptr[g] \
                        and not state.is_terminal():
                    round_ptr[g] = state._round_ptr
                    algo.batch_end([g * n + p for p in range(n)], list(pending[g]))
                    pending[g] = 0.0
        return seat0
