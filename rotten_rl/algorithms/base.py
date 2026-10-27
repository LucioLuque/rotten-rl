"""
rotten_rl/algorithms/base.py

Interfaz comun de los algoritmos de entrenamiento. scripts/train.py solo usa
esto, asi que la evaluacion, el logging, los checkpoints, --resume, --stop_at y
la cola de experimentos son los mismos para todos los algoritmos.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

# Politica para evaluar o para usar de rival: (state, legal_actions) -> accion.
ActFn = Callable[[object, list], int]


class Algorithm:
    """Un algoritmo de entrenamiento sobre el juego `podrida`.

    Ciclo de uso en scripts/train.py:
      - `seats()`: un objeto por asiento con `step(time_step, is_evaluation)`
        (la interfaz de `rl_agent` de OpenSpiel). Se los pasa a
        `EpisodeRunner.play`; aprenden mientras juegan.
      - `begin_episode(ep)`: antes de cada partida de entrenamiento (para
        schedules por episodio, p.ej. epsilon).
      - `eval_act()`: la politica a evaluar.
      - `frozen_state()` / `frozen_act(state, rng)`: copia congelada de la
        politica actual, para el rival "past" de la evaluacion (y para guardar
        en <corrida>/checkpoints/).
      - Con --vector_games K > 1, en vez de `seats()`/`begin_episode` se juegan
        K partidas a la vez con rotten_rl/vector.py: `begin_batch`,
        `batch_step`/`batch_end` (ver ahi) y `end_batch`. Solo los algoritmos
        con `supports_vector`.
      - `losses()`: metricas de entrenamiento para loguear.
      - `checkpoint_state()` / `checkpoint_buffers()` / `load_checkpoint_state()`:
        todo lo necesario para cortar y retomar (ver rotten_rl/checkpoint.py).
    """

    name: str = ""
    # Flags de scripts/train.py que usa este algoritmo (van al config.json).
    FLAG_KEYS: List[str] = []
    # Defaults propios de este algoritmo para flags compartidos, cuando no se
    # pasan explicitamente (p.ej. otra exploracion para DQN que para NFSP).
    FLAG_DEFAULTS: Dict[str, object] = {}
    # Si sabe jugar muchas partidas a la vez (rotten_rl/vector.py).
    supports_vector: bool = False

    def seats(self) -> list:
        raise NotImplementedError

    def begin_episode(self, episode: int) -> None:
        pass

    def begin_batch(self, episode: int, num_games: int) -> None:
        """Antes de un lote de partidas (--vector_games); `episode` es la primera."""

    def end_batch(self, num_games: int) -> None:
        """Despues de un lote de partidas, todas terminadas."""

    def eval_act(self) -> ActFn:
        raise NotImplementedError

    def frozen_state(self):
        raise NotImplementedError

    def frozen_act(self, state, rng: np.random.RandomState) -> ActFn:
        raise NotImplementedError

    def losses(self) -> Dict[str, float]:
        return {}

    def checkpoint_state(self) -> Dict:
        raise NotImplementedError

    def checkpoint_buffers(self) -> Dict[str, object]:
        """{nombre: buffer} a guardar (None si todavia no existe)."""
        return {}

    def load_checkpoint_state(self, meta: Dict, load_buffer: Callable) -> None:
        """Restaura desde `meta` (lo que devolvio checkpoint_state, mas
        "buffers"). `load_buffer(nombre, buffer_class, transition_class)`
        devuelve el buffer guardado con ese nombre (o None)."""
        raise NotImplementedError
