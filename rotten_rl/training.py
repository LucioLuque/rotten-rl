"""
rotten_rl/training.py

Utilidades compartidas por los scripts de entrenamiento.
"""

from __future__ import annotations

from typing import List


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
