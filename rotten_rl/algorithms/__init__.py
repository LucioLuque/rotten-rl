"""Algoritmos de entrenamiento, con la interfaz comun de base.Algorithm."""

from .base import ActFn, Algorithm
from .dqn import SharedDQN
from .nfsp import NFSPAlgorithm, make_nfsp_agents
from .ppo import SharedPPO

ALGORITHMS = {cls.name: cls for cls in (NFSPAlgorithm, SharedDQN, SharedPPO)}

__all__ = ["ActFn", "Algorithm", "ALGORITHMS", "NFSPAlgorithm", "SharedDQN", "SharedPPO",
           "make_nfsp_agents"]
