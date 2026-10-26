"""Algoritmos de entrenamiento, con la interfaz comun de base.Algorithm."""

from .base import ActFn, Algorithm
from .dqn import SharedDQN
from .nfsp import NFSPAlgorithm, make_nfsp_agents

ALGORITHMS = {cls.name: cls for cls in (NFSPAlgorithm, SharedDQN)}

__all__ = ["ActFn", "Algorithm", "ALGORITHMS", "NFSPAlgorithm", "SharedDQN", "make_nfsp_agents"]
