"""
rotten_rl/buffers.py

Buffers de NFSP y DQN que guardan el info-state comprimido, sin perder nada.

De los 808 valores del info-state, 776 son siempre 0/1 (one-hots y mascaras) y
solo 32 son fracciones (cartas en mano /h, bazas /h, puntaje /maximo,
(canto - bazas) /h, cartas jugadas por palo /13). Los binarios se guardan como
bits (np.packbits: 97 bytes) y las fracciones en float32 (128 bytes): 225 bytes
por info-state en vez de 3232, unas 14 veces menos, y al samplear se recupera
exactamente el mismo tensor. Un agente con los buffers llenos pasa de ~8.3 GB a
~1.2 GB (ver docs/optimizacion.md).

Uso: NFSP(..., replay_buffer_class=CompactReservoirBuffer) para el reservoir, y
`use_compact_buffers(agent)` antes de que el agente vea la primera transicion
(el DQN interno no recibe la clase por constructor: NFSP usa el nombre
`replay_buffer_class` para el reservoir).
"""

from __future__ import annotations

import numpy as np

from open_spiel.python.pytorch import dqn, nfsp

from .openspiel_game import _INFO_PIECES

_CONTINUOUS_PIECES = {"cards_remaining", "tricks_won", "scores", "bid_minus_tricks",
                      "suits_played"}


def _piece_mask(names) -> np.ndarray:
    mask = []
    for name, size in _INFO_PIECES:
        mask.extend([name in names] * size)
    return np.array(mask)


_CONT_MASK = _piece_mask(_CONTINUOUS_PIECES)
_BIN_IDX = np.flatnonzero(~_CONT_MASK)
_CONT_IDX = np.flatnonzero(_CONT_MASK)
INFO_SIZE = len(_CONT_MASK)
_N_BITS_BYTES = (len(_BIN_IDX) + 7) // 8
ENCODED_SIZE = _N_BITS_BYTES + 4 * len(_CONT_IDX)        # bytes por info-state


def encode_info_states(x: np.ndarray) -> np.ndarray:
    """(..., 808) float32 -> (..., 225) uint8, sin perdida."""
    x = np.asarray(x, dtype=np.float32)
    bits = np.packbits(x[..., _BIN_IDX] != 0, axis=-1)
    cont = np.ascontiguousarray(x[..., _CONT_IDX]).view(np.uint8)
    return np.concatenate([bits, cont], axis=-1)


def decode_info_states(e: np.ndarray) -> np.ndarray:
    """(..., 225) uint8 -> (..., 808) float32."""
    e = np.asarray(e, dtype=np.uint8)
    out = np.zeros(e.shape[:-1] + (INFO_SIZE,), dtype=np.float32)
    out[..., _BIN_IDX] = np.unpackbits(e[..., :_N_BITS_BYTES], axis=-1,
                                       count=len(_BIN_IDX))
    out[..., _CONT_IDX] = np.ascontiguousarray(e[..., _N_BITS_BYTES:]).view(np.float32)
    return out


def _is_encoded(arr) -> bool:
    return np.asarray(arr).dtype == np.uint8


class _CompactMixin:
    """Comprime los campos de info-state (`_INFO_FIELDS`) al guardar y los
    descomprime al samplear. `experience` guarda la version comprimida, asi que
    rotten_rl/checkpoint.py la guarda y la restaura tal cual."""

    _INFO_FIELDS: tuple = ()

    @classmethod
    def _encode(cls, experience):
        changes = {f: encode_info_states(getattr(experience, f))
                   for f in cls._INFO_FIELDS if not _is_encoded(getattr(experience, f))}
        return experience._replace(**changes) if changes else experience

    @classmethod
    def init(cls, capacity, experience):
        return super().init(capacity, cls._encode(experience))

    def append(self, experience) -> None:
        super().append(self._encode(experience))

    def sample(self, num_samples):
        batch = super().sample(num_samples)
        return batch._replace(**{f: decode_info_states(getattr(batch, f))
                                 for f in self._INFO_FIELDS})


class CompactReservoirBuffer(_CompactMixin, nfsp.ReservoirBuffer):
    """Reservoir de la politica promedio de NFSP, con el info-state comprimido."""
    _INFO_FIELDS = ("info_state",)


class CompactReplayBuffer(_CompactMixin, dqn.ReplayBuffer):
    """Replay buffer del DQN, con info_state y next_info_state comprimidos."""
    _INFO_FIELDS = ("info_state", "next_info_state")


def use_compact_buffers(agent: nfsp.NFSP) -> None:
    """Hace que el DQN interno de `agent` use CompactReplayBuffer. Llamar antes
    de la primera transicion (el buffer se crea con la primera)."""
    if agent._rl_agent._replay_buffer is not None:
        raise RuntimeError("El replay buffer ya existe: llamar antes de entrenar.")
    agent._rl_agent._replay_buffer_class = CompactReplayBuffer
