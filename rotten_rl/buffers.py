"""
rotten_rl/buffers.py

Buffers de NFSP y DQN que guardan el info-state comprimido, sin perder nada.

De los 808 valores del info-state, 776 son siempre 0/1 (one-hots y mascaras) y
solo 32 son fracciones (cartas en mano /h, bazas /h, puntaje /maximo,
(canto - bazas) /h, cartas jugadas por palo /13). Los binarios se guardan como
bits (np.packbits: 97 bytes) y las fracciones en float32 (128 bytes): 225 bytes
por info-state en vez de 3232, unas 14 veces menos, y al samplear se recupera
exactamente el mismo tensor. Un agente con los buffers llenos pasa de ~8.3 GB a
~1.2 GB (ver docs/tecnico/optimizacion.md).

Uso: NFSP(..., replay_buffer_class=CompactReservoirBuffer) para el reservoir, y
`use_compact_buffers(agent)` antes de que el agente vea la primera transicion
(el DQN interno no recibe la clase por constructor: NFSP usa el nombre
`replay_buffer_class` para el reservoir).
"""

from __future__ import annotations

import numpy as np
import torch

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


_TORCH_IDX = {}


def decode_info_states_torch(e: torch.Tensor) -> torch.Tensor:
    """Como decode_info_states, para un tensor (B, 225) uint8, en su device
    (asi a la GPU se mandan 14 veces menos bytes)."""
    if e.device not in _TORCH_IDX:
        _TORCH_IDX[e.device] = (
            torch.as_tensor(_BIN_IDX, device=e.device), torch.as_tensor(_CONT_IDX, device=e.device),
            torch.arange(7, -1, -1, device=e.device, dtype=torch.uint8))
    bin_idx, cont_idx, shifts = _TORCH_IDX[e.device]
    bits = (e[:, :_N_BITS_BYTES, None] >> shifts) & 1          # packbits: primero el bit alto
    out = torch.zeros(e.shape[0], INFO_SIZE, device=e.device)
    out[:, bin_idx] = bits.reshape(e.shape[0], -1)[:, :len(_BIN_IDX)].float()
    out[:, cont_idx] = e[:, _N_BITS_BYTES:].contiguous().view(torch.float32)
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

    def append_batch(self, info_state, action, reward, next_info_state, is_final_step,
                     legal_actions_mask) -> None:
        """Como `append` para k transiciones a la vez (arrays con k filas), en el
        mismo orden en que quedarian con k appends."""
        k = len(action)
        idx = (int(self.entry_index) + np.arange(k)) % int(self.capacity)
        e = self.experience
        e.info_state[idx] = encode_info_states(info_state)
        e.action[idx] = action
        e.reward[idx] = reward
        e.next_info_state[idx] = encode_info_states(next_info_state)
        e.is_final_step[idx] = is_final_step
        e.legal_actions_mask[idx] = legal_actions_mask
        self.entry_index = self.entry_index + k
        return idx


def pack_rows(e, idx=slice(None)) -> np.ndarray:
    """Las transiciones `idx` de un CompactReplayBuffer como filas de bytes:
    info (225) | next_info (225) | accion (1) | reward (float32, 4) | final (1) |
    mascara (una por accion). Es lo que guarda el espejo en GPU del DQN."""
    reward = np.ascontiguousarray(e.reward[idx], dtype=np.float32)
    return np.concatenate([
        e.info_state[idx], e.next_info_state[idx],
        e.action[idx].astype(np.uint8)[:, None], reward.view(np.uint8).reshape(-1, 4),
        e.is_final_step[idx].astype(np.uint8)[:, None],
        e.legal_actions_mask[idx].astype(np.uint8)], axis=1)


def unpack_rows(rows: torch.Tensor):
    """Inversa de pack_rows, en el device de `rows` (con el info-state ya
    descomprimido): (info, accion, reward, next_info, final, mascara)."""
    a = 2 * ENCODED_SIZE
    return (decode_info_states_torch(rows[:, :ENCODED_SIZE]), rows[:, a].long(),
            rows[:, a + 1:a + 5].contiguous().view(torch.float32)[:, 0],
            decode_info_states_torch(rows[:, ENCODED_SIZE:a]), rows[:, a + 5].bool(),
            rows[:, a + 6:].bool())


def use_compact_buffers(agent: nfsp.NFSP) -> None:
    """Hace que el DQN interno de `agent` use CompactReplayBuffer. Llamar antes
    de la primera transicion (el buffer se crea con la primera)."""
    if agent._rl_agent._replay_buffer is not None:
        raise RuntimeError("El replay buffer ya existe: llamar antes de entrenar.")
    agent._rl_agent._replay_buffer_class = CompactReplayBuffer
