"""
rotten_rl/checkpoint.py

Checkpoints completos de un entrenamiento, para poder cortarlo (p.ej. por el
reinicio diario del laboratorio) y retomarlo exactamente donde quedo. Que se
guarda lo decide cada algoritmo (rotten_rl.algorithms.Algorithm:
checkpoint_state / checkpoint_buffers / load_checkpoint_state); aca se maneja el
formato en disco:

  ckpt_ep<episodio>/
    meta.pt            estado del algoritmo (redes, optimizadores, contadores),
                       RNGs y `extra`
    buffers/           <nombre>.<campo>.npy por cada buffer (solo la parte llena)

(El save/restore de NFSP de OpenSpiel 2.0.2 no sirve: ver
docs/openspiel-upstream.md.)

Se escribe primero en ckpt_ep<N>.tmp y despues se renombra, asi un corte en
medio del guardado nunca deja un checkpoint a medias.
"""

from __future__ import annotations

import os
import random
import re
import shutil
from typing import Dict, List, Optional

import numpy as np
import torch

_CKPT_RE = re.compile(r"^ckpt_ep(\d+)$")


# ----------------------------------------------------------------------
# Buffers (ReservoirBuffer de NFSP y ReplayBuffer de DQN)
# ----------------------------------------------------------------------

def _buffer_counter_name(buf) -> str:
    # ReservoirBuffer cuenta con `add_calls`; ReplayBuffer, con `entry_index`.
    return "add_calls" if hasattr(buf, "add_calls") else "entry_index"


def _save_buffer(buf, path_prefix: str) -> Optional[Dict]:
    if buf is None:
        return None
    size = len(buf)
    fields = buf.experience._fields
    for name in fields:
        np.save(f"{path_prefix}.{name}.npy", getattr(buf.experience, name)[:size])
    counter = _buffer_counter_name(buf)
    return {"capacity": int(buf.capacity), "size": int(size), "fields": list(fields),
            "counter": counter, "counter_value": int(getattr(buf, counter))}


def _load_buffer(buffer_class, transition_class, info: Optional[Dict], path_prefix: str):
    """Restaura con `buffer_class` (la que usa el agente: comun o la comprimida
    de rotten_rl.buffers). Un checkpoint sin comprimir se puede cargar en un
    buffer comprimido, no al reves."""
    if info is None:
        return None
    arrays = {name: np.load(f"{path_prefix}.{name}.npy") for name in info["fields"]}
    if arrays["info_state"].dtype == np.uint8 and not hasattr(buffer_class, "_encode"):
        raise ValueError(f"El checkpoint tiene buffers comprimidos y {buffer_class.__name__} "
                         "no los sabe leer: crear los agentes con --compact_buffers.")
    sample = transition_class(**{name: arr[0] for name, arr in arrays.items()})
    buf = buffer_class.init(info["capacity"], sample)
    for name, arr in arrays.items():
        target = getattr(buf.experience, name)
        if target.dtype != arr.dtype:
            if not hasattr(buffer_class, "_encode") or target.dtype != np.uint8:
                raise ValueError(f"El checkpoint guarda {name} como {arr.dtype} y el buffer "
                                 f"({buffer_class.__name__}) lo espera como {target.dtype}")
            from .buffers import encode_info_states
            arr = encode_info_states(arr)
        target[:info["size"]] = arr
    setattr(buf, info["counter"], np.array(info["counter_value"]))
    return buf


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------

def save_checkpoint(ckpt_root: str, episode: int, algorithm, rng: np.random.RandomState,
                    extra: Optional[Dict] = None, with_buffers: bool = True,
                    keep_last: int = 2) -> str:
    """Guarda el estado completo de `algorithm` (un rotten_rl.algorithms.Algorithm)
    despues de `episode` partidas y borra los checkpoints viejos (deja los
    `keep_last` mas nuevos). Devuelve la ruta."""
    final = os.path.join(ckpt_root, f"ckpt_ep{episode}")
    tmp = final + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "buffers"))

    buffers = {}
    if with_buffers:
        for name, buf in algorithm.checkpoint_buffers().items():
            buffers[name] = _save_buffer(buf, os.path.join(tmp, "buffers", name))

    meta = {
        "episode": episode,
        **algorithm.checkpoint_state(),
        "buffers": buffers,
        "rng": {
            "numpy_global": np.random.get_state(),
            "numpy_run": rng.get_state(),
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
        },
        "extra": extra or {},
    }
    torch.save(meta, os.path.join(tmp, "meta.pt"))
    shutil.rmtree(final, ignore_errors=True)
    os.rename(tmp, final)

    for old in list_checkpoints(ckpt_root)[:-keep_last]:
        shutil.rmtree(old, ignore_errors=True)
    return final


def load_checkpoint(path: str, algorithm, rng: np.random.RandomState) -> Dict:
    """Restaura en `algorithm` (ya construido con la misma config) y en `rng` el
    estado guardado en `path`. Devuelve {"episode": ..., "extra": ...}."""
    meta = torch.load(os.path.join(path, "meta.pt"), weights_only=False)
    infos = meta.get("buffers") or {}

    def load_buffer(name, buffer_class, transition_class):
        return _load_buffer(buffer_class, transition_class, infos.get(name),
                            os.path.join(path, "buffers", name))

    algorithm.load_checkpoint_state(meta, load_buffer)

    r = meta["rng"]
    np.random.set_state(r["numpy_global"])
    rng.set_state(r["numpy_run"])
    torch.set_rng_state(r["torch"])
    random.setstate(r["python"])
    return {"episode": meta["episode"], "extra": meta["extra"]}


def list_checkpoints(ckpt_root: str) -> List[str]:
    """Checkpoints completos en `ckpt_root`, del mas viejo al mas nuevo."""
    if not os.path.isdir(ckpt_root):
        return []
    found = []
    for name in os.listdir(ckpt_root):
        m = _CKPT_RE.match(name)
        if m and os.path.exists(os.path.join(ckpt_root, name, "meta.pt")):
            found.append((int(m.group(1)), os.path.join(ckpt_root, name)))
    return [p for _, p in sorted(found)]


def latest_checkpoint(ckpt_root: str) -> Optional[str]:
    ckpts = list_checkpoints(ckpt_root)
    return ckpts[-1] if ckpts else None
