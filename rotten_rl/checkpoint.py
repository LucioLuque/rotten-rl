"""
rotten_rl/checkpoint.py

Checkpoints completos de un entrenamiento con agentes NFSP de OpenSpiel, para
poder cortarlo (p.ej. por el reinicio diario del laboratorio) y retomarlo
exactamente donde quedo.

El save/restore de NFSP de OpenSpiel 2.0.2 no sirve: guarda con unas claves
("model", "optimiser") y lee con otras ("model_state_dict", ...), y no guarda
los buffers ni los contadores del DQN interno. Aca se guarda todo a mano:

  ckpt_ep<episodio>/
    meta.pt            redes, optimizadores, contadores, RNGs y `extra`
    buffers/           un .npy por campo de cada buffer (solo la parte llena)

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
    if info is None:
        return None
    arrays = {name: np.load(f"{path_prefix}.{name}.npy") for name in info["fields"]}
    sample = transition_class(**{name: arr[0] for name, arr in arrays.items()})
    buf = buffer_class.init(info["capacity"], sample)
    for name, arr in arrays.items():
        getattr(buf.experience, name)[:info["size"]] = arr
    setattr(buf, info["counter"], np.array(info["counter_value"]))
    return buf


# ----------------------------------------------------------------------
# Estado de un agente NFSP
# ----------------------------------------------------------------------

def _nfsp_state(agent) -> Dict:
    rl = agent._rl_agent
    return {
        "avg_network": agent._avg_network.state_dict(),
        "avg_optimizer": agent._optimizer.state_dict(),
        "iteration": agent._iteration,
        "mode": agent._mode.value,
        "last_sl_loss": agent._last_sl_loss_value,
        "q_network": rl._q_network.state_dict(),
        "target_q_network": rl._target_q_network.state_dict(),
        "rl_optimizer": rl._optimizer.state_dict(),
        "rl_iteration": rl._iteration,
        "last_rl_loss": rl._last_loss_value,
    }


def _load_nfsp_state(agent, s: Dict) -> None:
    from open_spiel.python.pytorch import nfsp

    rl = agent._rl_agent
    agent._avg_network.load_state_dict(s["avg_network"])
    agent._optimizer.load_state_dict(s["avg_optimizer"])
    agent._iteration = s["iteration"]
    agent._mode = nfsp.MODE(s["mode"])
    agent._last_sl_loss_value = s["last_sl_loss"]
    rl._q_network.load_state_dict(s["q_network"])
    rl._target_q_network.load_state_dict(s["target_q_network"])
    rl._optimizer.load_state_dict(s["rl_optimizer"])
    rl._iteration = s["rl_iteration"]
    rl._last_loss_value = s["last_rl_loss"]
    # Los checkpoints se toman entre partidas: no hay transicion a medias.
    agent._prev_timestep = agent._prev_action = None
    rl._prev_timestep = rl._prev_action = None


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------

def save_checkpoint(ckpt_root: str, episode: int, agents: List, rng: np.random.RandomState,
                    extra: Optional[Dict] = None, with_buffers: bool = True,
                    keep_last: int = 2) -> str:
    """Guarda el estado completo despues de `episode` partidas y borra los
    checkpoints viejos (deja los `keep_last` mas nuevos). Devuelve la ruta."""
    final = os.path.join(ckpt_root, f"ckpt_ep{episode}")
    tmp = final + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "buffers"))

    buffers = []
    for i, agent in enumerate(agents):
        b = {}
        if with_buffers:
            pre = os.path.join(tmp, "buffers", f"agent{i}")
            b["reservoir"] = _save_buffer(agent._reservoir_buffer, pre + ".reservoir")
            b["replay"] = _save_buffer(agent._rl_agent._replay_buffer, pre + ".replay")
        buffers.append(b)

    meta = {
        "episode": episode,
        "agents": [_nfsp_state(a) for a in agents],
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


def load_checkpoint(path: str, agents: List, rng: np.random.RandomState) -> Dict:
    """Restaura en `agents` (ya construidos con la misma config) y en `rng` el
    estado guardado en `path`. Devuelve {"episode": ..., "extra": ...}."""
    from open_spiel.python.pytorch import dqn, nfsp

    meta = torch.load(os.path.join(path, "meta.pt"), weights_only=False)
    for i, (agent, s) in enumerate(zip(agents, meta["agents"])):
        _load_nfsp_state(agent, s)
        b = meta["buffers"][i]
        if b:
            pre = os.path.join(path, "buffers", f"agent{i}")
            agent._reservoir_buffer = _load_buffer(
                nfsp.ReservoirBuffer, nfsp.Transition, b["reservoir"], pre + ".reservoir")
            agent._rl_agent._replay_buffer = _load_buffer(
                dqn.ReplayBuffer, dqn.Transition, b["replay"], pre + ".replay")

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
