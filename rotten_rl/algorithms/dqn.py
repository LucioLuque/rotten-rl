"""
rotten_rl/algorithms/dqn.py

DQN con UNA red compartida por todos los asientos (self-play con parametros
compartidos), como en el paper de Wizard (ver docs/bibliografia.md): todos los
jugadores deciden con la misma red Q y guardan su experiencia en el mismo
replay buffer, asi que la red recibe n veces mas datos que con un agente por
asiento. La observacion ya esta en posiciones relativas al jugador, asi que la
misma red sirve para cualquier asiento.

Es una implementacion propia (no la de OpenSpiel) para poder compartir la red
entre asientos y, mas adelante, evaluar muchas partidas en lote. Reusa el
ReplayBuffer y la Transition de OpenSpiel (y la version comprimida de
rotten_rl.buffers), asi los checkpoints funcionan igual que con NFSP.

- epsilon-greedy con epsilon lineal POR EPISODIO: de --epsilon_start a
  --epsilon_end a lo largo de --dqn_epsilon_decay_fraction de la corrida.
- target: r + gamma * max_{a' legal} Q_target(s', a') (0 si s' es final).
- La red target se copia de la online cada --dqn_target_update_every pasos.
- Con --vector_games K (rotten_rl/vector.py) se juegan K partidas a la vez y se
  decide en lote; cada asiento de cada partida recuerda su decision anterior
  ("stream"). Se aprende cada --learn_every decisiones, igual que de a una,
  pero mas barato: las transiciones se guardan de a lote, el muestreo es con
  reposicion (el de OpenSpiel, sin reposicion, baraja todo el buffer en cada
  muestreo) y el buffer tiene un espejo en el device (filas de bytes con el
  info-state comprimido, ver buffers.pack_rows): muestrear es indexar ahi, sin
  copiar nada de la CPU. El buffer de CPU sigue siendo el que va al checkpoint.
"""

from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from open_spiel.python import rl_agent
from open_spiel.python.pytorch import dqn as os_dqn

from ..buffers import CompactReplayBuffer, pack_rows, unpack_rows
from ..openspiel_game import info_state_tensor
from .base import ActFn, Algorithm

_ILLEGAL = -1e9


def mlp(input_size: int, hidden: List[int], output_size: int) -> nn.Module:
    layers, size = [], input_size
    for h in hidden:
        layers += [nn.Linear(size, h), nn.ReLU()]
        size = h
    layers.append(nn.Linear(size, output_size))
    return nn.Sequential(*layers)


def _greedy(q_values: torch.Tensor, legal) -> int:
    legal = list(legal)
    return int(legal[int(torch.argmax(q_values[legal]))])


def greedy_batch(net, states, legals, num_actions: int, device) -> np.ndarray:
    """La accion legal de mayor salida de `net` en cada estado, en una sola
    pasada (para evaluar muchas partidas a la vez, ver evaluation.evaluate)."""
    infos = np.stack([info_state_tensor(s, s.current_player()) for s in states])
    mask = np.zeros((len(states), num_actions), dtype=bool)
    for i, legal in enumerate(legals):
        mask[i, legal] = True
    with torch.no_grad():
        out = net(torch.as_tensor(infos, device=device))
        out = out.masked_fill(~torch.as_tensor(mask, device=device), _ILLEGAL)
    return out.argmax(dim=1).cpu().numpy()


class _Seat(rl_agent.AbstractAgent):
    """Un asiento: decide con la red compartida y recuerda su ultima decision
    para armar la transicion cuando le vuelve a tocar (o cuando termina el
    episodio)."""

    def __init__(self, algo: "SharedDQN", player_id: int):
        self._algo = algo
        self.player_id = player_id
        self._prev_info = None
        self._prev_action = None

    def step(self, time_step, is_evaluation=False):
        algo, p = self._algo, self.player_id
        last = time_step.last()
        if not last:
            info = np.asarray(time_step.observations["info_state"][p], dtype=np.float32)
            legal = time_step.observations["legal_actions"][p]
        if is_evaluation:
            return None if last else rl_agent.StepOutput(action=algo.greedy(info, legal),
                                                         probs=None)

        if self._prev_info is not None:
            algo.add_transition(
                self._prev_info, self._prev_action, time_step.rewards[p],
                algo.terminal_info if last else info, [] if last else legal, last)
        if last:
            self._prev_info = self._prev_action = None
            return None

        action = algo.act(info, legal)
        self._prev_info, self._prev_action = info, action
        algo.after_step()
        return rl_agent.StepOutput(action=action, probs=None)


class SharedDQN(Algorithm):
    name = "dqn"
    supports_vector = True
    FLAG_KEYS = [
        "hidden_layers_sizes", "replay_buffer_capacity", "min_buffer_size_to_learn",
        "batch_size", "learn_every", "rl_learning_rate", "discount_factor", "epsilon_start",
        "epsilon_end", "dqn_epsilon_decay_fraction", "dqn_target_update_every",
        "dqn_loss", "compact_buffers", "device"]
    # Exploracion como en Wizard (de 1.0 a ~0 a lo largo de buena parte de la
    # corrida), en vez de la de NFSP (0.06 -> 0.001).
    FLAG_DEFAULTS = {"epsilon_start": 1.0, "epsilon_end": 0.02, "learn_every": 16}

    def __init__(self, game, *, hidden_layers_sizes, replay_buffer_capacity,
                 min_buffer_size_to_learn, batch_size, learn_every, learning_rate,
                 discount_factor, epsilon_start, epsilon_end, epsilon_decay_episodes,
                 target_update_every, loss="mse", compact_buffers=True, device="cpu",
                 vector_games=1):
        n = game.num_players()
        size, num_actions = game.information_state_tensor_size(), game.num_distinct_actions()
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.q = mlp(size, hidden_layers_sizes, num_actions).to(self.device)
        self.target = copy.deepcopy(self.q)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss() if loss == "huber" else nn.MSELoss()
        self.buffer_class = CompactReplayBuffer if compact_buffers else os_dqn.ReplayBuffer
        self.buffer = None                       # se crea con la primera transicion
        self.capacity = replay_buffer_capacity
        self.min_buffer = max(min_buffer_size_to_learn, batch_size)
        self.batch_size = batch_size
        self.learn_every = learn_every
        self.gamma = discount_factor
        self.eps_start, self.eps_end = epsilon_start, epsilon_end
        self.eps_decay_episodes = max(1, epsilon_decay_episodes)
        self.target_update_every = target_update_every
        self.terminal_info = np.zeros(size, dtype=np.float32)
        self.steps = 0
        self.episode = 0
        self.last_loss = None
        self._seats = [_Seat(self, i) for i in range(n)]
        self.n = n
        self._prev = {}                          # stream -> (info, accion) de --vector_games
        self.fast_learn = vector_games > 1 and compact_buffers
        self._pending_loss = None                # loss sin sincronizar con la GPU
        self._mirror = None                      # espejo del buffer en el device (fast_learn)

    @classmethod
    def from_flags(cls, game, flags) -> "SharedDQN":
        return cls(
            game,
            hidden_layers_sizes=[int(x) for x in flags.hidden_layers_sizes],
            replay_buffer_capacity=flags.replay_buffer_capacity,
            min_buffer_size_to_learn=flags.min_buffer_size_to_learn,
            batch_size=flags.batch_size,
            learn_every=flags.learn_every,
            learning_rate=flags.rl_learning_rate,
            discount_factor=flags.discount_factor,
            epsilon_start=flags.epsilon_start,
            epsilon_end=flags.epsilon_end,
            epsilon_decay_episodes=int(flags.dqn_epsilon_decay_fraction
                                       * flags.num_train_episodes),
            target_update_every=flags.dqn_target_update_every,
            loss=flags.dqn_loss,
            compact_buffers=flags.compact_buffers,
            device=flags.device,
            vector_games=flags.vector_games,
        )

    # -- jugar y aprender -----------------------------------------------------

    def seats(self) -> list:
        return self._seats

    def begin_episode(self, episode: int) -> None:
        self.episode = episode

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.episode / self.eps_decay_episodes)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _q_values(self, info: np.ndarray, net=None) -> torch.Tensor:
        net = net or self.q
        with torch.no_grad():
            return net(torch.as_tensor(info, device=self.device)[None])[0]

    def greedy(self, info: np.ndarray, legal) -> int:
        return _greedy(self._q_values(info), legal)

    def act(self, info: np.ndarray, legal) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(legal))
        return self.greedy(info, legal)

    def add_transition(self, info, action, reward, next_info, next_legal, final) -> None:
        mask = np.zeros(self.num_actions, dtype=bool)
        mask[list(next_legal)] = True
        t = os_dqn.Transition(
            info_state=np.asarray(info, dtype=np.float32),
            action=np.asarray(action, dtype=int),
            reward=np.asarray(reward, dtype=np.float32),
            next_info_state=np.asarray(next_info, dtype=np.float32),
            is_final_step=np.asarray(final, dtype=bool),
            legal_actions_mask=mask)
        if self.buffer is None:
            self.buffer = self.buffer_class.init(self.capacity, t)
        self.buffer.append(t)

    def after_step(self) -> None:
        self.steps += 1
        if self.steps % self.learn_every == 0 and self.buffer is not None \
                and len(self.buffer) >= self.min_buffer:
            self.last_loss = self._learn()
        if self.steps % self.target_update_every == 0:
            self.target.load_state_dict(self.q.state_dict())

    # -- partidas en lote (rotten_rl/vector.py) --------------------------------

    def begin_batch(self, episode: int, num_games: int) -> None:
        self.episode = episode

    def _add_batch(self, infos, actions, rewards, next_infos, finals, next_masks) -> None:
        if not infos:
            return
        if self.buffer is None or not hasattr(self.buffer, "append_batch"):
            for t in zip(infos, actions, rewards, next_infos, next_masks, finals):
                self.add_transition(t[0], t[1], t[2], t[3], np.flatnonzero(t[4]), t[5])
            return
        idx = self.buffer.append_batch(
            np.stack(infos), np.asarray(actions), np.asarray(rewards, dtype=np.float32),
            np.stack(next_infos), np.asarray(finals), np.stack(next_masks))
        if self._mirror is not None:
            self._mirror[torch.as_tensor(idx, device=self.device)] = torch.as_tensor(
                pack_rows(self.buffer.experience, idx), device=self.device)

    def batch_step(self, streams, infos, masks, legals, rewards) -> np.ndarray:
        prev_infos, prev_actions, prev_rewards, next_infos, next_masks = [], [], [], [], []
        for i, s in enumerate(streams):
            prev = self._prev.get(s)
            if prev is not None:
                prev_infos.append(prev[0])
                prev_actions.append(prev[1])
                prev_rewards.append(rewards[i])
                next_infos.append(infos[i])
                next_masks.append(masks[i])
        self._add_batch(prev_infos, prev_actions, prev_rewards, next_infos,
                        [False] * len(prev_infos), next_masks)
        with torch.no_grad():
            q = self.q(torch.as_tensor(infos, device=self.device))
            q = q.masked_fill(~torch.as_tensor(masks, device=self.device), _ILLEGAL)
            actions = q.argmax(dim=1).cpu().numpy()
        eps = self.epsilon
        for i, s in enumerate(streams):
            if np.random.rand() < eps:
                actions[i] = np.random.choice(legals[i])
            self._prev[s] = (infos[i], int(actions[i]))
            self.after_step()
        return actions

    def batch_end(self, streams, rewards) -> None:
        prev_infos, prev_actions, prev_rewards = [], [], []
        for s, r in zip(streams, rewards):
            prev = self._prev.pop(s, None)
            if prev is not None:
                prev_infos.append(prev[0])
                prev_actions.append(prev[1])
                prev_rewards.append(r)
        k = len(prev_infos)
        self._add_batch(prev_infos, prev_actions, prev_rewards, [self.terminal_info] * k,
                        [True] * k, [np.zeros(self.num_actions, dtype=bool)] * k)

    def _sample_fast(self):
        """Muestreo con reposicion del espejo del buffer en el device (se arma
        entero la primera vez, y despues lo mantiene _add_batch)."""
        if self._mirror is None:
            self._mirror = torch.as_tensor(pack_rows(self.buffer.experience),
                                           device=self.device)
        idx = np.random.randint(len(self.buffer), size=self.batch_size)
        return unpack_rows(self._mirror[torch.as_tensor(idx, device=self.device)])

    def _learn(self) -> float:
        if self.fast_learn:
            s, a, r, s2, done, mask = self._sample_fast()
        else:
            b = self.buffer.sample(self.batch_size)
            dev = self.device
            s = torch.as_tensor(np.asarray(b.info_state, dtype=np.float32), device=dev)
            a = torch.as_tensor(np.asarray(b.action, dtype=np.int64), device=dev)
            r = torch.as_tensor(np.asarray(b.reward, dtype=np.float32), device=dev)
            s2 = torch.as_tensor(np.asarray(b.next_info_state, dtype=np.float32), device=dev)
            done = torch.as_tensor(np.asarray(b.is_final_step, dtype=bool), device=dev)
            mask = torch.as_tensor(np.asarray(b.legal_actions_mask, dtype=bool), device=dev)
        q = self.q(s).gather(1, a[:, None])[:, 0]
        with torch.no_grad():
            q2 = self.target(s2).masked_fill(~mask, _ILLEGAL).max(dim=1).values
            q2 = torch.where(done | ~mask.any(dim=1), torch.zeros_like(q2), q2)
            y = r + self.gamma * q2
        loss = self.loss_fn(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.fast_learn:                      # sin esperar a la GPU en cada update
            self._pending_loss = loss.detach()
            return self.last_loss
        return float(loss.item())

    # -- evaluar ---------------------------------------------------------------

    def eval_act(self) -> ActFn:
        def act(state, legal):
            return self.greedy(info_state_tensor(state, state.current_player()), legal)
        act.batch = lambda states, legals: greedy_batch(self.q, states, legals,
                                                        self.num_actions, self.device)
        return act

    def frozen_state(self):
        return copy.deepcopy(self.q.state_dict())

    def frozen_act(self, state, rng) -> ActFn:
        net = copy.deepcopy(self.q)
        net.load_state_dict(state)
        net.eval()

        def act(s, legal):
            return _greedy(self._q_values(info_state_tensor(s, s.current_player()), net), legal)
        act.batch = lambda states, legals: greedy_batch(net, states, legals, self.num_actions,
                                                        self.device)
        return act

    def _sync_loss(self) -> None:
        if self._pending_loss is not None:
            self.last_loss = float(self._pending_loss)
            self._pending_loss = None

    def losses(self) -> Dict[str, float]:
        self._sync_loss()
        return {"loss_dqn": float(self.last_loss) if self.last_loss is not None else float("nan"),
                "epsilon": self.epsilon}

    # -- checkpoints -----------------------------------------------------------

    def checkpoint_state(self) -> Dict:
        self._sync_loss()
        return {"dqn": {"q": self.q.state_dict(), "target": self.target.state_dict(),
                        "optimizer": self.optimizer.state_dict(), "steps": self.steps,
                        "episode": self.episode, "last_loss": self.last_loss}}

    def checkpoint_buffers(self) -> Dict[str, object]:
        return {"shared.replay": self.buffer}

    def load_checkpoint_state(self, meta: Dict, load_buffer) -> None:
        s = meta["dqn"]
        self.q.load_state_dict(s["q"])
        self.target.load_state_dict(s["target"])
        self.optimizer.load_state_dict(s["optimizer"])
        self.steps, self.episode, self.last_loss = s["steps"], s["episode"], s["last_loss"]
        buf = load_buffer("shared.replay", self.buffer_class, os_dqn.Transition)
        if buf is not None:
            self.buffer = buf
        for seat in self._seats:                 # entre partidas: nada a medias
            seat._prev_info = seat._prev_action = None
        self._prev = {}
        self._mirror = None
