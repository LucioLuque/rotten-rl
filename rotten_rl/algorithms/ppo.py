"""
rotten_rl/algorithms/ppo.py

PPO con UNA red de politica y una de valor compartidas por todos los asientos,
en self-play contra la politica actual. Es lo que mejor anduvo en Big 2 (Patwa
2026: PPO le gano a Q-learning, SARSA y Monte Carlo Q, la entropia moderada
ayuda y el self-play contra la politica actual le gano a usar checkpoints); de
MAPPO se toman pocas epocas por lote y clipping chico (ver docs/bibliografia.md).

- Se juntan --ppo_rollout_episodes partidas con la politica actual y se hace
  la actualizacion (--ppo_epochs pasadas en minibatches de --ppo_minibatch).
- Ventajas con GAE(lambda) por asiento; el episodio termina en cada LAST (con
  --round_episodes, en cada ronda).
- Acciones ilegales enmascaradas en el softmax. Se evalua con la accion mas
  probable.
- El critico ve lo mismo que el jugador (critico "local"). Un critico que vea
  todas las manos durante el entrenamiento queda para mas adelante.
"""

from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from open_spiel.python import rl_agent

from ..openspiel_game import info_state_tensor
from .base import ActFn, Algorithm
from .dqn import mlp

_ILLEGAL = -1e9


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, _ILLEGAL)


class _Seat(rl_agent.AbstractAgent):
    """Un asiento: decide con la politica compartida y anota su trayectoria."""

    def __init__(self, algo: "SharedPPO", player_id: int):
        self._algo = algo
        self.player_id = player_id
        self._open = False          # hay un paso propio esperando su reward

    def step(self, time_step, is_evaluation=False):
        algo, p = self._algo, self.player_id
        last = time_step.last()
        if is_evaluation:
            if last:
                return None
            info = np.asarray(time_step.observations["info_state"][p], dtype=np.float32)
            return rl_agent.StepOutput(
                action=algo.greedy(info, time_step.observations["legal_actions"][p]), probs=None)

        if self._open:
            algo.close_step(p, time_step.rewards[p], done=last)
            self._open = False
        if last:
            return None
        info = np.asarray(time_step.observations["info_state"][p], dtype=np.float32)
        action = algo.act_and_record(p, info, time_step.observations["legal_actions"][p])
        self._open = True
        return rl_agent.StepOutput(action=action, probs=None)


class SharedPPO(Algorithm):
    name = "ppo"
    FLAG_KEYS = [
        "hidden_layers_sizes", "rl_learning_rate", "discount_factor", "ppo_rollout_episodes",
        "ppo_epochs", "ppo_minibatch", "ppo_clip", "ppo_ent_coef", "ppo_vf_coef",
        "ppo_gae_lambda", "ppo_max_grad_norm", "device"]
    FLAG_DEFAULTS = {"rl_learning_rate": 3e-4}

    def __init__(self, game, *, hidden_layers_sizes, learning_rate, discount_factor,
                 rollout_episodes=32, epochs=5, minibatch=512, clip=0.2, ent_coef=0.01,
                 vf_coef=0.5, gae_lambda=0.95, max_grad_norm=0.5, device="cpu"):
        n = game.num_players()
        size, self.num_actions = game.information_state_tensor_size(), game.num_distinct_actions()
        self.device = torch.device(device)
        self.policy = mlp(size, hidden_layers_sizes, self.num_actions).to(self.device)
        self.value = mlp(size, hidden_layers_sizes, 1).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=learning_rate)
        self.gamma, self.lam = discount_factor, gae_lambda
        self.rollout_episodes, self.epochs, self.minibatch = rollout_episodes, epochs, minibatch
        self.clip, self.ent_coef, self.vf_coef = clip, ent_coef, vf_coef
        self.max_grad_norm = max_grad_norm
        self.n = n
        self._seats = [_Seat(self, i) for i in range(n)]
        self._reset_rollout()
        self.updates = 0
        self.last_stats: Dict[str, float] = {}

    @classmethod
    def from_flags(cls, game, flags) -> "SharedPPO":
        return cls(
            game,
            hidden_layers_sizes=[int(x) for x in flags.hidden_layers_sizes],
            learning_rate=flags.rl_learning_rate,
            discount_factor=flags.discount_factor,
            rollout_episodes=flags.ppo_rollout_episodes,
            epochs=flags.ppo_epochs,
            minibatch=flags.ppo_minibatch,
            clip=flags.ppo_clip,
            ent_coef=flags.ppo_ent_coef,
            vf_coef=flags.ppo_vf_coef,
            gae_lambda=flags.ppo_gae_lambda,
            max_grad_norm=flags.ppo_max_grad_norm,
            device=flags.device,
        )

    # -- rollout -----------------------------------------------------------------

    def _reset_rollout(self):
        # Por asiento: listas paralelas de los pasos de la partida en curso y
        # pasos ya cerrados (con reward y done) de las partidas de este lote.
        self.traj = [{"obs": [], "mask": [], "act": [], "logp": [], "val": [], "rew": [],
                      "done": []} for _ in range(self.n)]
        self.games_in_rollout = 0

    def seats(self) -> list:
        return self._seats

    def begin_episode(self, episode: int) -> None:
        """Antes de cada partida: si ya hay --ppo_rollout_episodes partidas
        juntadas, actualizar."""
        if episode > 1:
            self.games_in_rollout += 1
        if self.games_in_rollout >= self.rollout_episodes:
            self._update()
            self._reset_rollout()

    def _mask(self, legal) -> np.ndarray:
        m = np.zeros(self.num_actions, dtype=bool)
        m[list(legal)] = True
        return m

    def act_and_record(self, p: int, info: np.ndarray, legal) -> int:
        mask = self._mask(legal)
        with torch.no_grad():
            x = torch.as_tensor(info, device=self.device)[None]
            logits = _masked_logits(self.policy(x)[0], torch.as_tensor(mask, device=self.device))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)
            v = self.value(x)[0, 0]
        t = self.traj[p]
        t["obs"].append(info)
        t["mask"].append(mask)
        t["act"].append(int(a))
        t["logp"].append(float(logp))
        t["val"].append(float(v))
        return int(a)

    def close_step(self, p: int, reward: float, done: bool) -> None:
        t = self.traj[p]
        t["rew"].append(float(reward))
        t["done"].append(bool(done))

    # -- actualizacion -------------------------------------------------------------

    def _advantages(self, t):
        rew, val, done = np.array(t["rew"]), np.array(t["val"]), np.array(t["done"])
        adv = np.zeros(len(rew), dtype=np.float32)
        gae = 0.0
        for i in reversed(range(len(rew))):
            next_v = 0.0 if done[i] or i + 1 == len(rew) else val[i + 1]
            delta = rew[i] + self.gamma * next_v - val[i]
            gae = delta + self.gamma * self.lam * (0.0 if done[i] else gae)
            adv[i] = gae
        return adv, adv + val.astype(np.float32)

    def _update(self) -> None:
        parts = [t for t in self.traj if t["rew"]]
        if not parts:
            return
        advs, rets = zip(*(self._advantages(t) for t in parts))
        dev = self.device
        obs = torch.as_tensor(np.concatenate([np.stack(t["obs"]) for t in parts]), device=dev)
        mask = torch.as_tensor(np.concatenate([np.stack(t["mask"]) for t in parts]), device=dev)
        act = torch.as_tensor(np.concatenate([t["act"] for t in parts]), device=dev)
        old_logp = torch.as_tensor(np.concatenate([t["logp"] for t in parts]), device=dev,
                                   dtype=torch.float32)
        adv = torch.as_tensor(np.concatenate(advs), device=dev)
        ret = torch.as_tensor(np.concatenate(rets), device=dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        size = len(act)
        stats = {"loss_policy": 0.0, "loss_value": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        count = 0
        for _ in range(self.epochs):
            perm = np.random.permutation(size)
            for start in range(0, size, self.minibatch):
                idx = torch.as_tensor(perm[start:start + self.minibatch], device=dev)
                logits = _masked_logits(self.policy(obs[idx]), mask[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act[idx])
                ratio = torch.exp(logp - old_logp[idx])
                a = adv[idx]
                loss_pi = -torch.min(ratio * a,
                                     torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a).mean()
                loss_v = ((self.value(obs[idx])[:, 0] - ret[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = loss_pi + self.vf_coef * loss_v - self.ent_coef * ent
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm)
                self.optimizer.step()
                stats["loss_policy"] += float(loss_pi.detach())
                stats["loss_value"] += float(loss_v.detach())
                stats["entropy"] += float(ent.detach())
                stats["clip_frac"] += float(((ratio.detach() - 1).abs() > self.clip).float().mean())
                count += 1
        self.last_stats = {k: v / count for k, v in stats.items()}
        self.updates += 1

    # -- evaluar -------------------------------------------------------------------

    def greedy(self, info: np.ndarray, legal, net=None) -> int:
        net = net or self.policy
        with torch.no_grad():
            logits = net(torch.as_tensor(info, device=self.device)[None])[0]
        legal = list(legal)
        return int(legal[int(torch.argmax(logits[legal]))])

    def eval_act(self) -> ActFn:
        def act(state, legal):
            return self.greedy(info_state_tensor(state, state.current_player()), legal)
        return act

    def frozen_state(self):
        return copy.deepcopy(self.policy.state_dict())

    def frozen_act(self, state, rng) -> ActFn:
        net = copy.deepcopy(self.policy)
        net.load_state_dict(state)
        net.eval()

        def act(s, legal):
            return self.greedy(info_state_tensor(s, s.current_player()), legal, net)
        return act

    def losses(self) -> Dict[str, float]:
        out = {k: float("nan") for k in ("loss_policy", "loss_value", "entropy", "clip_frac")}
        out.update(self.last_stats)
        out["updates"] = float(self.updates)
        return out

    # -- checkpoints ---------------------------------------------------------------
    # Se guarda tambien el lote a medio juntar, para que retomar sea igual que
    # seguir de corrido.

    def checkpoint_state(self) -> Dict:
        return {"ppo": {"policy": self.policy.state_dict(), "value": self.value.state_dict(),
                        "optimizer": self.optimizer.state_dict(), "updates": self.updates,
                        "last_stats": self.last_stats, "traj": self.traj,
                        "games_in_rollout": self.games_in_rollout}}

    def load_checkpoint_state(self, meta: Dict, load_buffer) -> None:
        s = meta["ppo"]
        self.policy.load_state_dict(s["policy"])
        self.value.load_state_dict(s["value"])
        self.optimizer.load_state_dict(s["optimizer"])
        self.updates, self.last_stats = s["updates"], s["last_stats"]
        self.traj, self.games_in_rollout = s["traj"], s["games_in_rollout"]
        for seat in self._seats:                 # entre partidas: nada a medias
            seat._open = False
