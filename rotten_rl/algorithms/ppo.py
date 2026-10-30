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
- El critico ve lo mismo que el jugador (critico "local"). Con --ppo_critic
  central ve ademas las manos de los demas (MAPPO, Yu et al. 2022: critico
  centralizado, politica descentralizada): estima mejor el valor de cada
  situacion y baja la varianza de las ventajas. La politica sigue viendo solo
  lo suyo, asi que se evalua y se juega igual. Solo con --vector_games > 1.
- Con --vector_games K (rotten_rl/vector.py) se juegan K partidas a la vez y se
  decide en lote; cada asiento de cada partida tiene su trayectoria ("stream").
  Se actualiza al terminar un lote de partidas si ya hay --ppo_rollout_episodes.
- Opcionales, de la receta estandar de PPO (CleanRL; Andrychowicz et al. 2021,
  "What Matters in On-Policy RL"): --ppo_anneal_lr baja la tasa de aprendizaje
  linealmente hasta 0 en --num_train_episodes partidas, y --ppo_init orthogonal
  inicializa las redes ortogonales con la ultima capa de la politica 100 veces
  mas chica (arranca casi uniforme).
"""

from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from open_spiel.python import rl_agent

from ..openspiel_game import HIDDEN_HANDS_SIZE, info_state_tensor
from .base import ActFn, Algorithm
from .dqn import greedy_batch, mlp

_ILLEGAL = -1e9
# Metricas promediadas sobre los minibatches de una actualizacion.
_STATS_MEAN = ("loss_policy", "loss_value", "entropy", "clip_frac", "approx_kl", "grad_norm")


def _orthogonal_init(net: nn.Sequential, last_gain: float) -> None:
    """Pesos ortogonales (ganancia sqrt(2) en las capas ocultas, `last_gain` en
    la de salida) y bias en 0."""
    linears = [m for m in net if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linears):
        gain = last_gain if i == len(linears) - 1 else np.sqrt(2)
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.zeros_(layer.bias)


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, _ILLEGAL)


class _Seat(rl_agent.AbstractAgent):
    """Un asiento: decide con la politica compartida y anota su trayectoria."""

    def __init__(self, algo: "SharedPPO", player_id: int):
        self._algo = algo
        self.player_id = player_id

    def step(self, time_step, is_evaluation=False):
        algo, p = self._algo, self.player_id
        last = time_step.last()
        if is_evaluation:
            if last:
                return None
            info = np.asarray(time_step.observations["info_state"][p], dtype=np.float32)
            return rl_agent.StepOutput(
                action=algo.greedy(info, time_step.observations["legal_actions"][p]), probs=None)

        if algo.open[p]:
            algo.close_step(p, time_step.rewards[p], done=last)
        if last:
            return None
        info = np.asarray(time_step.observations["info_state"][p], dtype=np.float32)
        action = algo.act_and_record(p, info, time_step.observations["legal_actions"][p])
        return rl_agent.StepOutput(action=action, probs=None)


class SharedPPO(Algorithm):
    name = "ppo"
    supports_vector = True
    FLAG_KEYS = [
        "hidden_layers_sizes", "rl_learning_rate", "discount_factor", "ppo_rollout_episodes",
        "ppo_epochs", "ppo_minibatch", "ppo_clip", "ppo_ent_coef", "ppo_vf_coef",
        "ppo_gae_lambda", "ppo_max_grad_norm", "ppo_anneal_lr", "ppo_init", "ppo_critic",
        "device"]
    FLAG_DEFAULTS = {"rl_learning_rate": 3e-4}

    def __init__(self, game, *, hidden_layers_sizes, learning_rate, discount_factor,
                 rollout_episodes=32, epochs=5, minibatch=512, clip=0.2, ent_coef=0.01,
                 vf_coef=0.5, gae_lambda=0.95, max_grad_norm=0.5, anneal_episodes=0,
                 init="default", critic="local", device="cpu", vector_games=1):
        n = game.num_players()
        size, self.num_actions = game.information_state_tensor_size(), game.num_distinct_actions()
        self.device = torch.device(device)
        self.policy = mlp(size, hidden_layers_sizes, self.num_actions).to(self.device)
        if critic not in ("local", "central"):
            raise ValueError(f"Critico desconocido: {critic!r}")
        if critic == "central" and vector_games <= 1:
            raise ValueError("--ppo_critic central necesita --vector_games > 1")
        # Con critico central, VectorRunner le pasa a batch_step las manos ajenas.
        self.wants_hidden_hands = critic == "central"
        value_in = size + (HIDDEN_HANDS_SIZE if self.wants_hidden_hands else 0)
        self.value = mlp(value_in, hidden_layers_sizes, 1).to(self.device)
        if init == "orthogonal":
            _orthogonal_init(self.policy, 0.01)
            _orthogonal_init(self.value, 1.0)
        elif init != "default":
            raise ValueError(f"Inicializacion desconocida: {init!r}")
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=learning_rate)
        self.gamma, self.lam = discount_factor, gae_lambda
        self.rollout_episodes, self.epochs, self.minibatch = rollout_episodes, epochs, minibatch
        self.clip, self.ent_coef, self.vf_coef = clip, ent_coef, vf_coef
        self.max_grad_norm = max_grad_norm
        # Con anneal_episodes > 0, la tasa baja linealmente de learning_rate a 0
        # en ese numero de partidas (se ajusta antes de cada actualizacion).
        self.base_lr, self.anneal_episodes = learning_rate, anneal_episodes
        self.games_done = 0
        self._batch_start = 1
        self.n = n
        self.num_streams = n * max(1, vector_games)
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
            anneal_episodes=flags.num_train_episodes if flags.ppo_anneal_lr else 0,
            init=flags.ppo_init,
            critic=flags.ppo_critic,
            device=flags.device,
            vector_games=flags.vector_games,
        )

    # -- rollout -----------------------------------------------------------------

    def _reset_rollout(self):
        # Por stream (asiento, o asiento de cada partida con --vector_games):
        # listas paralelas de los pasos de la partida en curso y pasos ya
        # cerrados (con reward y done) de las partidas de este lote.
        self.traj = [{"obs": [], "mask": [], "act": [], "logp": [], "val": [], "rew": [],
                      "done": [], "hid": []} for _ in range(self.num_streams)]
        self.open = [False] * self.num_streams   # hay un paso esperando su reward
        self.games_in_rollout = 0

    def seats(self) -> list:
        return self._seats

    def begin_episode(self, episode: int) -> None:
        """Antes de cada partida: si ya hay --ppo_rollout_episodes partidas
        juntadas, actualizar."""
        if episode > 1:
            self.games_in_rollout += 1
        self.games_done = episode - 1
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
        self.open[p] = True
        return int(a)

    def close_step(self, p: int, reward: float, done: bool) -> None:
        t = self.traj[p]
        t["rew"].append(float(reward))
        t["done"].append(bool(done))
        self.open[p] = False

    # -- partidas en lote (rotten_rl/vector.py) --------------------------------

    def _value_input(self, obs: torch.Tensor, hidden) -> torch.Tensor:
        if not self.wants_hidden_hands:
            return obs
        return torch.cat([obs, torch.as_tensor(hidden, device=self.device,
                                               dtype=torch.float32)], dim=1)

    def batch_step(self, streams, infos, masks, legals, rewards, hidden=None) -> np.ndarray:
        for s, r in zip(streams, rewards):
            if self.open[s]:
                self.close_step(s, r, done=False)
        with torch.no_grad():
            x = torch.as_tensor(infos, device=self.device)
            logits = _masked_logits(self.policy(x), torch.as_tensor(masks, device=self.device))
            dist = torch.distributions.Categorical(logits=logits, validate_args=False)
            a = dist.sample()
            logp = dist.log_prob(a).cpu().numpy()
            v = self.value(self._value_input(x, hidden))[:, 0].cpu().numpy()
            a = a.cpu().numpy()
        for i, s in enumerate(streams):
            t = self.traj[s]
            t["obs"].append(infos[i])
            t["mask"].append(masks[i])
            t["act"].append(int(a[i]))
            t["logp"].append(float(logp[i]))
            t["val"].append(float(v[i]))
            if self.wants_hidden_hands:
                t["hid"].append(hidden[i].astype(bool))
            self.open[s] = True
        return a

    def batch_end(self, streams, rewards) -> None:
        for s, r in zip(streams, rewards):
            if self.open[s]:
                self.close_step(s, r, done=True)

    def begin_batch(self, episode: int, num_games: int) -> None:
        self._batch_start = episode

    def end_batch(self, num_games: int) -> None:
        """Despues de un lote de partidas (todas terminadas): actualizar si ya
        hay --ppo_rollout_episodes partidas juntadas."""
        self.games_in_rollout += num_games
        self.games_done = self._batch_start - 1 + num_games
        if self.games_in_rollout >= self.rollout_episodes:
            self._update()
            self._reset_rollout()

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
        if self.anneal_episodes:
            frac = max(0.0, 1.0 - self.games_done / self.anneal_episodes)
            for group in self.optimizer.param_groups:
                group["lr"] = self.base_lr * frac
        advs, rets = zip(*(self._advantages(t) for t in parts))
        dev = self.device
        obs = torch.as_tensor(np.concatenate([np.stack(t["obs"]) for t in parts]), device=dev)
        mask = torch.as_tensor(np.concatenate([np.stack(t["mask"]) for t in parts]), device=dev)
        act = torch.as_tensor(np.concatenate([t["act"] for t in parts]), device=dev)
        old_logp = torch.as_tensor(np.concatenate([t["logp"] for t in parts]), device=dev,
                                   dtype=torch.float32)
        adv = torch.as_tensor(np.concatenate(advs), device=dev)
        ret = torch.as_tensor(np.concatenate(rets), device=dev)
        hid = (torch.as_tensor(np.concatenate([np.stack(t["hid"]) for t in parts]), device=dev)
               if self.wants_hidden_hands else None)
        old_val = torch.as_tensor(np.concatenate([t["val"] for t in parts]), device=dev,
                                  dtype=torch.float32)
        explained_var = float(1 - (ret - old_val).var() / (ret.var() + 1e-8))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        size = len(act)
        stats = dict.fromkeys(_STATS_MEAN, 0.0)
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
                v_in = self._value_input(obs[idx], None if hid is None else hid[idx])
                loss_v = ((self.value(v_in)[:, 0] - ret[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = loss_pi + self.vf_coef * loss_v - self.ent_coef * ent
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm)
                self.optimizer.step()
                stats["loss_policy"] += float(loss_pi.detach())
                stats["loss_value"] += float(loss_v.detach())
                stats["entropy"] += float(ent.detach())
                stats["clip_frac"] += float(((ratio.detach() - 1).abs() > self.clip).float().mean())
                stats["approx_kl"] += float((old_logp[idx] - logp.detach()).mean())
                stats["grad_norm"] += float(grad_norm)
                count += 1
        self.last_stats = {k: v / count for k, v in stats.items()}
        self.last_stats["explained_variance"] = explained_var
        self.last_stats["learning_rate"] = self.optimizer.param_groups[0]["lr"]
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
        act.batch = lambda states, legals: greedy_batch(self.policy, states, legals,
                                                        self.num_actions, self.device)
        return act

    def frozen_state(self):
        return copy.deepcopy(self.policy.state_dict())

    def frozen_act(self, state, rng) -> ActFn:
        net = copy.deepcopy(self.policy)
        net.load_state_dict(state)
        net.eval()

        def act(s, legal):
            return self.greedy(info_state_tensor(s, s.current_player()), legal, net)
        act.batch = lambda states, legals: greedy_batch(net, states, legals, self.num_actions,
                                                        self.device)
        return act

    def losses(self) -> Dict[str, float]:
        out = dict.fromkeys(_STATS_MEAN + ("explained_variance", "learning_rate"), float("nan"))
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
        self.open = [False] * self.num_streams   # entre partidas: nada a medias
