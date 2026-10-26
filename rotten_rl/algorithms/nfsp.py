"""
rotten_rl/algorithms/nfsp.py

NFSP de OpenSpiel, un agente por asiento (aprendizaje independiente). Es lo que
hacia scripts/train_nfsp.py antes de tener varios algoritmos; el comportamiento
es el mismo (mismos resultados con la misma semilla).
"""

from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn, nfsp

from ..buffers import CompactReservoirBuffer, use_compact_buffers
from ..evaluation import network_opponent
from ..openspiel_game import info_state_tensor
from .base import ActFn, Algorithm


def make_nfsp_agents(game, *, hidden_layers_sizes, compact_buffers=True, **kwargs) -> List:
    """Un nfsp.NFSP por asiento. `kwargs` va directo al constructor de NFSP."""
    if compact_buffers:
        kwargs["replay_buffer_class"] = CompactReservoirBuffer      # el reservoir de NFSP
    agents = [nfsp.NFSP(i, game.information_state_tensor_size(), game.num_distinct_actions(),
                        hidden_layers_sizes, **kwargs)
              for i in range(game.num_players())]
    if compact_buffers:
        for agent in agents:
            use_compact_buffers(agent)                              # el replay del DQN interno
    return agents


def _timestep_for(info_state, legal_actions, player):
    obs = {
        "info_state": [None] * (player + 1),
        "legal_actions": [None] * (player + 1),
        "current_player": player,
    }
    obs["info_state"][player] = info_state
    obs["legal_actions"][player] = legal_actions
    return rl_environment.TimeStep(observations=obs, rewards=None, discounts=None, step_type=None)


class NFSPAlgorithm(Algorithm):
    """NFSP con un agente por asiento. Se evalua la politica promedio."""

    name = "nfsp"
    FLAG_KEYS = [
        "hidden_layers_sizes", "replay_buffer_capacity", "reservoir_buffer_capacity",
        "min_buffer_size_to_learn", "batch_size", "learn_every", "rl_learning_rate",
        "sl_learning_rate", "anticipatory_param", "discount_factor", "epsilon_start",
        "epsilon_end", "epsilon_decay_duration", "optimizer_str", "compact_buffers", "device"]

    def __init__(self, agents: List):
        self.agents = agents

    @classmethod
    def from_flags(cls, game, flags) -> "NFSPAlgorithm":
        return cls(make_nfsp_agents(
            game,
            hidden_layers_sizes=[int(x) for x in flags.hidden_layers_sizes],
            compact_buffers=flags.compact_buffers,
            replay_buffer_capacity=flags.replay_buffer_capacity,
            reservoir_buffer_capacity=flags.reservoir_buffer_capacity,
            min_buffer_size_to_learn=flags.min_buffer_size_to_learn,
            anticipatory_param=flags.anticipatory_param,
            batch_size=flags.batch_size,
            learn_every=flags.learn_every,
            rl_learning_rate=flags.rl_learning_rate,
            sl_learning_rate=flags.sl_learning_rate,
            optimizer_str=flags.optimizer_str,
            discount_factor=flags.discount_factor,
            epsilon_start=flags.epsilon_start,
            epsilon_end=flags.epsilon_end,
            epsilon_decay_duration=flags.epsilon_decay_duration,
            device=flags.device,
        ))

    # -- jugar ---------------------------------------------------------------

    def seats(self) -> list:
        return self.agents

    def eval_act(self) -> ActFn:
        """La POLITICA PROMEDIO del agente del asiento que juega: cada agente se
        entreno en su asiento, y la evaluacion rota el asiento evaluado."""
        agents = self.agents

        def act(state, legal):
            seat = state.current_player()
            ts = _timestep_for(info_state_tensor(state, seat), legal, seat)
            with agents[seat].temp_mode_as(nfsp.MODE.AVERAGE_POLICY):
                return agents[seat].step(ts, is_evaluation=True).action
        return act

    def frozen_state(self):
        return [copy.deepcopy(agent._avg_network.state_dict()) for agent in self.agents]

    def frozen_act(self, state, rng: np.random.RandomState) -> ActFn:
        nets = []
        for agent, sd in zip(self.agents, state):
            net = copy.deepcopy(agent._avg_network)
            net.load_state_dict(sd)
            net.eval()
            nets.append(net)
        return network_opponent(nets, rng)

    def losses(self) -> Dict[str, float]:
        out = {}
        for i, agent in enumerate(self.agents):
            sl, rl = (list(agent.loss) + [None, None])[:2]
            out[f"loss_agent_{i}_sl"] = float(sl) if sl is not None else float("nan")
            out[f"loss_agent_{i}_rl"] = float(rl) if rl is not None else float("nan")
        return out

    # -- checkpoints ---------------------------------------------------------
    # Mismo formato que antes de rotten_rl.algorithms ("agents" en meta.pt y
    # buffers/agent<i>.{reservoir,replay}.*.npy): las corridas viejas se retoman.

    def checkpoint_state(self) -> Dict:
        return {"agents": [_agent_state(a) for a in self.agents]}

    def checkpoint_buffers(self) -> Dict[str, object]:
        out = {}
        for i, agent in enumerate(self.agents):
            out[f"agent{i}.reservoir"] = agent._reservoir_buffer
            out[f"agent{i}.replay"] = agent._rl_agent._replay_buffer
        return out

    def load_checkpoint_state(self, meta: Dict, load_buffer) -> None:
        for i, (agent, s) in enumerate(zip(self.agents, meta["agents"])):
            _load_agent_state(agent, s)
            reservoir = load_buffer(f"agent{i}.reservoir", agent._replay_buffer_class,
                                    nfsp.Transition)
            replay = load_buffer(f"agent{i}.replay", agent._rl_agent._replay_buffer_class,
                                 dqn.Transition)
            if reservoir is not None:
                agent._reservoir_buffer = reservoir
            if replay is not None:
                agent._rl_agent._replay_buffer = replay


def _agent_state(agent) -> Dict:
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


def _load_agent_state(agent, s: Dict) -> None:
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
