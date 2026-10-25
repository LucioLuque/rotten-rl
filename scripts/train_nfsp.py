"""
scripts/train_nfsp.py  —  Fase 2: sanity check con NFSP + logging completo.

Entrena agentes NFSP (self-play) sobre `podrida` y evalua contra random, la
heuristica y la version anterior de si mismo (rotando el asiento evaluado),
registrando TODAS las metricas del documento, desglosadas por tamanio de
ronda h y por asiento, con:
  - progreso continuo cada --log_every episodios (reward movil + loss/agente),
  - evaluacion cada --eval_every episodios (incluye un baseline en el ep 0),
  - persistencia a TensorBoard + CSV + config.json, en runs/<nombre_unico>/.

El evaluador (rotten_rl/evaluation.py) y el logger (rotten_rl/run_logger.py) son modulos aparte.

Requisitos:
    pip install -e .        # desde la raiz del repo

Ejemplos:
    python scripts/train_nfsp.py --players 2 --num_train_episodes 1000000 \
        --eval_every 50000 --log_every 2000 --run_name baseline2p
    tensorboard --logdir runs
"""

import collections
import copy
import os

import numpy as np
import torch
from absl import app, flags, logging

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nfsp

import pyspiel
import rotten_rl.openspiel_game                    # registra el juego "podrida"
from rotten_rl.evaluation import evaluate, random_opponent, heuristic_opponent, network_opponent
from rotten_rl.run_logger import RunLogger

FLAGS = flags.FLAGS
# --- juego / evaluacion ---
flags.DEFINE_integer("players", 2, "Cantidad de jugadores (2..7).")
flags.DEFINE_string("reward_scheme", "raw", "raw | relative.")
flags.DEFINE_integer("num_train_episodes", 1000000, "Partidas de entrenamiento.")
flags.DEFINE_integer("eval_every", 25000, "Cada cuantas partidas evaluar.")
flags.DEFINE_integer("eval_games", 500, "Partidas por evaluacion.")
flags.DEFINE_list("eval_opponents", ["random", "heuristic", "past"],
                  "Rivales de cada evaluacion: random | heuristic | past (las "
                  "avg-nets de la evaluacion anterior).")
flags.DEFINE_bool("save_checkpoints", True,
                  "Guardar las avg-nets en runs/<corrida>/checkpoints/ en cada evaluacion.")
flags.DEFINE_integer("log_every", 2000, "Cada cuantas partidas loguear progreso de train.")
# --- red ---
flags.DEFINE_list("hidden_layers_sizes", [256, 256], "Capas ocultas (avg-net y Q-net).")
# --- best-response (DQN interno) ---
flags.DEFINE_integer("replay_buffer_capacity", int(2e5), "Replay buffer del DQN.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000, "Muestras minimas antes de aprender.")
flags.DEFINE_integer("batch_size", 128, "Batch de aprendizaje.")
flags.DEFINE_integer("learn_every", 64, "Pasos entre updates.")
flags.DEFINE_float("rl_learning_rate", 0.01, "LR del DQN interno.")
flags.DEFINE_float("discount_factor", 1.0, "Descuento gamma.")
# --- politica promedio (red supervisada) ---
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6), "Reservoir buffer (avg policy).")
flags.DEFINE_float("sl_learning_rate", 0.01, "LR de la red de politica promedio.")
# --- NFSP ---
flags.DEFINE_float("anticipatory_param", 0.1, "Prob de jugar la best-response por episodio.")
# --- genericos ---
flags.DEFINE_string("optimizer_str", "adam", "adam | sgd.")
flags.DEFINE_integer("seed", 0, "Semilla.")
flags.DEFINE_string("run_name", None, "Nombre base del experimento (se le agrega timestamp).")
flags.DEFINE_string("device", "cpu", "cpu | cuda (para chico casi no ayuda; ver notas).")


def _timestep_for(info_state, legal_actions, player):
    obs = {
        "info_state": [None] * (player + 1),
        "legal_actions": [None] * (player + 1),
        "current_player": player,
    }
    obs["info_state"][player] = info_state
    obs["legal_actions"][player] = legal_actions
    return rl_environment.TimeStep(observations=obs, rewards=None, discounts=None, step_type=None)


def make_nfsp_act_fn(agents):
    """act_fn(state, legal) usando la POLITICA PROMEDIO del agente del asiento
    que juega: cada agente se entreno en su asiento, y la evaluacion rota el
    asiento evaluado."""
    def act(state, legal):
        seat = state.current_player()
        info = state.information_state_tensor(seat)
        ts = _timestep_for(info, legal, seat)
        with agents[seat].temp_mode_as(nfsp.MODE.AVERAGE_POLICY):
            return agents[seat].step(ts, is_evaluation=True).action
    return act


def snapshot_avg_nets(agents):
    """Copia congelada de las avg-nets (una por asiento) para usar de rival."""
    nets = [copy.deepcopy(agent._avg_network) for agent in agents]
    for net in nets:
        net.eval()
    return nets


def main(_):
    np.random.seed(FLAGS.seed)
    rng = np.random.RandomState(FLAGS.seed)
    n = FLAGS.players
    hidden = [int(x) for x in FLAGS.hidden_layers_sizes]

    env = rl_environment.Environment("podrida", players=n, reward_scheme=FLAGS.reward_scheme)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    kwargs = dict(
        replay_buffer_capacity=FLAGS.replay_buffer_capacity,
        reservoir_buffer_capacity=FLAGS.reservoir_buffer_capacity,
        min_buffer_size_to_learn=FLAGS.min_buffer_size_to_learn,
        anticipatory_param=FLAGS.anticipatory_param,
        batch_size=FLAGS.batch_size,
        learn_every=FLAGS.learn_every,
        rl_learning_rate=FLAGS.rl_learning_rate,
        sl_learning_rate=FLAGS.sl_learning_rate,
        optimizer_str=FLAGS.optimizer_str,
        discount_factor=FLAGS.discount_factor,
        device=FLAGS.device,
    )
    agents = [nfsp.NFSP(idx, info_state_size, num_actions, hidden, **kwargs) for idx in range(n)]

    eval_game = pyspiel.load_game("podrida", {"players": n, "reward_scheme": FLAGS.reward_scheme})
    fixed_opponents = {"random": random_opponent(rng), "heuristic": heuristic_opponent(rng)}
    unknown = set(FLAGS.eval_opponents) - set(fixed_opponents) - {"past"}
    if unknown:
        raise ValueError(f"Rivales de evaluacion desconocidos: {sorted(unknown)}")
    nfsp_act = make_nfsp_act_fn(agents)
    past_nets = None          # avg-nets de la evaluacion anterior (rival "past")

    config = {f: getattr(FLAGS, f) for f in [
        "players", "reward_scheme", "num_train_episodes", "eval_every", "eval_games",
        "eval_opponents", "save_checkpoints", "hidden_layers_sizes", "replay_buffer_capacity",
        "reservoir_buffer_capacity", "min_buffer_size_to_learn", "batch_size",
        "learn_every", "rl_learning_rate", "sl_learning_rate", "anticipatory_param",
        "discount_factor", "optimizer_str", "seed", "device"]}
    config.update(info_state_size=info_state_size, num_actions=num_actions)
    log = RunLogger(run_name=FLAGS.run_name, config=config)
    logging.info("Corrida: %s  (info_state=%d, acciones=%d)", log.name, info_state_size, num_actions)

    def run_eval(ep):
        """Evalua contra cada rival de --eval_opponents (rotando asiento) y
        devuelve las metricas del rival mas exigente que se haya corrido."""
        nonlocal past_nets
        results = {}
        for name in FLAGS.eval_opponents:
            if name == "past":
                if past_nets is None:          # en la primera evaluacion no hay pasado
                    continue
                opponent = network_opponent(past_nets, rng)
            else:
                opponent = fixed_opponents[name]
            m = evaluate(eval_game, nfsp_act, opponent, n, FLAGS.eval_games, rng)
            log.log_eval(ep, m, opponent=name)
            results[name] = m
            o = m["overall"]
            logging.info(
                "[ep %d] EVAL vs %-9s | win=%.3f (empate=%.3f) diff=%+.2f exact_bid=%.3f "
                "MAE=%.3f sesgo=%+.3f normS=%.3f",
                ep, name, o["win_rate"], 1.0 / n, o["mean_score_diff"],
                o["exact_bid_rate"], o["mae_bid"], o["bid_bias"], o["norm_round_score"])

        past_nets = snapshot_avg_nets(agents)
        if FLAGS.save_checkpoints:
            ckpt_dir = os.path.join(log.run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save([net.state_dict() for net in past_nets],
                       os.path.join(ckpt_dir, f"avg_nets_ep{ep}.pt"))

        for name in ("heuristic", "random", "past"):
            if name in results:
                return results[name]
        return None

    # Baseline en el episodio 0 (redes sin entrenar ~ jugador aleatorio).
    run_eval(0)

    reward_window = collections.deque(maxlen=FLAGS.log_every)
    last_metrics = None
    for ep in range(1, FLAGS.num_train_episodes + 1):
        time_step = env.reset()
        ep_reward = 0.0
        while not time_step.last():
            pid = time_step.observations["current_player"]
            action = agents[pid].step(time_step).action
            time_step = env.step([action])
            if time_step.rewards is not None:
                ep_reward += time_step.rewards[0]     # suma de rewards del asiento 0
        for agent in agents:
            agent.step(time_step)
        reward_window.append(ep_reward)               # = retorno de la partida (asiento 0)

        if ep % FLAGS.log_every == 0:
            reward_ma = float(np.mean(reward_window)) if reward_window else 0.0
            losses = [agent.loss for agent in agents]
            log.log_train(ep, reward_ma, losses)
            logging.info("[ep %d] train | reward_ma(seat0)=%.3f", ep, reward_ma)

        if ep % FLAGS.eval_every == 0:
            last_metrics = run_eval(ep)

    log.close(final_metrics=last_metrics)
    logging.info("Listo. Resultados en %s   (tensorboard --logdir runs)", log.run_dir)


if __name__ == "__main__":
    app.run(main)
