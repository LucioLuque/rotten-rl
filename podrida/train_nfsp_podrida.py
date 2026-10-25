"""
train_nfsp_podrida.py  —  Fase 2: sanity check con NFSP + logging completo.

Entrena agentes NFSP (self-play) sobre `podrida` y evalua contra random (o
heuristica), registrando TODAS las metricas del documento, desglosadas por
tamanio de ronda h, con:
  - progreso continuo cada --log_every episodios (reward movil + loss/agente),
  - evaluacion cada --eval_every episodios (incluye un baseline en el ep 0),
  - persistencia a TensorBoard + CSV + config.json, en runs/<nombre_unico>/.

El evaluador (evaluation.py) y el logger (run_logger.py) son modulos aparte.

Requisitos:
    pip install open_spiel torch dm-tree tensorboard
    # con podrida.py, podrida_openspiel.py, evaluation.py, run_logger.py al lado

Ejemplos:
    python train_nfsp_podrida.py --players 2 --num_train_episodes 1000000 \
        --eval_every 50000 --log_every 2000 --run_name baseline2p
    tensorboard --logdir runs
"""

import collections

import numpy as np
from absl import app, flags, logging

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nfsp

import pyspiel
import podrida_openspiel                      # registra el juego "podrida"
from evaluation import evaluate, random_opponent, heuristic_opponent
from run_logger import RunLogger

FLAGS = flags.FLAGS
# --- juego / evaluacion ---
flags.DEFINE_integer("players", 2, "Cantidad de jugadores (2..7).")
flags.DEFINE_string("reward_scheme", "raw", "raw | relative.")
flags.DEFINE_integer("num_train_episodes", 1000000, "Partidas de entrenamiento.")
flags.DEFINE_integer("eval_every", 25000, "Cada cuantas partidas evaluar.")
flags.DEFINE_integer("eval_games", 500, "Partidas por evaluacion.")
flags.DEFINE_string("eval_opponent", "random", "random | heuristic.")
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


def make_nfsp_act_fn(agents, seat):
    """act_fn(state, legal) usando la POLITICA PROMEDIO del agente `seat`."""
    def act(state, legal):
        info = state.information_state_tensor(seat)
        ts = _timestep_for(info, legal, seat)
        with agents[seat].temp_mode_as(nfsp.MODE.AVERAGE_POLICY):
            return agents[seat].step(ts, is_evaluation=True).action
    return act


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
    opponent = (heuristic_opponent(rng) if FLAGS.eval_opponent == "heuristic"
                else random_opponent(rng))
    nfsp_act = make_nfsp_act_fn(agents, seat=0)

    config = {f: getattr(FLAGS, f) for f in [
        "players", "reward_scheme", "num_train_episodes", "eval_every", "eval_games",
        "eval_opponent", "hidden_layers_sizes", "replay_buffer_capacity",
        "reservoir_buffer_capacity", "min_buffer_size_to_learn", "batch_size",
        "learn_every", "rl_learning_rate", "sl_learning_rate", "anticipatory_param",
        "discount_factor", "optimizer_str", "seed", "device"]}
    config.update(info_state_size=info_state_size, num_actions=num_actions)
    log = RunLogger(run_name=FLAGS.run_name, config=config)
    logging.info("Corrida: %s  (info_state=%d, acciones=%d)", log.name, info_state_size, num_actions)

    def run_eval(ep):
        m = evaluate(eval_game, nfsp_act, opponent, n, FLAGS.eval_games, rng, seat=0)
        log.log_eval(ep, m)
        o = m["overall"]
        logging.info(
            "[ep %d] EVAL vs %s | win=%.3f diff=%+.2f exact_bid=%.3f MAE=%.3f sesgo=%+.3f normS=%.3f",
            ep, FLAGS.eval_opponent, o["win_rate"], o["mean_score_diff"],
            o["exact_bid_rate"], o["mae_bid"], o["bid_bias"], o["norm_round_score"])
        return m

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
