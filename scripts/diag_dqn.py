"""
scripts/diag_dqn.py  —  Diagnostico: DQN solo contra rivales fijos.

Si el entorno y el reward permiten aprender, un DQN contra rivales al azar
tiene que despegarse rapido del nivel aleatorio (sobre todo en el canto). Es
el caso mas facil posible: si esto no aprende, el problema no es NFSP.

El DQN juega siempre en el asiento 0 y los demas asientos los juega
--opponent. Se evalua con la politica greedy (epsilon = 0), en el asiento 0.

Ejemplo:
    OMP_NUM_THREADS=1 python scripts/diag_dqn.py --players 2 --num_train_episodes 20000
"""

import numpy as np
from absl import app, flags, logging

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn

import pyspiel
import rotten_rl.openspiel_game                # registra el juego "podrida"
from rotten_rl.evaluation import evaluate, heuristic_opponent, random_opponent
from rotten_rl.run_logger import RunLogger
from rotten_rl.training import with_pending_reward

FLAGS = flags.FLAGS
flags.DEFINE_integer("players", 2, "Cantidad de jugadores.")
flags.DEFINE_string("opponent", "random", "Rival de entrenamiento: random | heuristic.")
flags.DEFINE_integer("num_train_episodes", 20000, "Partidas de entrenamiento.")
flags.DEFINE_integer("eval_every", 2000, "Cada cuantas partidas evaluar.")
flags.DEFINE_integer("eval_games", 300, "Partidas por evaluacion.")
flags.DEFINE_list("hidden_layers_sizes", [256, 256], "Capas ocultas.")
flags.DEFINE_float("learning_rate", 1e-3, "LR del DQN.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e5), "Replay buffer.")
flags.DEFINE_integer("batch_size", 128, "Batch.")
flags.DEFINE_integer("learn_every", 10, "Pasos entre updates.")
flags.DEFINE_integer("update_target_network_every", 1000, "Pasos entre copias a la target net.")
flags.DEFINE_float("epsilon_start", 1.0, "Epsilon inicial.")
flags.DEFINE_float("epsilon_end", 0.05, "Epsilon final.")
flags.DEFINE_integer("epsilon_decay_duration", int(3e5), "Pasos del DQN para bajar epsilon.")
flags.DEFINE_integer("seed", 0, "Semilla.")
flags.DEFINE_string("run_name", "diag_dqn", "Nombre base del experimento.")


def main(_):
    np.random.seed(FLAGS.seed)
    rng = np.random.RandomState(FLAGS.seed)
    n = FLAGS.players
    env = rl_environment.Environment("podrida", players=n)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agent = dqn.DQN(
        0, info_state_size, num_actions,
        hidden_layers_sizes=[int(x) for x in FLAGS.hidden_layers_sizes],
        replay_buffer_capacity=FLAGS.replay_buffer_capacity,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        learn_every=FLAGS.learn_every,
        update_target_network_every=FLAGS.update_target_network_every,
        epsilon_start=FLAGS.epsilon_start,
        epsilon_end=FLAGS.epsilon_end,
        epsilon_decay_duration=FLAGS.epsilon_decay_duration,
        epsilon_decay_schedule_str="linear",
        optimizer_str="adam",
        discount_factor=1.0,
        seed=FLAGS.seed,
    )
    train_opp = heuristic_opponent(rng) if FLAGS.opponent == "heuristic" else random_opponent(rng)

    eval_game = pyspiel.load_game("podrida", {"players": n})
    opponents = {"random": random_opponent(rng), "heuristic": heuristic_opponent(rng)}

    def greedy_act(state, legal):
        info = np.asarray(state.information_state_tensor(0), dtype=np.float32)
        action, _ = agent.act_epsilon_greedy(info, legal, epsilon=0.0)
        return int(action)

    log = RunLogger(run_name=FLAGS.run_name, config={
        f: getattr(FLAGS, f) for f in [
            "players", "opponent", "num_train_episodes", "eval_every", "eval_games",
            "hidden_layers_sizes", "learning_rate", "replay_buffer_capacity", "batch_size",
            "learn_every", "update_target_network_every", "epsilon_start", "epsilon_end",
            "epsilon_decay_duration", "seed"]})

    def run_eval(ep):
        for name, opp in opponents.items():
            m = evaluate(eval_game, greedy_act, opp, n, FLAGS.eval_games, rng, seat=0)
            log.log_eval(ep, m, opponent=name)
            o = m["overall"]
            logging.info(
                "[ep %d] EVAL vs %-9s | win=%.3f (empate=%.3f) diff=%+.2f exact_bid=%.3f "
                "MAE=%.3f sesgo=%+.3f", ep, name, o["win_rate"], 1.0 / n,
                o["mean_score_diff"], o["exact_bid_rate"], o["mae_bid"], o["bid_bias"])

    run_eval(0)
    for ep in range(1, FLAGS.num_train_episodes + 1):
        time_step = env.reset()
        pending = [0.0] * n
        while not time_step.last():
            pid = time_step.observations["current_player"]
            if pid == 0:
                action = agent.step(with_pending_reward(time_step, pending, 0)).action
            else:
                state = env.get_state
                action = train_opp(state, time_step.observations["legal_actions"][pid])
            time_step = env.step([action])
            for i in range(n):
                pending[i] += time_step.rewards[i]
        agent.step(with_pending_reward(time_step, pending, 0))

        if ep % FLAGS.eval_every == 0:
            logging.info("[ep %d] epsilon=%.3f loss=%s", ep,
                         agent.epsilon_schedule(agent._iteration), agent.loss)
            run_eval(ep)

    log.close()
    logging.info("Listo. Resultados en %s", log.run_dir)


if __name__ == "__main__":
    app.run(main)
