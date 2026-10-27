"""
scripts/train.py  —  Entrenamiento con self-play, para cualquier algoritmo de
rotten_rl/algorithms (--algo nfsp | dqn | ppo), con logging, evaluacion y checkpoints.

Evalua contra random, la heuristica y la version anterior de si mismo (rotando
el asiento evaluado), registrando TODAS las metricas del documento,
desglosadas por tamanio de ronda h y por asiento, con:
  - progreso continuo cada --log_every episodios (reward movil + losses),
  - evaluacion cada --eval_every episodios (incluye un baseline en el ep 0),
  - persistencia a TensorBoard + CSV + config.json, en <runs_dir>/<nombre_unico>/,
  - checkpoints completos cada --checkpoint_every episodios en <corrida>/state/.

Algoritmos:
  - nfsp: NFSP de OpenSpiel, un agente por asiento.
  - dqn:  DQN con una red compartida por todos los asientos (como en Wizard).
  - ppo:  PPO con politica y critico compartidos, self-play contra la politica actual.
--round_episodes hace que para los agentes cada ronda sea un episodio.
--vector_games K (dqn y ppo) juega K partidas a la vez y decide en lote
(rotten_rl/vector.py); con --device cuda es mucho mas rapido.

Cortar y retomar: con --stop_at HH:MM (o si llega SIGTERM/SIGHUP/SIGINT, como
en el reinicio del laboratorio) guarda un checkpoint al terminar la partida en
curso y sale. --resume <carpeta de la corrida> sigue desde el ultimo checkpoint
con la config guardada (solo se pueden cambiar flags que no afectan al modelo,
como --num_train_episodes o --eval_every).

Ejemplos:
    OMP_NUM_THREADS=1 python scripts/train.py --algo dqn --players 2 --round_episodes \
        --num_train_episodes 50000 --runs_dir ~/scratch/runs --run_name dqn2p --stop_at 06:55
    OMP_NUM_THREADS=1 python scripts/train.py --resume ~/scratch/runs/dqn2p_<timestamp>
    tensorboard --logdir runs
"""

import collections
import datetime
import json
import os
import signal

import numpy as np
import torch
from absl import app, flags, logging

import pyspiel
import rotten_rl.openspiel_game                   # registra el juego "podrida"
from rotten_rl.algorithms import ALGORITHMS
from rotten_rl.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from rotten_rl.evaluation import evaluate, random_opponent, heuristic_opponent
from rotten_rl.run_logger import RunLogger
from rotten_rl.training import EpisodeRunner
from rotten_rl.vector import VectorRunner

FLAGS = flags.FLAGS
flags.DEFINE_enum("algo", "nfsp", sorted(ALGORITHMS), "Algoritmo de entrenamiento.")
# --- juego / evaluacion ---
flags.DEFINE_integer("players", 2, "Cantidad de jugadores (2..7).")
flags.DEFINE_string("reward_scheme", "raw", "raw | relative.")
flags.DEFINE_bool("round_episodes", False,
                  "Para los agentes, cada ronda es un episodio (el valor de una decision no "
                  "incluye los puntos de las rondas siguientes).")
flags.DEFINE_integer("num_train_episodes", 1000000, "Partidas de entrenamiento (en total).")
flags.DEFINE_integer("vector_games", 1,
                     "Partidas de entrenamiento a la vez, decidiendo en lote (solo dqn y ppo; "
                     "1 = de a una, como siempre).")
flags.DEFINE_integer("eval_every", 25000, "Cada cuantas partidas evaluar.")
flags.DEFINE_integer("eval_games", 500, "Partidas por evaluacion.")
flags.DEFINE_list("eval_opponents", ["random", "heuristic", "past"],
                  "Rivales de cada evaluacion: random | heuristic | past (la "
                  "politica de la evaluacion anterior).")
flags.DEFINE_bool("save_eval_snapshots", True,
                  "Guardar la politica en <corrida>/checkpoints/ en cada evaluacion.")
flags.DEFINE_integer("log_every", 2000, "Cada cuantas partidas loguear progreso de train.")
# --- checkpoints / cortar y retomar ---
flags.DEFINE_string("runs_dir", "runs", "Carpeta donde se crean las corridas. En el lab: "
                    "~/scratch/runs (el home tiene cuota de 2 GB).")
flags.DEFINE_string("resume", None, "Carpeta de una corrida a retomar desde su ultimo checkpoint.")
flags.DEFINE_integer("checkpoint_every", 10000, "Cada cuantas partidas guardar un checkpoint.")
flags.DEFINE_bool("checkpoint_buffers", True,
                  "Guardar tambien los buffers (pesan GB; sin ellos, al retomar se vuelven a llenar).")
flags.DEFINE_integer("keep_checkpoints", 2, "Cuantos checkpoints completos conservar.")
flags.DEFINE_bool("compact_buffers", True,
                  "Guardar el info-state comprimido en los buffers (sin perdida, ~7 veces menos "
                  "RAM; ver rotten_rl/buffers.py).")
flags.DEFINE_string("stop_at", None, "Hora local HH:MM a la que cortar guardando un checkpoint "
                    "(p.ej. 06:55, antes del reinicio de las 07:30 del lab).")
# --- red ---
flags.DEFINE_list("hidden_layers_sizes", [256, 256], "Capas ocultas.")
# --- DQN (el de NFSP y el compartido) ---
flags.DEFINE_integer("replay_buffer_capacity", int(2e5), "Replay buffer del DQN.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000, "Muestras minimas antes de aprender.")
flags.DEFINE_integer("batch_size", 128, "Batch de aprendizaje.")
flags.DEFINE_integer("learn_every", 64, "Pasos entre updates.")
flags.DEFINE_float("rl_learning_rate", 0.01, "LR del DQN.")
flags.DEFINE_float("discount_factor", 1.0, "Descuento gamma.")
# Exploracion del DQN interno de NFSP. Los defaults de OpenSpiel 2.x (1.0 -> 0.1 en 1M pasos)
# dejan la best-response jugando casi al azar durante cientos de miles de episodios, y la
# avg-policy aprende a imitar eso. Estos son los valores del paper de NFSP / examples/nfsp.py.
# El DQN compartido tiene sus propios defaults (SharedDQN.FLAG_DEFAULTS).
flags.DEFINE_float("epsilon_start", 0.06, "Epsilon inicial del DQN.")
flags.DEFINE_float("epsilon_end", 0.001, "Epsilon final del DQN.")
flags.DEFINE_integer("epsilon_decay_duration", int(2e6),
                     "NFSP: pasos del DQN (no episodios) en que epsilon va de start a end.")
# --- solo DQN compartido ---
flags.DEFINE_float("dqn_epsilon_decay_fraction", 0.5,
                   "DQN: fraccion de --num_train_episodes en que epsilon va de start a end.")
flags.DEFINE_integer("dqn_target_update_every", 1000,
                     "DQN: cada cuantos pasos se copia la red online a la target.")
flags.DEFINE_enum("dqn_loss", "mse", ["mse", "huber"], "DQN: funcion de perdida.")
# --- solo PPO compartido ---
flags.DEFINE_integer("ppo_rollout_episodes", 32, "PPO: partidas por lote antes de actualizar.")
flags.DEFINE_integer("ppo_epochs", 5, "PPO: pasadas sobre cada lote.")
flags.DEFINE_integer("ppo_minibatch", 512, "PPO: tamanio de minibatch.")
flags.DEFINE_float("ppo_clip", 0.2, "PPO: clipping del ratio.")
flags.DEFINE_float("ppo_ent_coef", 0.01, "PPO: peso de la entropia.")
flags.DEFINE_float("ppo_vf_coef", 0.5, "PPO: peso de la perdida del critico.")
flags.DEFINE_float("ppo_gae_lambda", 0.95, "PPO: lambda de GAE.")
flags.DEFINE_float("ppo_max_grad_norm", 0.5, "PPO: clipping de la norma del gradiente.")
# --- solo NFSP: politica promedio ---
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6), "NFSP: reservoir (avg policy).")
flags.DEFINE_float("sl_learning_rate", 0.01, "NFSP: LR de la red de politica promedio.")
flags.DEFINE_float("anticipatory_param", 0.1, "NFSP: prob de jugar la best-response.")
# --- genericos ---
flags.DEFINE_string("optimizer_str", "adam", "adam | sgd (NFSP).")
flags.DEFINE_integer("seed", 0, "Semilla.")
flags.DEFINE_string("run_name", None, "Nombre base del experimento (se le agrega timestamp).")
flags.DEFINE_string("device", "cpu", "cpu | cuda (para chico casi no ayuda; ver notas).")

# Flags que se guardan en config.json: los comunes y los de cada algoritmo.
_COMMON_KEYS = [
    "algo", "players", "reward_scheme", "round_episodes", "vector_games", "num_train_episodes",
    "eval_every",
    "eval_games", "eval_opponents", "save_eval_snapshots", "log_every", "checkpoint_every",
    "checkpoint_buffers", "keep_checkpoints", "seed"]
_ALL_KEYS = list(dict.fromkeys(_COMMON_KEYS + [k for cls in ALGORITHMS.values()
                                                for k in cls.FLAG_KEYS]))
# Los que se pueden cambiar al retomar (no cambian el modelo ni el entrenamiento).
_RESUMABLE_OVERRIDES = {
    "num_train_episodes", "eval_every", "eval_games", "eval_opponents", "save_eval_snapshots",
    "log_every", "checkpoint_every", "checkpoint_buffers", "keep_checkpoints"}


def _apply_resume_config(run_dir):
    """Carga la config guardada de la corrida y la impone en FLAGS. Solo deja
    cambiar (pasandolos explicitamente) los flags de _RESUMABLE_OVERRIDES."""
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    for key, saved in config.items():
        if key not in _ALL_KEYS:
            continue
        if FLAGS[key].present and key in _RESUMABLE_OVERRIDES:
            continue
        if FLAGS[key].present and FLAGS[key].value != saved:
            raise ValueError(f"--{key} no se puede cambiar al retomar "
                             f"(guardado: {saved!r}, pasado: {FLAGS[key].value!r})")
        FLAGS[key].value = saved


def _apply_algo_defaults(cls):
    """Defaults propios del algoritmo para flags compartidos no pasados."""
    for key, value in cls.FLAG_DEFAULTS.items():
        if not FLAGS[key].present:
            FLAGS[key].value = value


def _deadline(stop_at):
    """Proximo instante local HH:MM (hoy si todavia no paso; si no, manana)."""
    if not stop_at:
        return None
    hh, mm = (int(x) for x in stop_at.split(":"))
    now = datetime.datetime.now()
    when = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if when <= now:
        when += datetime.timedelta(days=1)
    return when


def main(_):
    if FLAGS.resume:
        _apply_resume_config(FLAGS.resume)
    algo_cls = ALGORITHMS[FLAGS.algo]
    if not FLAGS.resume:
        _apply_algo_defaults(algo_cls)

    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    rng = np.random.RandomState(FLAGS.seed)
    n = FLAGS.players

    game = pyspiel.load_game("podrida", {"players": n, "reward_scheme": FLAGS.reward_scheme})
    info_state_size = game.information_state_tensor_size()
    num_actions = game.num_distinct_actions()
    algo = algo_cls.from_flags(game, FLAGS)

    fixed_opponents = {"random": random_opponent(rng), "heuristic": heuristic_opponent(rng)}
    unknown = set(FLAGS.eval_opponents) - set(fixed_opponents) - {"past"}
    if unknown:
        raise ValueError(f"Rivales de evaluacion desconocidos: {sorted(unknown)}")
    eval_act = algo.eval_act()
    past_state = None         # politica congelada de la evaluacion anterior (rival "past")

    keys = list(dict.fromkeys(_COMMON_KEYS + algo_cls.FLAG_KEYS))
    config = {f: getattr(FLAGS, f) for f in keys}
    config.update(info_state_size=info_state_size, num_actions=num_actions)

    start_ep = 0
    resume_extra = {}
    if FLAGS.resume:
        ckpt = latest_checkpoint(os.path.join(FLAGS.resume, "state"))
        if ckpt is None:
            raise ValueError(f"No hay checkpoints en {FLAGS.resume}/state")
        info = load_checkpoint(ckpt, algo, rng)
        start_ep = info["episode"]
        resume_extra = info["extra"]
        past_state = resume_extra.get("past_state")
        log = RunLogger(run_dir=FLAGS.resume, config=config, resume_after=start_ep)
        logging.info("Retomando %s desde %s (episodio %d)", log.name, ckpt, start_ep)
    else:
        log = RunLogger(base_dir=os.path.expanduser(FLAGS.runs_dir), run_name=FLAGS.run_name,
                        config=config)
        logging.info("Corrida: %s  (algo=%s, info_state=%d, acciones=%d)",
                     log.run_dir, FLAGS.algo, info_state_size, num_actions)
    state_dir = os.path.join(log.run_dir, "state")

    def run_eval(ep):
        """Evalua contra cada rival de --eval_opponents (rotando asiento) y
        devuelve las metricas del rival mas exigente que se haya corrido."""
        nonlocal past_state
        results = {}
        for name in FLAGS.eval_opponents:
            if name == "past":
                if past_state is None:         # en la primera evaluacion no hay pasado
                    continue
                opponent = algo.frozen_act(past_state, rng)
            else:
                opponent = fixed_opponents[name]
            m = evaluate(game, eval_act, opponent, n, FLAGS.eval_games, rng,
                         parallel=FLAGS.vector_games > 1)
            log.log_eval(ep, m, opponent=name)
            results[name] = m
            o = m["overall"]
            logging.info(
                "[ep %d] EVAL vs %-9s | win=%.3f (empate=%.3f) diff=%+.2f exact_bid=%.3f "
                "MAE=%.3f sesgo=%+.3f normS=%.3f",
                ep, name, o["win_rate"], 1.0 / n, o["mean_score_diff"],
                o["exact_bid_rate"], o["mae_bid"], o["bid_bias"], o["norm_round_score"])

        past_state = algo.frozen_state()
        if FLAGS.save_eval_snapshots:
            snap_dir = os.path.join(log.run_dir, "checkpoints")
            os.makedirs(snap_dir, exist_ok=True)
            torch.save(past_state, os.path.join(snap_dir, f"policy_ep{ep}.pt"))

        for name in ("heuristic", "random", "past"):
            if name in results:
                return results[name]
        return None

    def checkpoint(ep):
        extra = {"past_state": past_state, "reward_window": list(reward_window)}
        path = save_checkpoint(state_dir, ep, algo, rng, extra=extra,
                               with_buffers=FLAGS.checkpoint_buffers,
                               keep_last=FLAGS.keep_checkpoints)
        logging.info("[ep %d] checkpoint en %s", ep, path)

    # Cortar prolijo: la senal solo marca; se corta al terminar la partida en curso.
    stop = {"why": None}

    def _on_signal(signum, _frame):
        stop["why"] = signal.Signals(signum).name
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _on_signal)
    deadline = _deadline(FLAGS.stop_at)
    if deadline is not None:
        logging.info("Corta a las %s", deadline.strftime("%Y-%m-%d %H:%M"))

    if start_ep == 0:
        run_eval(0)               # baseline (redes sin entrenar ~ jugador aleatorio)

    if FLAGS.vector_games > 1 and not algo.supports_vector:
        raise ValueError(f"--vector_games no esta implementado para --algo {FLAGS.algo}")
    runner = EpisodeRunner(game, rng)
    vrunner = VectorRunner(game, rng) if FLAGS.vector_games > 1 else None
    reward_window = collections.deque(resume_extra.get("reward_window", []),
                                      maxlen=FLAGS.log_every)

    def next_stop(ep):
        """Proximo episodio en el que hay que loguear, evaluar, guardar o terminar:
        los lotes de --vector_games se achican para caer justo ahi."""
        every = (FLAGS.log_every, FLAGS.eval_every, FLAGS.checkpoint_every)
        return min([(ep // e + 1) * e for e in every] + [FLAGS.num_train_episodes])

    last_metrics = None
    ep = start_ep
    while ep < FLAGS.num_train_episodes:
        if vrunner is not None:
            k = min(FLAGS.vector_games, next_stop(ep) - ep)
            algo.begin_batch(ep + 1, k)
            reward_window.extend(vrunner.play(algo, k, round_episodes=FLAGS.round_episodes))
            algo.end_batch(k)
            ep += k
        else:
            ep += 1
            algo.begin_episode(ep)
            seat0 = [0.0]
            runner.play(algo.seats(), round_episodes=FLAGS.round_episodes,
                        on_round_reward=lambda r: seat0.__setitem__(0, seat0[0] + r[0]))
            reward_window.append(seat0[0])             # = retorno de la partida (asiento 0)

        if ep % FLAGS.log_every == 0:
            reward_ma = float(np.mean(reward_window)) if reward_window else 0.0
            log.log_train(ep, reward_ma, algo.losses())
            logging.info("[ep %d] train | reward_ma(seat0)=%.3f", ep, reward_ma)

        if ep % FLAGS.eval_every == 0:
            last_metrics = run_eval(ep)

        if deadline is not None and datetime.datetime.now() >= deadline:
            stop["why"] = f"--stop_at {FLAGS.stop_at}"
        if stop["why"] is not None:
            checkpoint(ep)
            log.close()
            logging.info("Cortado en el episodio %d (%s). Para seguir: --resume %s",
                         ep, stop["why"], log.run_dir)
            return
        if ep % FLAGS.checkpoint_every == 0:
            checkpoint(ep)

    if ep % FLAGS.checkpoint_every != 0:
        checkpoint(ep)
    log.close(final_metrics=last_metrics)
    logging.info("Listo. Resultados en %s   (tensorboard --logdir %s)",
                 log.run_dir, os.path.dirname(log.run_dir))


if __name__ == "__main__":
    app.run(main)
