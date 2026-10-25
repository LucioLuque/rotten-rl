"""
scripts/train_nfsp.py  —  Fase 2: NFSP con self-play, logging completo y checkpoints.

Entrena agentes NFSP (self-play) sobre `podrida` y evalua contra random, la
heuristica y la version anterior de si mismo (rotando el asiento evaluado),
registrando TODAS las metricas del documento, desglosadas por tamanio de
ronda h y por asiento, con:
  - progreso continuo cada --log_every episodios (reward movil + loss/agente),
  - evaluacion cada --eval_every episodios (incluye un baseline en el ep 0),
  - persistencia a TensorBoard + CSV + config.json, en <runs_dir>/<nombre_unico>/,
  - checkpoints completos cada --checkpoint_every episodios en <corrida>/state/.

Cortar y retomar: con --stop_at HH:MM (o si llega SIGTERM/SIGHUP/SIGINT, como
en el reinicio del laboratorio) guarda un checkpoint al terminar la partida en
curso y sale. --resume <carpeta de la corrida> sigue desde el ultimo checkpoint
con la config guardada (solo se pueden cambiar flags que no afectan al modelo,
como --num_train_episodes o --eval_every).

Requisitos:
    pip install -e .        # desde la raiz del repo

Ejemplos:
    OMP_NUM_THREADS=1 python scripts/train_nfsp.py --players 2 --num_train_episodes 1000000 \
        --runs_dir ~/scratch/runs --run_name nfsp2p --stop_at 06:55
    OMP_NUM_THREADS=1 python scripts/train_nfsp.py --resume ~/scratch/runs/nfsp2p_<timestamp> \
        --stop_at 06:55
    tensorboard --logdir runs
"""

import collections
import copy
import datetime
import json
import os
import signal

import numpy as np
import torch
from absl import app, flags, logging

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nfsp

import pyspiel
import rotten_rl.openspiel_game as og             # registra el juego "podrida"
from rotten_rl.buffers import CompactReservoirBuffer, use_compact_buffers
from rotten_rl.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from rotten_rl.evaluation import evaluate, random_opponent, heuristic_opponent, network_opponent
from rotten_rl.run_logger import RunLogger
from rotten_rl.training import EpisodeRunner

FLAGS = flags.FLAGS
# --- juego / evaluacion ---
flags.DEFINE_integer("players", 2, "Cantidad de jugadores (2..7).")
flags.DEFINE_string("reward_scheme", "raw", "raw | relative.")
flags.DEFINE_integer("num_train_episodes", 1000000, "Partidas de entrenamiento (en total).")
flags.DEFINE_integer("eval_every", 25000, "Cada cuantas partidas evaluar.")
flags.DEFINE_integer("eval_games", 500, "Partidas por evaluacion.")
flags.DEFINE_list("eval_opponents", ["random", "heuristic", "past"],
                  "Rivales de cada evaluacion: random | heuristic | past (las "
                  "avg-nets de la evaluacion anterior).")
flags.DEFINE_bool("save_eval_snapshots", True,
                  "Guardar las avg-nets en <corrida>/checkpoints/ en cada evaluacion.")
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
                  "RAM por agente; ver rotten_rl/buffers.py).")
flags.DEFINE_string("stop_at", None, "Hora local HH:MM a la que cortar guardando un checkpoint "
                    "(p.ej. 06:55, antes del reinicio de las 07:30 del lab).")
# --- red ---
flags.DEFINE_list("hidden_layers_sizes", [256, 256], "Capas ocultas (avg-net y Q-net).")
# --- best-response (DQN interno) ---
flags.DEFINE_integer("replay_buffer_capacity", int(2e5), "Replay buffer del DQN.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000, "Muestras minimas antes de aprender.")
flags.DEFINE_integer("batch_size", 128, "Batch de aprendizaje.")
flags.DEFINE_integer("learn_every", 64, "Pasos entre updates.")
flags.DEFINE_float("rl_learning_rate", 0.01, "LR del DQN interno.")
flags.DEFINE_float("discount_factor", 1.0, "Descuento gamma.")
# Exploracion del DQN. Los defaults de OpenSpiel 2.x (1.0 -> 0.1 en 1M pasos) dejan la
# best-response jugando casi al azar durante cientos de miles de episodios, y la avg-policy
# aprende a imitar eso. Estos son los valores del paper de NFSP / examples/nfsp.py.
flags.DEFINE_float("epsilon_start", 0.06, "Epsilon inicial del DQN.")
flags.DEFINE_float("epsilon_end", 0.001, "Epsilon final del DQN.")
flags.DEFINE_integer("epsilon_decay_duration", int(2e6),
                     "Pasos del DQN (no episodios) en que epsilon va de start a end.")
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

# Flags que se guardan en config.json.
_CONFIG_KEYS = [
    "players", "reward_scheme", "num_train_episodes", "eval_every", "eval_games",
    "eval_opponents", "save_eval_snapshots", "log_every", "checkpoint_every",
    "checkpoint_buffers", "keep_checkpoints", "compact_buffers", "hidden_layers_sizes",
    "replay_buffer_capacity",
    "reservoir_buffer_capacity", "min_buffer_size_to_learn", "batch_size", "learn_every",
    "rl_learning_rate", "sl_learning_rate", "anticipatory_param", "discount_factor",
    "epsilon_start", "epsilon_end", "epsilon_decay_duration", "optimizer_str", "seed", "device"]
# Los que se pueden cambiar al retomar (no cambian el modelo ni el entrenamiento).
_RESUMABLE_OVERRIDES = {
    "num_train_episodes", "eval_every", "eval_games", "eval_opponents", "save_eval_snapshots",
    "log_every", "checkpoint_every", "checkpoint_buffers", "keep_checkpoints"}


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
        ts = _timestep_for(og.info_state_tensor(state, seat), legal, seat)
        with agents[seat].temp_mode_as(nfsp.MODE.AVERAGE_POLICY):
            return agents[seat].step(ts, is_evaluation=True).action
    return act


def snapshot_avg_nets(agents):
    """Copia congelada de las avg-nets (una por asiento) para usar de rival."""
    nets = [copy.deepcopy(agent._avg_network) for agent in agents]
    for net in nets:
        net.eval()
    return nets


def _apply_resume_config(run_dir):
    """Carga la config guardada de la corrida y la impone en FLAGS. Solo deja
    cambiar (pasandolos explicitamente) los flags de _RESUMABLE_OVERRIDES."""
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    for key, saved in config.items():
        if key not in _CONFIG_KEYS:
            continue
        if FLAGS[key].present and key in _RESUMABLE_OVERRIDES:
            continue
        if FLAGS[key].present and FLAGS[key].value != saved:
            raise ValueError(f"--{key} no se puede cambiar al retomar "
                             f"(guardado: {saved!r}, pasado: {FLAGS[key].value!r})")
        FLAGS[key].value = saved


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

    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    rng = np.random.RandomState(FLAGS.seed)
    n = FLAGS.players
    hidden = [int(x) for x in FLAGS.hidden_layers_sizes]

    game = pyspiel.load_game("podrida", {"players": n, "reward_scheme": FLAGS.reward_scheme})
    info_state_size = game.information_state_tensor_size()
    num_actions = game.num_distinct_actions()

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
        epsilon_start=FLAGS.epsilon_start,
        epsilon_end=FLAGS.epsilon_end,
        epsilon_decay_duration=FLAGS.epsilon_decay_duration,
        device=FLAGS.device,
    )
    if FLAGS.compact_buffers:
        kwargs["replay_buffer_class"] = CompactReservoirBuffer   # el reservoir de NFSP
    agents = [nfsp.NFSP(idx, info_state_size, num_actions, hidden, **kwargs) for idx in range(n)]
    if FLAGS.compact_buffers:
        for agent in agents:
            use_compact_buffers(agent)                           # el replay del DQN interno

    fixed_opponents = {"random": random_opponent(rng), "heuristic": heuristic_opponent(rng)}
    unknown = set(FLAGS.eval_opponents) - set(fixed_opponents) - {"past"}
    if unknown:
        raise ValueError(f"Rivales de evaluacion desconocidos: {sorted(unknown)}")
    nfsp_act = make_nfsp_act_fn(agents)
    past_nets = None          # avg-nets de la evaluacion anterior (rival "past")

    config = {f: getattr(FLAGS, f) for f in _CONFIG_KEYS}
    config.update(info_state_size=info_state_size, num_actions=num_actions)

    start_ep = 0
    resume_extra = {}
    if FLAGS.resume:
        ckpt = latest_checkpoint(os.path.join(FLAGS.resume, "state"))
        if ckpt is None:
            raise ValueError(f"No hay checkpoints en {FLAGS.resume}/state")
        info = load_checkpoint(ckpt, agents, rng)
        start_ep = info["episode"]
        resume_extra = info["extra"]
        if info["extra"].get("past_nets") is not None:
            past_nets = snapshot_avg_nets(agents)
            for net, sd in zip(past_nets, info["extra"]["past_nets"]):
                net.load_state_dict(sd)
        log = RunLogger(run_dir=FLAGS.resume, config=config, resume_after=start_ep)
        logging.info("Retomando %s desde %s (episodio %d)", log.name, ckpt, start_ep)
    else:
        log = RunLogger(base_dir=os.path.expanduser(FLAGS.runs_dir), run_name=FLAGS.run_name,
                        config=config)
        logging.info("Corrida: %s  (info_state=%d, acciones=%d)",
                     log.run_dir, info_state_size, num_actions)
    state_dir = os.path.join(log.run_dir, "state")

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
            m = evaluate(game, nfsp_act, opponent, n, FLAGS.eval_games, rng)
            log.log_eval(ep, m, opponent=name)
            results[name] = m
            o = m["overall"]
            logging.info(
                "[ep %d] EVAL vs %-9s | win=%.3f (empate=%.3f) diff=%+.2f exact_bid=%.3f "
                "MAE=%.3f sesgo=%+.3f normS=%.3f",
                ep, name, o["win_rate"], 1.0 / n, o["mean_score_diff"],
                o["exact_bid_rate"], o["mae_bid"], o["bid_bias"], o["norm_round_score"])

        past_nets = snapshot_avg_nets(agents)
        if FLAGS.save_eval_snapshots:
            snap_dir = os.path.join(log.run_dir, "checkpoints")
            os.makedirs(snap_dir, exist_ok=True)
            torch.save([net.state_dict() for net in past_nets],
                       os.path.join(snap_dir, f"avg_nets_ep{ep}.pt"))

        for name in ("heuristic", "random", "past"):
            if name in results:
                return results[name]
        return None

    def checkpoint(ep):
        extra = {"past_nets": None if past_nets is None else
                 [net.state_dict() for net in past_nets],
                 "reward_window": list(reward_window)}
        path = save_checkpoint(state_dir, ep, agents, rng, extra=extra,
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

    runner = EpisodeRunner(game, rng)
    reward_window = collections.deque(resume_extra.get("reward_window", []),
                                      maxlen=FLAGS.log_every)
    last_metrics = None
    ep = start_ep
    while ep < FLAGS.num_train_episodes:
        ep += 1
        seat0 = [0.0]
        runner.play(agents, on_round_reward=lambda r: seat0.__setitem__(0, seat0[0] + r[0]))
        reward_window.append(seat0[0])                 # = retorno de la partida (asiento 0)

        if ep % FLAGS.log_every == 0:
            reward_ma = float(np.mean(reward_window)) if reward_window else 0.0
            losses = [agent.loss for agent in agents]
            log.log_train(ep, reward_ma, losses)
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
