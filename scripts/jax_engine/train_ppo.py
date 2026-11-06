"""
scripts/jax_engine/train_ppo.py  —  Train PPO entirely in JAX on the GPU.

Same recipe as `scripts/train.py --algo ppo --round_episodes --ppo_anneal_lr` (see
rotten_rl/jax_engine/ppo.py), ~30 times faster per run. Logging and evaluation are those of
train.py: every --eval_every games the policy is saved as <run>/checkpoints/policy_ep<N>.pt (a
PyTorch state_dict) and evaluated against random, the heuristic and its previous snapshot ("past",
greedy) with rotten_rl/evaluation.py (duplicate deals, --eval_deal_seed), written with RunLogger
(eval_<opponent>_overall/by_h/by_seat/... CSVs, train.csv, TensorBoard). So the runs work with
tournaments, play.py and the backups like any other.

The whole training state goes to <run>/state/ckpt.pkl every --checkpoint_every games and when it
stops (--stop_at, SIGTERM/SIGHUP, Ctrl+C); --resume <run> continues it (with --games, extending the
horizon). Accepts --runs_dir, --run_name and --stop_at like train.py, so run_queue.py can launch it.

On the lab it runs from its own venv (~/scratch/work/venvs/jax). Next to PyTorch runs, set
XLA_PYTHON_CLIENT_PREALLOCATE=false (run_queue.py does it).

    python scripts/jax_engine/train_ppo.py --players 2 --games 1000000 --rollout_games 512 \
        --run_name j23 --runs_dir ~/scratch/runs --stop_at lab
"""

import argparse
import json
import os
import pickle
import signal
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from rotten_rl.jax_engine import ppo  # noqa: E402
from rotten_rl.lab_hours import Deadline  # noqa: E402

CONFIG_KEYS = ("players", "games", "rollout_games", "hidden", "learning_rate", "epochs",
               "minibatch", "reward_scheme", "seed")
EVAL_OPPONENTS = ("random", "heuristic", "past")


def torch_policy(params, hidden):
    from rotten_rl.algorithms.dqn import mlp
    net = mlp(808, list(hidden), 60)
    net.load_state_dict(ppo.to_torch_state_dict(params))
    return net.eval()


def greedy_act(net):
    """An evaluation agent that plays the most probable legal action, deciding in batches."""
    import torch
    from rotten_rl.algorithms.dqn import greedy_batch

    def act(state, legal):
        return int(act.batch([state], [legal])[0])
    act.batch = lambda states, legals: greedy_batch(net, states, legals, 60, torch.device("cpu"))
    return act


def save_checkpoint(run_dir, ts):
    """The whole training state (networks, Adam, games played, RNG key), written atomically."""
    state_dir = os.path.join(run_dir, "state")
    os.makedirs(state_dir, exist_ok=True)
    tmp = os.path.join(state_dir, "ckpt.pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, ts), f)
    os.replace(tmp, os.path.join(state_dir, "ckpt.pkl"))


def load_checkpoint(run_dir):
    with open(os.path.join(run_dir, "state", "ckpt.pkl"), "rb") as f:
        return jax.tree_util.tree_map(jax.numpy.asarray, pickle.load(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--games", type=int, default=1_000_000, help="total matches (lr reaches 0 here)")
    p.add_argument("--rollout_games", type=int, default=512)
    p.add_argument("--hidden", default="512,512")
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--reward_scheme", default="relative")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=25000)
    p.add_argument("--eval_games", type=int, default=1500)
    p.add_argument("--eval_deal_seed", type=int, default=1234)
    p.add_argument("--checkpoint_every", type=int, default=25000)
    p.add_argument("--no_eval", action="store_true", help="only train and save (to measure speed)")
    p.add_argument("--stop_at", default=None,
                   help="HH:MM, 'YYYY-MM-DD HH:MM' or lab (rotten_rl/lab_hours.py): save and stop")
    p.add_argument("--resume", default=None, help="run folder to continue (its config is used)")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--run_name", default="jax_ppo")
    a = p.parse_args()

    if a.resume:
        run_dir = os.path.expanduser(a.resume).rstrip("/")
        with open(os.path.join(run_dir, "config.json")) as f:
            saved = json.load(f)
        for k in CONFIG_KEYS:
            if k == "games" and "--games" in sys.argv:
                continue                     # extending the horizon is allowed (the lr follows it)
            setattr(a, k, saved["jax_args"][k])
    hidden = tuple(int(x) for x in str(a.hidden).split(","))
    cfg = ppo.Config(num_players=a.players, hidden=hidden, learning_rate=a.learning_rate,
                     rollout_games=a.rollout_games, epochs=a.epochs, minibatch=a.minibatch,
                     total_games=a.games, reward_scheme=a.reward_scheme)
    config = {"algo": "ppo", "engine": "jax", "players": a.players,
              "hidden_layers_sizes": [str(h) for h in hidden], "num_train_episodes": a.games,
              "ppo_rollout_episodes": a.rollout_games, "ppo_epochs": a.epochs,
              "ppo_minibatch": a.minibatch, "rl_learning_rate": a.learning_rate,
              "reward_scheme": a.reward_scheme, "round_episodes": True, "ppo_anneal_lr": True,
              "eval_games": a.eval_games, "eval_deal_seed": a.eval_deal_seed, "seed": a.seed,
              "jax_args": {k: getattr(a, k) for k in CONFIG_KEYS}}

    import pyspiel
    import torch
    import rotten_rl.openspiel_game  # noqa: F401  registers "podrida"
    from rotten_rl.evaluation import evaluate, heuristic_opponent, random_opponent
    from rotten_rl.run_logger import RunLogger

    if a.resume:
        ts = load_checkpoint(run_dir)
        log = RunLogger(run_dir=run_dir, config=config, resume_after=int(ts.games_done))
    else:
        ts = ppo.init(jax.random.PRNGKey(a.seed), cfg)
        log = RunLogger(base_dir=os.path.expanduser(a.runs_dir), run_name=a.run_name, config=config)
        run_dir = log.run_dir
    snap_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(snap_dir, exist_ok=True)
    print(f"JAX {jax.__version__} on {jax.devices()}; run in {run_dir}"
          + (f"; resuming at game {int(ts.games_done)}" if a.resume else ""), flush=True)

    game = pyspiel.load_game("podrida", {"players": a.players, "reward_scheme": a.reward_scheme})
    rng = np.random.RandomState(a.seed)
    fixed = {"random": random_opponent(rng), "heuristic": heuristic_opponent(rng)}
    past = {"net": None}
    if a.resume:                             # the snapshot of the last evaluation is "past"
        snaps = [int(f[len("policy_ep"):-3]) for f in os.listdir(snap_dir) if f.startswith("policy_ep")]
        snaps = [e for e in snaps if e <= int(ts.games_done)]
        if snaps:
            net = torch_policy(ts.policy, hidden)
            net.load_state_dict(torch.load(os.path.join(snap_dir, f"policy_ep{max(snaps)}.pt")))
            past["net"] = net.eval()

    def save_and_eval(games):
        torch.save(ppo.to_torch_state_dict(ts.policy), os.path.join(snap_dir, f"policy_ep{games}.pt"))
        net = torch_policy(ts.policy, hidden)
        if not a.no_eval:
            t0 = time.perf_counter()
            for name in EVAL_OPPONENTS:
                if name == "past":
                    if past["net"] is None:
                        continue
                    opponent = greedy_act(past["net"])
                else:
                    opponent = fixed[name]
                m = evaluate(game, greedy_act(net), opponent, a.players, a.eval_games, rng,
                             parallel=True, deal_seed=None if a.eval_deal_seed < 0 else a.eval_deal_seed)
                log.log_eval(games, m, opponent=name)
                o = m["overall"]
                print(f"[{games}] EVAL vs {name:9s} | win={o['win_rate']:.3f}±{o['win_rate_se']:.3f} "
                      f"(empate={1 / a.players:.3f}) diff={o['mean_score_diff']:+.2f}±"
                      f"{o['mean_score_diff_se']:.2f} exact_bid={o['exact_bid_rate']:.3f}", flush=True)
            print(f"[{games}] eval took {time.perf_counter() - t0:.0f} s", flush=True)
        past["net"] = net

    deadline = Deadline(a.stop_at) if a.stop_at else None
    if deadline is not None:
        print(f"stops at {deadline.when:%Y-%m-%d %H:%M}", flush=True)
    stop = {"why": None}

    def on_signal(signum, _frame):
        stop["why"] = signal.Signals(signum).name
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, on_signal)

    update = ppo.make_update(cfg)
    games = start_games = int(ts.games_done)
    if games == 0:
        save_and_eval(0)
    next_eval = (games // a.eval_every + 1) * a.eval_every
    next_ckpt = (games // a.checkpoint_every + 1) * a.checkpoint_every
    start = time.perf_counter()
    while games < a.games:
        ts, stats = update(ts)
        games = int(ts.games_done)
        stats = {k: float(v) for k, v in stats.items()}
        reward = stats.pop("reward_seat0")
        stats["games_per_second"] = (games - start_games) / (time.perf_counter() - start)
        log.log_train(games, reward, stats)
        if games >= next_eval or games >= a.games:
            print(f"[{games}] {stats['games_per_second']:,.0f} games/s | ent={stats['entropy']:.3f} "
                  f"kl={stats['approx_kl']:.4f} ev={stats['explained_variance']:.3f} "
                  f"lr={stats['learning_rate']:.2e}", flush=True)
            save_and_eval(games)
            next_eval += a.eval_every
        if deadline is not None and deadline.reached():
            stop["why"] = f"--stop_at {a.stop_at}"
        if games >= next_ckpt or games >= a.games or stop["why"]:
            save_checkpoint(run_dir, ts)
            next_ckpt += a.checkpoint_every
        if stop["why"]:
            log.close()
            print(f"Cortado en el episodio {games} ({stop['why']}). Para seguir: --resume {run_dir}",
                  flush=True)
            return
    log.close()
    print(f"Listo. Resultados en {run_dir}", flush=True)


if __name__ == "__main__":
    main()
