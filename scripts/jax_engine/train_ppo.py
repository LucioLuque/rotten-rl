"""
scripts/jax_engine/train_ppo.py  —  Train PPO entirely in JAX (stage 3 prototype).

Same recipe as `scripts/train.py --algo ppo --round_episodes --ppo_anneal_lr` (see
rotten_rl/jax_engine/ppo.py). Every `--eval_every` games the policy is saved as
<run>/checkpoints/policy_ep<N>.pt (a PyTorch state_dict) and evaluated against the heuristic with
the usual evaluation (rotten_rl/evaluation.py: duplicate deals, same --eval_deal_seed as train.py),
so the run can be compared in tournaments with the PyTorch ones. The whole training state is saved
in <run>/state/ckpt.pkl every --checkpoint_every games and when it stops (--stop_at, SIGTERM, Ctrl+C);
--resume <run> continues it (with --games, extending the horizon).

    python scripts/jax_engine/train_ppo.py --players 2 --games 1000000 --rollout_games 512 \
        --run_name j23 --runs_dir ~/scratch/runs
"""

import argparse
import csv
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


def evaluate_vs_heuristic(params, players, games, deal_seed, hidden):
    import pyspiel
    import torch
    import rotten_rl.openspiel_game  # noqa: F401
    from rotten_rl.algorithms.dqn import greedy_batch, mlp
    from rotten_rl.evaluation import evaluate, heuristic_opponent
    game = pyspiel.load_game("podrida", {"players": players})
    net = mlp(808, list(hidden), 60)
    net.load_state_dict(ppo.to_torch_state_dict(params))
    net.eval()

    def act(state, legal):
        return int(act.batch([state], [legal])[0])
    act.batch = lambda states, legals: greedy_batch(net, states, legals, 60, torch.device("cpu"))
    rng = np.random.RandomState(0)
    m = evaluate(game, act, heuristic_opponent(rng), players, games, rng, parallel=True,
                 deal_seed=deal_seed)
    return m["overall"]


CONFIG_KEYS = ("players", "games", "rollout_games", "hidden", "learning_rate", "epochs",
               "minibatch", "reward_scheme", "seed")


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
    else:
        run_dir = os.path.join(os.path.expanduser(a.runs_dir),
                               f"{a.run_name}_{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(os.path.join(run_dir, "checkpoints"))
    hidden = tuple(int(x) for x in str(a.hidden).split(","))
    cfg = ppo.Config(num_players=a.players, hidden=hidden, learning_rate=a.learning_rate,
                     rollout_games=a.rollout_games, epochs=a.epochs, minibatch=a.minibatch,
                     total_games=a.games, reward_scheme=a.reward_scheme)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"algo": "ppo", "engine": "jax", "players": a.players,
                   "hidden_layers_sizes": [str(h) for h in hidden], "num_train_episodes": a.games,
                   "ppo_rollout_episodes": a.rollout_games, "ppo_epochs": a.epochs,
                   "ppo_minibatch": a.minibatch, "rl_learning_rate": a.learning_rate,
                   "reward_scheme": a.reward_scheme, "round_episodes": True, "ppo_anneal_lr": True,
                   "seed": a.seed, "jax_args": {k: getattr(a, k) for k in CONFIG_KEYS}}, f, indent=1)
    print(f"JAX {jax.__version__} on {jax.devices()}; run in {run_dir}", flush=True)

    import torch
    if a.resume:
        ts = load_checkpoint(run_dir)
        print(f"resuming at game {int(ts.games_done)}", flush=True)
    else:
        ts = ppo.init(jax.random.PRNGKey(a.seed), cfg)
    update = ppo.make_update(cfg)
    deadline = Deadline(a.stop_at) if a.stop_at else None
    if deadline is not None:
        print(f"stops at {deadline.when:%Y-%m-%d %H:%M}", flush=True)
    stop = {"why": None}

    def on_signal(signum, _frame):
        stop["why"] = signal.Signals(signum).name
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, on_signal)

    mode = "a" if a.resume else "w"
    train_log = open(os.path.join(run_dir, "train.csv"), mode, newline="")
    eval_log = open(os.path.join(run_dir, "eval_heuristic_overall.csv"), mode, newline="")
    writers = {}

    def write(log, key, row):
        if key not in writers:
            writers[key] = csv.DictWriter(log, fieldnames=list(row))
            if not a.resume:
                writers[key].writeheader()
        writers[key].writerow(row)
        log.flush()

    def save_and_eval(games):
        torch.save(ppo.to_torch_state_dict(ts.policy),
                   os.path.join(run_dir, "checkpoints", f"policy_ep{games}.pt"))
        if a.no_eval:
            return
        t0 = time.perf_counter()
        o = evaluate_vs_heuristic(ts.policy, a.players, a.eval_games, a.eval_deal_seed, hidden)
        write(eval_log, "eval", {"episode": games, **{k: o[k] for k in (
            "win_rate", "win_rate_se", "mean_score_diff", "mean_score_diff_se", "exact_bid_rate")}})
        print(f"[{games}] EVAL vs heuristic | win={o['win_rate']:.3f}±{o['win_rate_se']:.3f} "
              f"diff={o['mean_score_diff']:+.2f}±{o['mean_score_diff_se']:.2f} "
              f"exact_bid={o['exact_bid_rate']:.3f} ({time.perf_counter() - t0:.0f} s)", flush=True)

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
        write(train_log, "train", {"episode": games, "seconds": round(time.perf_counter() - start, 1),
                                   **stats})
        if games >= next_eval or games >= a.games:
            rate = (games - start_games) / (time.perf_counter() - start)
            print(f"[{games}] {rate:,.0f} games/s | ent={stats['entropy']:.3f} "
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
            print(f"stopped at game {games} ({stop['why']}). To continue: --resume {run_dir}",
                  flush=True)
            return
    print(f"done: {run_dir}", flush=True)

if __name__ == "__main__":
    main()
