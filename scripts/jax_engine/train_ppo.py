"""
scripts/jax_engine/train_ppo.py  —  Train PPO entirely in JAX (stage 3 prototype).

Same recipe as `scripts/train.py --algo ppo --round_episodes --ppo_anneal_lr` (see
rotten_rl/jax_engine/ppo.py). Every `--eval_every` games the policy is saved as
<run>/checkpoints/policy_ep<N>.pt (a PyTorch state_dict) and evaluated against the heuristic with
the usual evaluation (rotten_rl/evaluation.py: duplicate deals, same --eval_deal_seed as train.py),
so the run can be compared in tournaments with the PyTorch ones. Not yet: --resume, --stop_at.

    python scripts/jax_engine/train_ppo.py --players 2 --games 1000000 --rollout_games 512 \
        --run_name j23 --runs_dir ~/scratch/runs
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from rotten_rl.jax_engine import ppo  # noqa: E402


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
    p.add_argument("--no_eval", action="store_true", help="only train and save (to measure speed)")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--run_name", default="jax_ppo")
    a = p.parse_args()

    hidden = tuple(int(x) for x in a.hidden.split(","))
    cfg = ppo.Config(num_players=a.players, hidden=hidden, learning_rate=a.learning_rate,
                     rollout_games=a.rollout_games, epochs=a.epochs, minibatch=a.minibatch,
                     total_games=a.games, reward_scheme=a.reward_scheme)
    run_dir = os.path.join(os.path.expanduser(a.runs_dir), f"{a.run_name}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(os.path.join(run_dir, "checkpoints"))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"algo": "ppo", "engine": "jax", "players": a.players,
                   "hidden_layers_sizes": [str(h) for h in hidden], "num_train_episodes": a.games,
                   "ppo_rollout_episodes": a.rollout_games, "ppo_epochs": a.epochs,
                   "ppo_minibatch": a.minibatch, "rl_learning_rate": a.learning_rate,
                   "reward_scheme": a.reward_scheme, "round_episodes": True, "ppo_anneal_lr": True,
                   "seed": a.seed}, f, indent=1)
    print(f"JAX {jax.__version__} on {jax.devices()}; run in {run_dir}", flush=True)

    import torch
    ts = ppo.init(jax.random.PRNGKey(a.seed), cfg)
    update = ppo.make_update(cfg)
    train_log = open(os.path.join(run_dir, "train.csv"), "w", newline="")
    eval_log = open(os.path.join(run_dir, "eval_heuristic_overall.csv"), "w", newline="")
    tw = ew = None

    def save_and_eval(games):
        torch.save(ppo.to_torch_state_dict(ts.policy),
                   os.path.join(run_dir, "checkpoints", f"policy_ep{games}.pt"))
        if a.no_eval:
            return
        t0 = time.perf_counter()
        o = evaluate_vs_heuristic(ts.policy, a.players, a.eval_games, a.eval_deal_seed, hidden)
        nonlocal ew
        row = {"episode": games, **{k: o[k] for k in ("win_rate", "win_rate_se", "mean_score_diff",
                                                      "mean_score_diff_se", "exact_bid_rate")}}
        if ew is None:
            ew = csv.DictWriter(eval_log, fieldnames=list(row))
            ew.writeheader()
        ew.writerow(row)
        eval_log.flush()
        print(f"[{games}] EVAL vs heuristic | win={o['win_rate']:.3f}±{o['win_rate_se']:.3f} "
              f"diff={o['mean_score_diff']:+.2f}±{o['mean_score_diff_se']:.2f} "
              f"exact_bid={o['exact_bid_rate']:.3f} ({time.perf_counter() - t0:.0f} s)", flush=True)

    save_and_eval(0)
    next_eval = a.eval_every
    start = time.perf_counter()
    games = 0
    while games < a.games:
        ts, stats = update(ts)
        games = int(ts.games_done)
        stats = {k: float(v) for k, v in stats.items()}
        row = {"episode": games, "seconds": round(time.perf_counter() - start, 1), **stats}
        if tw is None:
            tw = csv.DictWriter(train_log, fieldnames=list(row))
            tw.writeheader()
        tw.writerow(row)
        train_log.flush()
        if games >= next_eval or games >= a.games:
            rate = games / (time.perf_counter() - start)
            print(f"[{games}] {rate:,.0f} games/s | ent={stats['entropy']:.3f} "
                  f"kl={stats['approx_kl']:.4f} ev={stats['explained_variance']:.3f} "
                  f"lr={stats['learning_rate']:.2e}", flush=True)
            save_and_eval(games)
            next_eval += a.eval_every


if __name__ == "__main__":
    main()
