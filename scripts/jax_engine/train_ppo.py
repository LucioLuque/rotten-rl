"""
scripts/jax_engine/train_ppo.py  —  Train PPO entirely in JAX on the GPU, one or several seeds.

Same recipe as `scripts/train.py --algo ppo --round_episodes --ppo_anneal_lr` (see
rotten_rl/jax_engine/ppo.py), ~30 times faster per run. With --seeds 0,1,2,3 the seeds train
together in one process (vmap: one compiled step trains all of them, each with its own networks,
optimizer and matches); one seed leaves the GPU half idle, so 2 to 4 seeds give ~1.8 times the
matches per second of one (scripts/jax_engine/bench_seeds.py). Every seed is its own run folder
(<run_name>_s<seed>_<time>), with the logs, CSVs and checkpoints of any other run.

--learning_rate and --ent_coef also take lists (e.g. --learning_rate 1e-4,3e-4,1e-3): every
combination of seed and values trains as one more member of the same vmapped step, so a sweep costs
about one run. Members with the same seed start from the same networks. Each is its own run
folder, tagged with what varies (<run_name>_lr1e-04_s0_<time>); the tournaments name it the same
way (p20lr1e-04s0). Once the sweep picks a value, train it alone with several seeds.

Every --eval_every games each policy is saved as <run>/checkpoints/policy_ep<N>.pt (a PyTorch
state_dict) and evaluated against random, the heuristic and its previous snapshot ("past", greedy)
with duplicate deals (--eval_deal_seed), written with RunLogger (eval_<opponent>_overall/by_h/
by_seat CSVs, train.csv, TensorBoard). The evaluation runs on the GPU too
(rotten_rl/jax_engine/evaluate.py, under a second); --eval_engine pyspiel uses
rotten_rl/evaluation.py on the CPU instead (the deals differ: compare within standard errors).

The whole training state of each seed goes to <run>/state/ckpt.pkl every --checkpoint_every games
and when it stops (--stop_at, SIGTERM/SIGHUP, Ctrl+C); --resume <run>[,<run>...] continues them
(with --games, extending the horizon). Accepts --runs_dir, --run_name and --stop_at like train.py,
so run_queue.py can launch it.

On the lab it runs from its own venv (~/scratch/work/venvs/jax). Next to other GPU processes, set
XLA_PYTHON_CLIENT_PREALLOCATE=false (run_queue.py does it).

    python scripts/jax_engine/train_ppo.py --players 2 --games 3000000 --seeds 0,1,2,3 \
        --run_name g2_3M --runs_dir ~/scratch/runs --stop_at lab
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

CONFIG_KEYS = ("players", "games", "rollout_games", "hidden", "learning_rate", "ent_coef", "epochs",
               "minibatch", "reward_scheme", "dtype", "seed", "episode", "match_bonus",
               "rank_weight")
# may differ between the members of a process
MEMBER_KEYS = ("seed", "learning_rate", "ent_coef", "match_bonus")
CONFIG_DEFAULTS = {"dtype": "float32", "ent_coef": 0.01,  # for runs saved before the key existed
                   "episode": "round", "match_bonus": 0.0, "rank_weight": 0.25}
EVAL_OPPONENTS = ("random", "heuristic", "past")


def save_checkpoint(run_dir, ts):
    """The whole training state of one seed (networks, Adam, games played, RNG key), atomically."""
    state_dir = os.path.join(run_dir, "state")
    os.makedirs(state_dir, exist_ok=True)
    tmp = os.path.join(state_dir, "ckpt.pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, ts), f)
        f.flush()
        os.fsync(f.fileno())                 # a machine reboot must not leave it empty
    os.replace(tmp, os.path.join(state_dir, "ckpt.pkl"))


def load_checkpoint(run_dir):
    with open(os.path.join(run_dir, "state", "ckpt.pkl"), "rb") as f:
        return jax.tree_util.tree_map(jax.numpy.asarray, pickle.load(f))


def stack(states):
    return jax.tree_util.tree_map(lambda *x: jax.numpy.stack(x), *states)


class PyspielEval:
    """--eval_engine pyspiel: rotten_rl/evaluation.py on the CPU, as before."""

    def __init__(self, a, hidden):
        import pyspiel
        import torch
        import rotten_rl.openspiel_game  # noqa: F401  registers "podrida"
        from rotten_rl.algorithms.dqn import greedy_batch, mlp
        from rotten_rl.evaluation import evaluate, heuristic_opponent, random_opponent
        self.a, self.hidden, self.evaluate = a, hidden, evaluate
        self.game = pyspiel.load_game("podrida", {"players": a.players,
                                                  "reward_scheme": a.reward_scheme})
        self.rng = np.random.RandomState(a.seed)
        self.fixed = {"random": random_opponent(self.rng), "heuristic": heuristic_opponent(self.rng)}
        self.torch, self.greedy_batch, self.mlp = torch, greedy_batch, mlp

    def _agent(self, params):
        net = self.mlp(808, list(self.hidden), 60)
        net.load_state_dict(ppo.to_torch_state_dict(params))
        net.eval()

        def act(state, legal):
            return int(act.batch([state], [legal])[0])
        act.batch = lambda states, legals: self.greedy_batch(net, states, legals, 60,
                                                             self.torch.device("cpu"))
        return act

    def __call__(self, params, opponent, past):
        opp = self._agent(past) if opponent == "past" else self.fixed[opponent]
        a = self.a
        return self.evaluate(self.game, self._agent(params), opp, a.players, a.eval_games,
                             self.rng, parallel=True,
                             deal_seed=None if a.eval_deal_seed < 0 else a.eval_deal_seed)


class JaxEval:
    def __init__(self, a):
        from rotten_rl.jax_engine import evaluate
        self.a, self.evaluate = a, evaluate.evaluate
        self.key = jax.random.PRNGKey(a.seed + 10_000)

    def __call__(self, params, opponent, past):
        self.key, sub = jax.random.split(self.key)
        a = self.a
        return self.evaluate(params, opponent, a.players, a.eval_games,
                             deal_seed=max(a.eval_deal_seed, 0), opponent_params=past, key=sub,
                             reward_scheme=a.reward_scheme)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--games", type=int, default=1_000_000, help="total matches (lr reaches 0 here)")
    p.add_argument("--rollout_games", type=int, default=512)
    p.add_argument("--hidden", default="512,512")
    p.add_argument("--learning_rate", default="3e-4",
                   help="one value, or several (comma separated) trained together")
    p.add_argument("--ent_coef", default="0.01",
                   help="entropy bonus; one value, or several trained together")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--reward_scheme", default="relative")
    p.add_argument("--episode", default="round", choices=["round", "match"],
                   help="match: the whole match is one episode (the score matters)")
    p.add_argument("--match_bonus", default="0",
                   help="lambda of the end-of-match bonus (winning + some for the place); one value "
                        "or several trained together; only with --episode match")
    p.add_argument("--rank_weight", type=float, default=0.25,
                   help="share of the place in the end-of-match bonus (rotten_rl/jax_engine/game.py)")
    p.add_argument("--init_from", default=None,
                   help="run folder(s), comma-separated: new runs that start from their networks "
                        "(policy and value; fresh Adam, games and lr schedule), one per folder and "
                        "value of the lists, with that folder's seed")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                   help="type of the matmuls (weights and Adam stay float32)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default=None,
                   help="several seeds trained together, e.g. 0,1,2,3 (one run folder each)")
    p.add_argument("--eval_every", type=int, default=25000)
    p.add_argument("--eval_games", type=int, default=1500)
    p.add_argument("--eval_deal_seed", type=int, default=1234)
    p.add_argument("--eval_engine", default="jax", choices=["jax", "pyspiel"])
    p.add_argument("--checkpoint_every", type=int, default=25000)
    p.add_argument("--no_eval", action="store_true", help="only train and save (to measure speed)")
    p.add_argument("--stop_at", default=None,
                   help="HH:MM, 'YYYY-MM-DD HH:MM' or lab (rotten_rl/lab_hours.py): save and stop")
    p.add_argument("--resume", default=None,
                   help="run folder(s) to continue, comma-separated (their config is used)")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--run_name", default="jax_ppo")
    a = p.parse_args()

    if a.resume:
        run_dirs = [os.path.expanduser(r).rstrip("/") for r in a.resume.split(",")]
        saved = []
        for r in run_dirs:
            with open(os.path.join(r, "config.json")) as f:
                saved.append(json.load(f)["jax_args"])
        for k in CONFIG_KEYS:
            if k in MEMBER_KEYS or (k == "games" and "--games" in sys.argv):
                continue                     # extending the horizon is allowed (the lr follows it)
            values = [s.get(k, CONFIG_DEFAULTS.get(k)) for s in saved]
            if len({str(v) for v in values}) > 1:
                sys.exit(f"--resume: the runs differ in {k}; resume them separately")
            setattr(a, k, values[0])
        members = [(int(s["seed"]), float(s["learning_rate"]),
                    float(s.get("ent_coef", CONFIG_DEFAULTS["ent_coef"])),
                    float(s.get("match_bonus", CONFIG_DEFAULTS["match_bonus"]))) for s in saved]
        lrs = sorted({m[1] for m in members})
        ents = sorted({m[2] for m in members})
        bonuses = sorted({m[3] for m in members})
    else:
        lrs = [float(x) for x in str(a.learning_rate).split(",")]
        ents = [float(x) for x in str(a.ent_coef).split(",")]
        bonuses = [float(x) for x in str(a.match_bonus).split(",")]
        if a.episode == "round" and any(b != 0 for b in bonuses):
            sys.exit("--match_bonus needs --episode match")
        if a.init_from:
            init_dirs = [os.path.expanduser(r).rstrip("/") for r in a.init_from.split(",")]
            seeds = []
            for r in init_dirs:
                with open(os.path.join(r, "config.json")) as f:
                    src = json.load(f)["jax_args"]
                for k in ("players", "hidden"):
                    if str(src[k]) != str(getattr(a, k)):
                        sys.exit(f"--init_from {r}: {k} is {src[k]}, here {getattr(a, k)}")
                seeds.append(int(src["seed"]))
            if len(set(seeds)) != len(seeds):
                sys.exit("--init_from: two runs with the same seed")
            init_of = dict(zip(seeds, init_dirs))
        else:
            seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [a.seed]
        members = [(s, lr, ent, b) for lr in lrs for ent in ents for b in bonuses for s in seeds]
    seeds = [m[0] for m in members]
    a.seed = seeds[0]
    multi = len(members) > 1

    def tag(member):                         # what tells this member apart, besides its seed
        _, lr, ent, bonus = member
        return "".join([f"_lr{lr:.0e}" if len(lrs) > 1 else "", f"_ent{ent:g}" if len(ents) > 1 else "",
                        f"_mb{bonus:g}" if len(bonuses) > 1 else ""])
    names = [f"s{m[0]}{tag(m)}" for m in members]
    if len(set(names)) != len(names):
        sys.exit(f"repeated members (values that round to the same tag?): {names}")
    hidden = tuple(int(x) for x in str(a.hidden).split(","))
    cfg = ppo.Config(num_players=a.players, hidden=hidden, learning_rate=members[0][1],
                     ent_coef=members[0][2], rollout_games=a.rollout_games, epochs=a.epochs,
                     minibatch=a.minibatch, total_games=a.games, reward_scheme=a.reward_scheme,
                     dtype=a.dtype, episode=a.episode, match_bonus=members[0][3],
                     rank_weight=a.rank_weight)
    hp = {"learning_rate": np.array([m[1] for m in members], np.float32),
          "ent_coef": np.array([m[2] for m in members], np.float32),
          "match_bonus": np.array([m[3] for m in members], np.float32)} if multi else None

    import torch
    from rotten_rl.run_logger import RunLogger

    def config_of(member):
        seed, lr, ent, bonus = member
        args = {k: getattr(a, k) for k in CONFIG_KEYS}
        args.update(seed=seed, learning_rate=lr, ent_coef=ent, match_bonus=bonus)
        return {"algo": "ppo", "engine": "jax", "players": a.players,
                "hidden_layers_sizes": [str(h) for h in hidden], "num_train_episodes": a.games,
                "ppo_rollout_episodes": a.rollout_games, "ppo_epochs": a.epochs,
                "ppo_minibatch": a.minibatch, "rl_learning_rate": lr, "ppo_entropy_coef": ent,
                "reward_scheme": a.reward_scheme, "round_episodes": a.episode == "round",
                "episode": a.episode, "match_bonus": bonus, "rank_weight": a.rank_weight,
                "init_from": init_of.get(seed) if a.init_from else None, "ppo_anneal_lr": True,
                "eval_games": a.eval_games, "eval_deal_seed": a.eval_deal_seed,
                "eval_engine": a.eval_engine, "dtype": a.dtype, "seed": seed, "seeds_together": seeds,
                "member_tag": tag(member).replace("_", ""), "jax_args": args}

    if a.resume:
        states = [load_checkpoint(r) for r in run_dirs]
        if len({int(s.games_done) for s in states}) > 1:
            sys.exit("--resume: the runs stopped at different games; resume them separately")
        logs = [RunLogger(run_dir=r, config=config_of(m), resume_after=int(ts.games_done))
                for r, m, ts in zip(run_dirs, members, states)]
    else:
        states = [ppo.init(jax.random.PRNGKey(m[0]), cfg) for m in members]
        if a.init_from:                      # the networks of those runs; the rest starts fresh
            loaded = {s: load_checkpoint(r) for s, r in init_of.items()}
            states = [ts._replace(policy=loaded[m[0]].policy, value=loaded[m[0]].value,
                                  opt_state=ppo.make_optimizer(cfg).init(
                                      (loaded[m[0]].policy, loaded[m[0]].value)))
                      for ts, m in zip(states, members)]
            print("starting from " + ", ".join(f"s{s}: {r} (game {int(loaded[s].games_done)})"
                                               for s, r in init_of.items()), flush=True)
            del loaded
        logs = [RunLogger(base_dir=os.path.expanduser(a.runs_dir),
                          run_name=f"{a.run_name}{tag(m)}_s{m[0]}" if multi else a.run_name,
                          config=config_of(m)) for m in members]
        run_dirs = [log.run_dir for log in logs]
    snap_dirs = [os.path.join(r, "checkpoints") for r in run_dirs]
    for d in snap_dirs:
        os.makedirs(d, exist_ok=True)
    ts = stack(states) if multi else states[0]
    games = start_games = int(states[0].games_done)
    del states
    print(f"JAX {jax.__version__} on {jax.devices()}; members {' '.join(names)}; runs in {', '.join(run_dirs)}"
          + (f"; resuming at game {games}" if a.resume else ""), flush=True)

    def seed_state(i):
        return ppo.unstack(ts, i) if multi else ts

    evaluate = None if a.no_eval else (JaxEval(a) if a.eval_engine == "jax" else PyspielEval(a, hidden))
    past = [None] * len(seeds)
    if a.resume:                             # the snapshot of the last evaluation is "past"
        for i, d in enumerate(snap_dirs):
            # An empty file is a save that a machine reboot cut: skip it.
            snaps = [int(f[len("policy_ep"):-3]) for f in os.listdir(d)
                     if f.startswith("policy_ep") and os.path.getsize(os.path.join(d, f)) > 0]
            snaps = [e for e in snaps if e <= games]
            if snaps:
                past[i] = ppo.from_torch_state_dict(
                    torch.load(os.path.join(d, f"policy_ep{max(snaps)}.pt")))

    def save_and_eval(games):
        t0 = time.perf_counter()
        for i, (member, log, d) in enumerate(zip(names, logs, snap_dirs)):
            policy = seed_state(i).policy
            with open(os.path.join(d, f"policy_ep{games}.pt"), "wb") as f:
                torch.save(ppo.to_torch_state_dict(policy), f)
                f.flush()
                os.fsync(f.fileno())
            if evaluate is not None:
                for name in EVAL_OPPONENTS:
                    if name == "past" and past[i] is None:
                        continue
                    m = evaluate(policy, name, past[i])
                    log.log_eval(games, m, opponent=name)
                    o = m["overall"]
                    print(f"[{games}] {member} EVAL vs {name:9s} | win={o['win_rate']:.3f}±"
                          f"{o['win_rate_se']:.3f} (empate={1 / a.players:.3f}) "
                          f"diff={o['mean_score_diff']:+.2f}±{o['mean_score_diff_se']:.2f} "
                          f"exact_bid={o['exact_bid_rate']:.3f}", flush=True)
            past[i] = policy
        print(f"[{games}] eval took {time.perf_counter() - t0:.1f} s", flush=True)

    deadline = Deadline(a.stop_at) if a.stop_at else None
    if deadline is not None:
        print(f"stops at {deadline.when:%Y-%m-%d %H:%M}", flush=True)
    stop = {"why": None}

    def on_signal(signum, _frame):
        stop["why"] = signal.Signals(signum).name
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, on_signal)

    def checkpoint():
        for i, r in enumerate(run_dirs):
            save_checkpoint(r, seed_state(i))

    update = ppo.make_update(cfg, multi_seed=multi)
    if games == 0:
        save_and_eval(0)
    next_eval = (games // a.eval_every + 1) * a.eval_every
    next_ckpt = (games // a.checkpoint_every + 1) * a.checkpoint_every
    start = time.perf_counter()
    while games < a.games:
        ts, stats = update(ts, hp)
        stats = jax.device_get(stats)
        games = int(np.asarray(ts.games_done).ravel()[0])
        speed = (games - start_games) / (time.perf_counter() - start)
        for i, log in enumerate(logs):
            s = {k: float(np.asarray(v).ravel()[i]) for k, v in stats.items()}
            reward = s.pop("reward_seat0")
            s["games_per_second"] = speed
            log.log_train(games, reward, s)
        if games >= next_eval or games >= a.games:
            ent = np.asarray(stats["entropy"]).ravel()
            lr_now = np.asarray(stats["learning_rate"]).ravel()
            print(f"[{games}] {speed:,.0f} games/s per member ({speed * len(members):,.0f} in total) | "
                  f"ent={' '.join(f'{e:.3f}' for e in ent)} "
                  f"lr={' '.join(f'{x:.2e}' for x in sorted(set(lr_now.tolist())))}", flush=True)
            save_and_eval(games)
            next_eval += a.eval_every
        if deadline is not None and deadline.reached():
            stop["why"] = f"--stop_at {a.stop_at}"
        if games >= next_ckpt or games >= a.games or stop["why"]:
            checkpoint()
            next_ckpt += a.checkpoint_every
        if stop["why"]:
            for log in logs:
                log.close()
            print(f"Cortado en el episodio {games} ({stop['why']}). Para seguir: --resume "
                  f"{','.join(run_dirs)}", flush=True)
            return
    for log in logs:
        log.close()
    print(f"Listo. Resultados en {', '.join(run_dirs)}", flush=True)


if __name__ == "__main__":
    main()
