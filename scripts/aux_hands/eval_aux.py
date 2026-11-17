"""
scripts/aux_hands/eval_aux.py  —  How well does the auxiliary head predict the other hands?

Loads the training state of runs trained with --aux_hands (state/ckpt.pkl: policy + aux head), plays
`--matches` self-play matches with the policy (greedy) and, at every decision, compares the head's
predictions about the cards the player cannot see with the truth:

  - loss: binary cross-entropy (as in training), against the "blind" guess that only knows how many
    cards each player holds (probability count / unseen cards for every unseen card);
  - hits: for each other player holding c cards, the share of the c cards the head finds most likely
    that really are in that hand, against c / unseen (a random pick).

Reported overall, for bids and plays, and by hand size.

    python scripts/aux_hands/eval_aux.py --runs ~/scratch/runs/a30_* --players 3 --matches 128
"""

import argparse
import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from rotten_rl.jax_engine import game as jg  # noqa: E402
from rotten_rl.jax_engine import ppo  # noqa: E402


def measure(ts, n, matches, key):
    T = ppo.match_length(n)

    def one_step(states, _):
        player = states.current
        obs = jax.vmap(jg.observation)(states, player)
        logits = jnp.where(jax.vmap(jg.legal_mask)(states), ppo.mlp(ts.policy, obs), ppo.ILLEGAL)
        feats = ppo.mlp_features(ts.policy, obs)
        w, b = ts.aux[0]
        pred = jax.nn.sigmoid((feats @ w + b).reshape(-1, ppo.AUX_SEATS, 52))
        t = ppo._aux_targets(states, player)
        live = ~jax.vmap(jg.is_terminal)(states)
        rec = dict(pred=pred, hands=t["aux_hands"], unseen=t["aux_unseen"], live=live,
                   bidding=states.phase == jg.BIDDING, h=jax.vmap(jg.hand_size)(states))
        states, _ = jax.vmap(jg.step)(states, jnp.argmax(logits, axis=1).astype(jnp.int32))
        return states, rec

    states = jax.vmap(lambda k: jg.random_game(k, n))(jax.random.split(key, matches))
    _, rec = jax.lax.scan(one_step, states, None, length=T)
    return {k: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:]) for k, v in rec.items()}


def summarize(rec, n, sel):
    pred, hands, unseen = rec["pred"][sel], rec["hands"][sel], rec["unseen"][sel]
    seats = np.arange(1, ppo.AUX_SEATS + 1) < n
    m = unseen[:, None, :] & seats[None, :, None]                 # [D, 6, 52]
    counts = (hands & m).sum(axis=2)                              # cards each other player holds
    nun = unseen.sum(axis=1)                                      # unseen cards
    eps = 1e-6
    bce = -(hands * np.log(pred + eps) + (~hands) * np.log(1 - pred + eps))
    prior = np.clip(counts / np.maximum(nun, 1)[:, None], eps, 1 - eps)[:, :, None]
    bce0 = -(hands * np.log(prior) + (~hands) * np.log(1 - prior))
    loss, loss0 = (bce * m).sum() / m.sum(), (bce0 * m).sum() / m.sum()
    # hits: top-c unseen cards per seat
    score = np.where(m, pred, -1.0)
    order = np.argsort(-score, axis=2)
    rank = np.argsort(order, axis=2)
    top = (rank < counts[:, :, None]) & m
    valid = counts > 0
    hits = (top & hands).sum() / max(counts[valid].sum(), 1)
    rand = (counts[valid] ** 2 / np.maximum(nun, 1)[:, None].repeat(ppo.AUX_SEATS, 1)[valid]).sum() \
        / max(counts[valid].sum(), 1)
    return loss, loss0, hits, rand, int(sel.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="run folders (comma-separated, * allowed)")
    p.add_argument("--players", type=int, required=True)
    p.add_argument("--matches", type=int, default=128)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    runs = sorted(sum((glob.glob(os.path.expanduser(x)) for x in a.runs.split(",")), []))
    rows = []
    for r in runs:
        with open(os.path.join(r, "state", "ckpt.pkl"), "rb") as f:
            ts = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
        if ts.aux is None:
            sys.exit(f"{r}: trained without --aux_hands")
        rec = measure(ts, a.players, a.matches, jax.random.PRNGKey(a.seed))
        live = rec["live"]
        for label, sel in [("todas", live), ("canto", live & rec["bidding"]),
                           ("juego", live & ~rec["bidding"])] + \
                [(f"mano de {k}", live & (rec["h"] == k)) for k in range(1, 8)]:
            if sel.sum() == 0:
                continue
            loss, loss0, hits, rand, d = summarize(rec, a.players, sel)
            rows.append((os.path.basename(r)[:40], label, d, loss, loss0, hits, rand))
    print("| corrida | qué | decisiones | error | error a ciegas | aciertos | al azar |")
    print("|---|---|---|---|---|---|---|")
    for run, label, d, loss, loss0, hits, rand in rows:
        print(f"| {run} | {label} | {d:,} | {loss:.3f} | {loss0:.3f} | {hits:.1%} | {rand:.1%} |")


if __name__ == "__main__":
    main()
