"""
scripts/search/analyze_decisions.py  —  How good are the agent's decisions, measured with PIMC?

Plays `--matches` whole matches on the device (JAX, all at once) and, at every decision, asks PIMC
(rotten_rl/search/pimc.py) for the value of each legal action. Two modes:

  analyze  The agent plays itself (greedy, every seat). At each decision: the gap between PIMC's
           best action and the agent's action (in round points, PIMC's own estimate) and whether
           they agree. Report by phase (bid / play) and hand size. A large gap means PIMC thinks
           the agent left points on the table there (PIMC is noisy and has its own blind spots:
           read it as a lead, not a verdict).
  play     PIMC in one seat and the agent in the others, against the agent in that seat with the
           same deals and the same rivals (duplicate): does PIMC score more than the network? Every
           seat in turn. Answers whether PIMC is a stronger rival.

Example (lab, GPU):
    python scripts/search/analyze_decisions.py --run ~/scratch/runs/s20_<...>_s0_<ts> --players 2 \\
        --mode analyze --matches 256 --samples 32 --out ~/scratch/pimc_s20.md
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rotten_rl.jax_engine import game as jg  # noqa: E402
from rotten_rl.jax_engine.ppo import ILLEGAL, from_torch_state_dict, match_length, mlp  # noqa: E402
from rotten_rl.search import pimc  # noqa: E402
from rotten_rl.tournament_report import policy_file  # noqa: E402

POINTS = jg.ROUND_SCORE_NORM          # relative round reward * 24 = points over the rivals' mean


def greedy(params, state):
    obs = jg.observation(state, state.current)
    logits = jnp.where(jg.legal_mask(state), mlp(params, obs[None, :])[0], ILLEGAL)
    return jnp.argmax(logits).astype(jnp.int32)


def analyze(params, n, matches, samples, key):
    """Self-play of the agent; PIMC values at every decision. Returns numpy arrays [T, B]."""
    T = match_length(n)

    def one_step(state, k):
        live = ~jg.is_terminal(state)
        a = greedy(params, state)
        actions, values, valid, ok = pimc.action_values(params, state, k, samples)
        slot = jnp.argmax((actions == a) & valid)
        gap = (values.max() - values[slot]) * POINTS
        rec = dict(live=live, gap=gap, agree=actions[jnp.argmax(values)] == a,
                   bidding=state.phase == jg.BIDDING, h=jg.hand_size(state), nlegal=valid.sum(),
                   sampled_ok=ok)
        state, _ = jg.step(state, a)
        return state, rec

    def match(k):
        k_deal, k_run = jax.random.split(k)
        s = jg.random_game(k_deal, n)
        _, rec = jax.lax.scan(one_step, s, jax.random.split(k_run, T))
        return rec

    rec = jax.jit(jax.vmap(match))(jax.random.split(key, matches))
    return {k: np.asarray(v) for k, v in rec.items()}


def play(params, n, matches, samples, key):
    """PIMC in seat `seat`, the agent elsewhere, against the agent everywhere, on the same deals.
    Returns float[n, B]: (PIMC's match points over the rivals' mean) - (the agent's, same seat)."""
    T = match_length(n)

    def run(k_deal, k_run, seat, use_pimc):
        def one_step(state, k):
            a = greedy(params, state)
            a = jnp.where(use_pimc & (state.current == seat),
                          pimc.pimc_action(params, state, k, samples), a)
            state, r = jg.step(state, a)
            return state, r[seat]
        s = jg.random_game(k_deal, n)
        _, r = jax.lax.scan(one_step, s, jax.random.split(k_run, T))
        return r.sum() * POINTS

    def per_deal(k):
        k_deal, k_run = jax.random.split(k)
        seats = jnp.arange(n)
        with_p = jax.vmap(lambda s: run(k_deal, k_run, s, True))(seats)
        base = jax.vmap(lambda s: run(k_deal, k_run, s, False))(seats)
        return with_p - base

    return np.asarray(jax.jit(jax.vmap(per_deal))(jax.random.split(key, matches))).T


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def report_analyze(rec, name, n, matches, samples):
    live = rec["live"] & (rec["nlegal"] > 1)            # decisions with a choice
    gap, agree = rec["gap"][live], rec["agree"][live]
    bid, h = rec["bidding"][live], rec["h"][live]

    def row(label, m):
        g = gap[m]
        return [label, f"{m.sum():,}", f"{g.mean():.3f}", f"{(g > 0.5).mean():.1%}",
                f"{(g > 2).mean():.1%}", f"{agree[m].mean():.1%}"]

    rows = [row("todas", np.ones_like(gap, bool)), row("canto", bid), row("juego", ~bid)]
    rows += [row(f"mano de {k}", h == k) for k in range(1, 8)]
    return "\n".join([
        f"# PIMC sobre las decisiones de {name}", "",
        f"{n} jugadores, {matches} partidas de self-play (acción más probable), PIMC con "
        f"{samples} repartos por decisión (repartos que respetan los palos conocidos: "
        f"{rec['sampled_ok'][live].mean():.1%}). Solo decisiones con más de una opción.", "",
        "Diferencia = valor de la mejor jugada según PIMC menos el de la del agente, en puntos de "
        "la ronda sobre el promedio de los rivales (estimación de PIMC, con su ruido).", "",
        table(rows, ["", "decisiones", "diferencia media", "> 0,5 pts", "> 2 pts",
                     "coincide con PIMC"]), ""])


def report_play(diff, name, n, matches, samples):
    rows = [[f"asiento {k + 1}", f"{diff[k].mean():+.2f} ± {diff[k].std() / np.sqrt(matches):.2f}"]
            for k in range(n)]
    allv = diff.ravel()
    rows.append(["todos", f"{allv.mean():+.2f} ± {allv.std() / np.sqrt(allv.size):.2f}"])
    return "\n".join([
        f"# PIMC jugando contra {name}", "",
        f"{n} jugadores, {matches} repartos por asiento, PIMC con {samples} repartos por decisión "
        "(sus simulaciones juegan con la red). Diferencia por partida de PIMC en un asiento contra "
        "la red en ese asiento, con los mismos repartos y los mismos rivales (la red).", "",
        table(rows, ["", "PIMC menos la red (puntos por partida)"]), ""])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="<run> or <run>@<ep>")
    p.add_argument("--players", type=int, required=True)
    p.add_argument("--mode", choices=["analyze", "play"], default="analyze")
    p.add_argument("--matches", type=int, default=256)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    path, _, name = policy_file(os.path.expanduser(a.run))
    params = from_torch_state_dict(torch.load(path, map_location="cpu"))
    key = jax.random.PRNGKey(a.seed)
    t0 = time.perf_counter()
    if a.mode == "analyze":
        text = report_analyze(analyze(params, a.players, a.matches, a.samples, key),
                              name, a.players, a.matches, a.samples)
    else:
        text = report_play(play(params, a.players, a.matches, a.samples, key),
                           name, a.players, a.matches, a.samples)
    text += f"\n({time.perf_counter() - t0:.0f} s)\n"
    print(text)
    if a.out:
        with open(os.path.expanduser(a.out), "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
