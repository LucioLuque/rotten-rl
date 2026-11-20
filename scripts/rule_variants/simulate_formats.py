"""
scripts/rule_variants/simulate_formats.py  —  How often each match format comes out of the draw.

Draws --matches formats with rotten_rl/jax_engine/formats.py and prints the --top most frequent
(ladder and rounds without trump; the no-repeat-bid rule is drawn apart and summarized below),
with their share and the cumulative share, how many distinct formats there are, and the
distributions of the number of rounds and of rounds without trump.

    python scripts/rule_variants/simulate_formats.py --matches 1000000 --top 25
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from rotten_rl.jax_engine import formats  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matches", type=int, default=1_000_000)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    keys = jax.random.split(jax.random.PRNGKey(a.seed), a.matches)
    h, nt, rep = jax.device_get(jax.jit(jax.vmap(formats.sample_format))(keys))
    h, nt, rep = np.asarray(h), np.asarray(nt), np.asarray(rep)
    codes = collections.Counter(map(bytes, np.concatenate([h.astype(np.int8), nt.astype(np.int8)], 1)))
    M = a.matches
    print(f"{M:,} partidas sorteadas, {len(codes):,} formatos distintos "
          f"(sin contar la regla de no repetir)\n")
    print("| # | Formato (* = sin triunfo) | Rondas | Sin triunfo | % | % acumulado |")
    print("|---|---|---|---|---|---|")
    acc = 0.0
    for i, (code, c) in enumerate(codes.most_common(a.top), 1):
        x = np.frombuffer(code, np.int8)
        hh, tt = x[:formats.MAX_ROUNDS], x[formats.MAX_ROUNDS:].astype(bool)
        pct = 100 * c / M
        acc += pct
        print(f"| {i} | {formats.format_string(hh, tt)} | {(hh > 0).sum()} | {tt.sum()} | "
              f"{pct:.2f} | {acc:.1f} |")
    rounds = (h > 0).sum(1)
    k = nt.sum(1)
    print("\nRondas por partida: " + ", ".join(
        f"{r}: {100 * (rounds == r).mean():.1f} %" for r in np.unique(rounds)))
    print("Rondas sin triunfo por partida: " + ", ".join(
        f"{r}: {100 * (k == r).mean():.1f} %" for r in np.unique(k)))
    print(f"Promedio de rondas sin triunfo: {k.mean():.2f}; partidas con alguna: {100 * (k > 0).mean():.1f} %")
    print(f"Regla de no repetir el canto: {100 * rep.mean():.1f} % de las partidas")
    by_h = {int(x): 100 * nt[h == x].mean() for x in range(1, 8)}
    print("Sin triunfo según las cartas de la ronda: " + ", ".join(f"{x}: {v:.1f} %" for x, v in by_h.items()))


if __name__ == "__main__":
    main()
