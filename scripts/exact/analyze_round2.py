"""
scripts/exact/analyze_round2.py  —  Preguntas sobre la estrategia calculada de
la ronda de 2 cartas (la que exporta `round2 dump`).

Hoy: cuando el que sale tiene un triunfo y una carta de otro palo, ¿con qué
probabilidad sale con la otra (guardando el triunfo para la segunda baza)?
Separado por el numero de su triunfo y por lo que canto.

    python scripts/exact/analyze_round2.py --tsv estrategia.tsv --n 3
"""

import argparse
from collections import defaultdict

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]


def leader_trump_and_other(path, n):
    """{(numero del triunfo, canto): [peso total, peso de salir con la otra]}."""
    acc = defaultdict(lambda: [0.0, 0.0])
    with open(path) as f:
        for line in f:
            r = line.rstrip("\n").split("\t")
            if r[0] != "play" or r[1] != "0":
                continue
            c0, c1 = int(r[3]), int(r[4])
            if c0 // 13 != 0 or c1 // 13 < 2:          # un triunfo y una de otro palo
                continue
            bid = int(r[5]) >> 2 * (n - 1) & 3
            p_other, w = float(r[9]), float(r[11])
            a = acc[(c0 % 13, bid)]
            a[0] += w
            a[1] += w * p_other
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", required=True)
    p.add_argument("--n", type=int, required=True)
    a = p.parse_args()
    acc = leader_trump_and_other(a.tsv, a.n)
    print("Sale primero con un triunfo y otra carta: probabilidad de salir con la otra")
    print("| Su triunfo | canto 0 | canto 1 | canto 2 |")
    print("|---|---|---|---|")
    for r in range(13):
        cells = []
        for bid in range(3):
            w, x = acc.get((r, bid), (0.0, 0.0))
            cells.append(f"{x / w:.2f}" if w > 0 else "—")
        print(f"| {RANKS[r]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
