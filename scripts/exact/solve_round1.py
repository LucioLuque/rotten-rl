"""
scripts/exact/solve_round1.py  —  La ronda de 1 carta resuelta, de 2 a 7 jugadores.

Para cada cantidad de jugadores calcula el equilibrio con CFR+
(rotten_rl/exact/round1.py) y lo compara con la heuristica y con "mirar solo la
carta" (no darle peso a los cantos de los demas). Escribe un reporte en
markdown y guarda las tablas (probabilidad de cantar 1 en cada situacion).

    python scripts/exact/solve_round1.py
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from rotten_rl.exact import round1  # noqa: E402
from rotten_rl.engine import RANKS  # noqa: E402

_CHECKPOINTS = (100, 500, 1000, 2000, 4000)


def bid1_by_card(n, tables, deals):
    """Probabilidad de cantar 1 por posicion y tipo de carta, promediada sobre
    las situaciones en que se da (numero del triunfo y cantos anteriores),
    pesadas por lo seguido que aparecen. (posiciones, 26)."""
    out = np.zeros((n, round1.NUM_TYPES))
    for k, (ix, reach, _, _, s, free) in enumerate(round1._sweep(n, tables, deals)):
        card = (ix // 2 ** k) % round1.NUM_TYPES
        w = reach * free
        num = np.bincount(card.ravel(), (w * s).ravel(), round1.NUM_TYPES)
        den = np.bincount(card.ravel(), w.ravel(), round1.NUM_TYPES)
        out[k] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--players", default="2,3,4,5,6,7")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--deals_per_iteration", type=int, default=20000)
    p.add_argument("--eval_deals", type=int, default=500000)
    p.add_argument("--out", default="docs/primera-ronda/resultados.md")
    p.add_argument("--tables", default="docs/primera-ronda/equilibrio.npz")
    a = p.parse_args()

    saved, sections = {}, []
    for n in [int(x) for x in a.players.split(",")]:
        t0 = time.time()
        eval_deals = round1.sample_deals(n, a.eval_deals, np.random.RandomState(1000 + n))
        conv = []

        def log(it, tables):
            if it in _CHECKPOINTS and it <= a.iterations:
                m = round1.evaluate(n, tables, eval_deals)
                conv.append((it, np.mean(m["gain_best"])))
                print(f"n={n} it={it}: ganancia media por desviarse {conv[-1][1]:.4f} "
                      f"({time.time() - t0:.0f} s)", flush=True)

        eq = round1.solve(n, a.iterations, a.deals_per_iteration,
                          np.random.RandomState(n), log=log)
        for k, t in enumerate(eq):
            saved[f"n{n}_pos{k}"] = t.astype(np.float32)
        heur = round1.heuristic_tables(n)
        m_eq = round1.evaluate(n, eq, eval_deals)
        m_h = round1.evaluate(n, heur, eval_deals)
        by_card = bid1_by_card(n, eq, eval_deals)

        lines = [f"## {n} jugadores", "",
                 f"{round1.num_situations(n):,} situaciones; {a.iterations} iteraciones de "
                 f"{a.deals_per_iteration:,} repartos; evaluado con otros {a.eval_deals:,} "
                 f"({time.time() - t0:.0f} s).", "",
                 "Convergencia (cuánto ganaría en promedio quien se desvíe del equilibrio, "
                 "puntos por ronda): " + ", ".join(f"{it} it: {g:.3f}" for it, g in conv) + ".",
                 "",
                 "| Posición de canto | " + " | ".join(str(k + 1) for k in range(n)) + " |",
                 "|---|" + "---|" * n,
                 "| Valor en el equilibrio | " + " | ".join(
                     f"{v:+.2f}" for v in m_eq["value"]) + " |",
                 "| Ganancia por desviarse del equilibrio | " + " | ".join(
                     f"{v:.3f}" for v in m_eq["gain_best"]) + " |",
                 "| Pérdida por mirar solo la carta (rivales en equilibrio) | " + " | ".join(
                     f"{-v:.3f}" if f else "—"
                     for v, f in zip(m_eq["gain_card_only"], _can_listen(n))) + " |",
                 "| Valor de la heurística (todos heurística) | " + " | ".join(
                     f"{v:+.2f}" for v in m_h["value"]) + " |",
                 "| Ganancia de la mejor respuesta a la heurística | " + " | ".join(
                     f"{v:.2f}" for v in m_h["gain_best"]) + " |",
                 "| ... mirando solo la carta | " + " | ".join(
                     f"{v:.2f}" for v in m_h["gain_card_only"]) + " |",
                 "",
                 "Probabilidad de cantar 1 en el equilibrio, por carta y posición (promedio "
                 "sobre el triunfo dado vuelta y los cantos anteriores, pesado por lo seguido "
                 "que se da; el último solo cuenta cuando puede elegir):", "",
                 "| Carta | " + " | ".join(str(k + 1) for k in range(n)) + " |",
                 "|---|" + "---|" * n]
        for t in range(round1.NUM_TYPES):
            name = ("triunfo " if t >= round1.NUM_RANKS else "") + RANKS[t % round1.NUM_RANKS]
            lines.append(f"| {name} | " + " | ".join(
                "—" if np.isnan(v) else f"{v:.2f}" for v in by_card[:, t]) + " |")
        sections.append("\n".join(lines))
        print(sections[-1] + "\n", flush=True)

    os.makedirs(os.path.dirname(a.tables) or ".", exist_ok=True)
    np.savez_compressed(a.tables, **saved)
    with open(a.out, "w") as f:
        f.write(_HEADER + "\n\n".join(sections) + "\n")


def _can_listen(n):
    """Las posiciones que escuchan cantos y eligen: todas menos la primera (no
    hay cantos antes) y, con 2 jugadores, la ultima (esta forzada)."""
    return [k > 0 and not (n == 2 and k == n - 1) for k in range(n)]


_HEADER = """# La ronda de 1 carta, resuelta

Generado con `scripts/exact/solve_round1.py` (ver `rotten_rl/exact/round1.py`; qué es y por qué, en `README.md` de esta carpeta). Con una carta la jugada es
forzada: la única decisión es cantar 0 o 1, y quién gana la baza depende solo de las cartas.
Cada situación es el número del triunfo dado vuelta, la carta propia (triunfo o no y su
número: los otros tres palos son intercambiables) y lo cantado antes. El reward es el mismo con
el que entrena el agente: puntaje de la ronda menos el promedio de los rivales, en puntos.

Cómo leer las tablas:

- **Posición de canto**: 1 es el primero en cantar (y el que sale); la última tiene la
  restricción de que la suma no puede dar 1.
- **Valor en el equilibrio**: cuánto gana en promedio esa posición contra el resto si todos
  juegan el equilibrio (suma 0 entre todas).
- **Ganancia por desviarse**: cuánto ganaría quien cambie su forma de cantar mientras los demás
  siguen en equilibrio. Cerca de 0 = es un equilibrio. Es una cota algo alta (elegir lo mejor
  sobre una muestra finita suma un poco de ruido a favor).
- **Pérdida por mirar solo la carta**: cuánto pierde quien canta solo según su carta y su
  posición, sin darle peso a lo que cantaron los demás, contra rivales en equilibrio.
- **Heurística**: la de `evaluation.py` (cantar 1 si la carta "vale" media baza o más). La
  mejor respuesta es lo máximo que se le puede sacar a una mesa de heurísticas desde esa
  posición, con toda la información o mirando solo la carta.

"""


if __name__ == "__main__":
    main()
