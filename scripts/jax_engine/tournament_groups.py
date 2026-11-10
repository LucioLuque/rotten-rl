"""
scripts/jax_engine/tournament_groups.py  —  A tournament summarized by group of agents.

Reads the .md of scripts/jax_engine/tournament.py and averages its two tables (score difference
and win rate) over the agents of each group: by default an agent's group is its name without the
seed ("s20s0@24000k" and "s20s3@24000k" -> "s20@24000k"; "x20mb1s2@12000256" -> "x20mb1@12000256"),
so the 4 seeds of a run (and each lambda of a run) are one group. Prints, for each pair of groups,
"row vs column": mean difference, mean win rate, and how many of the agent pairs have the row ahead
(diff > 0). The tie in win rate is 1/n.

    python scripts/jax_engine/tournament_groups.py ~/scratch/torneo_x20.md
    python scripts/jax_engine/tournament_groups.py torneo.md --group 's20.*@24' --group 'x20mb1@'
"""

import argparse
import re
import statistics


def read_table(lines, title):
    i = next(k for k, line in enumerate(lines) if line.startswith("## " + title))
    i = next(k for k in range(i, len(lines)) if lines[k].startswith("|"))
    cols = [c.strip() for c in lines[i].split("|")[1:-1]][1:]
    rows = {}
    for line in lines[i + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.split("|")[1:-1]]
        rows[cells[0]] = {cols[j]: (None if v == "—" else float(v.split()[0]))
                          for j, v in enumerate(cells[1:])}
    return rows


def default_group(name):
    return re.sub(r"s\d+(?=@|:|$)", "", name, count=1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("tournament", help=".md of scripts/jax_engine/tournament.py")
    p.add_argument("--group", action="append", default=[],
                   help="regex: the agents it matches are one group (repeatable; the rest keep the "
                        "default grouping)")
    a = p.parse_args()

    lines = open(a.tournament).read().split("\n")
    n = int(re.search(r"(\d+) jugadores", lines[0]).group(1))
    diff, win = read_table(lines, "Diferencia"), read_table(lines, "Win rate")

    def group(name):
        for g in a.group:
            if re.search(g, name):
                return g
        return default_group(name)

    groups = list(dict.fromkeys(group(r) for r in diff))

    def cell(table, ga, gb):
        v = [table[r][c] for r in table for c in table[r]
             if r != c and table[r][c] is not None and group(r) == ga and group(c) == gb]
        return v

    print(f"{n} jugadores, empate en win rate = {100 / n:.1f} %. Fila contra columna: diferencia, "
          f"win rate de la fila, cruces con la fila arriba.\n")
    width = max(len(g) for g in groups) + 2
    print(" " * width + "".join(g.rjust(24) for g in groups))
    for ga in groups:
        out = []
        for gb in groups:
            d, w = cell(diff, ga, gb), cell(win, ga, gb)
            if not d:
                out.append("—".rjust(24))
                continue
            out.append(f"{statistics.mean(d):+.2f} {100 * statistics.mean(w):.1f}% "
                       f"{sum(x > 0 for x in d)}/{len(d)}".rjust(24))
        print(ga.ljust(width) + "".join(out))


if __name__ == "__main__":
    main()
