"""
scripts/agents/export_agents.py  —  Copies the agent of each number of players to data/agents/.

Since 26/11 (docs/single_model/) there is one network per number of players: the specialists with
match formats. This copies seed 0 of each, at its last policy, to data/agents/<n>p/ (config.json and
checkpoints/policy_ep<N>.pt), which is what scripts/play.py loads by default for that number of
players. Run it where the runs are (the server, runs/lab/...) and commit data/agents/.

    python scripts/agents/export_agents.py --runs runs/lab/labo-s004-14/runs
"""

import argparse
import glob
import os
import re
import shutil

AGENTS = {2: "m20_jaxppo2p_formats_ent0.06_24M", 3: "m30_jaxppo3p_formats_ent0.06_24M",
          4: "m40_jaxppo4p_formats_ent0.06_24M", 5: "m50_jaxppo5p_formats_ent0.08_24M",
          6: "m60_jaxppo6p_formats_ent0.1_24M", 7: "m72_jaxppo7p_formats_ent0.1_24M_bf16"}
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "agents")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/lab/labo-s004-14/runs")
    a = p.parse_args()
    for n, name in AGENTS.items():
        runs = sorted(glob.glob(os.path.join(a.runs, f"{name}_s0_*")))
        if not runs:
            print(f"{n}p: no esta {name}")
            continue
        run = runs[0]
        policies = glob.glob(os.path.join(run, "checkpoints", "policy_ep*.pt"))
        last = max(policies, key=lambda f: int(re.search(r"policy_ep(\d+)", f).group(1)))
        dest = os.path.join(OUT, f"{n}p")
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(os.path.join(dest, "checkpoints"))
        shutil.copy(os.path.join(run, "config.json"), dest)
        shutil.copy(last, os.path.join(dest, "checkpoints"))
        with open(os.path.join(dest, "SOURCE"), "w") as f:
            f.write(f"{os.path.basename(run)}/{os.path.basename(last)}\n")
        print(f"{n}p: {os.path.basename(run)} {os.path.basename(last)}")


if __name__ == "__main__":
    main()
