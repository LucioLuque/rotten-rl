#!/bin/bash
# Rebuilds the environment on a lab machine we were just moved to. /scratch belongs to each machine
# but ~/rotten-rl is in the shared home, so this is always there. Safe to run twice.
#
#   ssh labs004 'bash -l ~/rotten-rl/scripts/lab/setup_machine.sh'
#
# Leaves a tmux session "setup" installing the environment and running the tests; the last line of
# ~/scratch/logs/setup.log is SETUP_DONE. Then upload what is to be resumed from the laptop with
#   rsync -a runs/lab/<machine>/runs/<run> labs004:scratch/runs/
set -e
mkdir -p ~/scratch/runs ~/scratch/logs ~/scratch/work/venvs
rm -f ~/scratch/logs/setup.log
tmux new-session -d -s setup "cd ~/rotten-rl && git pull --ff-only; \
  uv venv --allow-existing --python 3.12 ~/scratch/work/venvs/rotten-rl \
  && source ~/scratch/work/venvs/rotten-rl/bin/activate && uv pip install -e '.[dev]' \
  && python -c 'import torch, pyspiel; print(torch.cuda.is_available())' \
  && pytest -q -p no:anyio 2>&1 | tail -3; echo SETUP_DONE"
tmux pipe-pane -t setup -o 'cat >> ~/scratch/logs/setup.log'
hostname
