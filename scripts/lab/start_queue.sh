#!/bin/bash
# Starts a queue on the lab machine this runs on, in tmux, with its log in ~/scratch/logs/<queue>.log.
#   bash ~/rotten-rl/scripts/lab/start_queue.sh queue22      (reads ~/scratch/queue22.txt)
q=$1
cd ~/rotten-rl
tmux new-session -d -s "$q" "~/scratch/work/venvs/jax/bin/python scripts/run_queue.py --queue ~/scratch/$q.txt --parallel 1 --script scripts/jax_engine/train_ppo.py --python ~/scratch/work/venvs/jax/bin/python --runs_dir ~/scratch/runs --stop_at lab --no_launch_after lab 2>&1 | tee -a ~/scratch/logs/$q.log"
