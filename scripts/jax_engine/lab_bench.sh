#!/bin/bash
# Measures the JAX engine and PPO on the lab GPU (run it with the GPU free: no training running).
# Writes everything to ~/scratch/logs/jax_bench.log and ends with BENCH_DONE.
#
#   ssh labs004 'bash -lc "tmux new-session -d -s jaxbench ~/rotten-rl/scripts/jax_engine/lab_bench.sh"'
cd ~/rotten-rl || exit 1
source ~/scratch/work/venvs/jax/bin/activate
log=~/scratch/logs/jax_bench.log
: > "$log"
{
  date
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
  echo "== trick phase, 2 players"
  python scripts/jax_engine/bench_tricks.py --players 2 --hand_size 7 --seconds 3 --repeats 10 \
    --batches 1024,16384,65536
  for n in 2 3 7; do
    echo "== whole matches, $n players"
    python scripts/jax_engine/bench_game.py --players $n --seconds 3 --repeats 5 \
      --batches 1024,4096,16384
  done
  for n in 2 3 7; do
    echo "== where the time goes in one PPO update, $n players, 512 matches"
    python scripts/jax_engine/profile_ppo.py --players $n --rollout_games 512 --repeats 3
  done
  for n in 2 7; do
    for b in 512 2048; do
      echo "== PPO training speed, $n players, $b matches per update"
      timeout 600 python scripts/jax_engine/train_ppo.py --players $n --games $((b * 12)) \
        --rollout_games $b --eval_every $((b * 4)) --no_eval --runs_dir ~/scratch/runs_bench \
        --run_name bench${n}p_${b}
    done
  done
  date
  echo BENCH_DONE
} >> "$log" 2>&1
