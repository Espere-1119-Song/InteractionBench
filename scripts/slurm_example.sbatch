#!/bin/bash
# Example Slurm job: one system, one protocol, split over an array of shards.
# Adjust the partition, the environment and the paths to your cluster.
#
#   sbatch --array=0-3 scripts/slurm_example.sbatch qwen3vl-8b sliding
#   ibench merge results/runs/qwen3vl-8b_sliding/preds.jsonl results/runs/qwen3vl-8b_sliding_s*/preds.jsonl
#
#SBATCH -J ibench
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%A_%a.log
MODEL=${1:?model}; PROTOCOL=${2:?protocol}; shift 2
NSHARDS=${SLURM_ARRAY_TASK_COUNT:-1}; SHARD=${SLURM_ARRAY_TASK_ID:-0}
OUT=results/runs/${MODEL}_${PROTOCOL}_s${SHARD}
mkdir -p logs
bash scripts/run_with_watchdog.sh 5400 logs/runner_${MODEL}_${PROTOCOL}_s${SHARD}.log -- \
  python -m interactionbench run --model "$MODEL" --protocol "$PROTOCOL" --mcq \
  --num-shards "$NSHARDS" --shard-index "$SHARD" --out "$OUT" "$@"
