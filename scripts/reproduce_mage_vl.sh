#!/usr/bin/env bash
# Reproduce the Mage-VL default sliding-window InteractionBench evaluation.
#
# Environment overrides:
#   DATA_DIR     benchmark root (default: data/interactionbench)
#   OUT_DIR      run directory (default: results/runs/mage-vl-sliding-all1060)
#   ITEMS_FILE   item list (default: benchmark/splits/all1060.txt)
#   GPUS         comma-separated physical GPU IDs (default: 0)
#   RUN_MCQ_EVAL set to 0 to skip the mechanical MCQ evaluation
#   RUN_JUDGE    set to 0 to skip the Qwen3-14B paper-judge evaluation

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

data_dir=${DATA_DIR:-data/interactionbench}
out_dir=${OUT_DIR:-results/runs/mage-vl-sliding-all1060}
items_file=${ITEMS_FILE:-benchmark/splits/all1060.txt}
gpus_csv=${GPUS:-0}

IFS=',' read -r -a gpu_ids <<< "$gpus_csv"
num_shards=${#gpu_ids[@]}
if (( num_shards == 0 )); then
  echo "GPUS must contain at least one GPU ID" >&2
  exit 2
fi

for command_name in ibench ffmpeg; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "missing required command: $command_name" >&2
    exit 2
  fi
done

if ! ffmpeg -hide_banner -decoders 2>/dev/null \
    | grep -Eq 'libdav1d|libaom-av1|[[:space:]]av1[[:space:]]'; then
  echo "ffmpeg has no AV1 decoder; see docs/MAGE_VL_RESULTS.md" >&2
  exit 2
fi

mkdir -p "$out_dir/logs" "$out_dir/shards"

pids=()
for ((shard = 0; shard < num_shards; shard++)); do
  gpu=${gpu_ids[$shard]}
  shard_out="$out_dir/shards/$shard"
  log_file="$out_dir/logs/$shard.log"
  echo "launching shard $shard/$num_shards on GPU $gpu -> $log_file"
  (
    # Mage's nested processor asks once for trust confirmation. The checkpoint
    # revision itself is pinned in the model-zoo entry.
    set +o pipefail
    yes y | env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONUNBUFFERED=1 \
      HF_HUB_DISABLE_PROGRESS_BARS=1 \
      ibench run \
        --data "$data_dir" \
        --model mage-vl \
        --protocol sliding \
        --interval 1 \
        --sample-fps 2 \
        --max-frames 16 \
        --max-long-side 512 \
        --max-new-tokens 96 \
        --mcq \
        --items "$items_file" \
        --num-shards "$num_shards" \
        --shard-index "$shard" \
        --out "$shard_out"
  ) >"$log_file" 2>&1 &
  pids+=("$!")
done

run_failed=0
for ((shard = 0; shard < num_shards; shard++)); do
  if ! wait "${pids[$shard]}"; then
    echo "shard $shard failed; inspect $out_dir/logs/$shard.log" >&2
    run_failed=1
  fi
done
if (( run_failed )); then
  echo "Re-run this command to resume completed shard items." >&2
  exit 1
fi

shard_predictions=()
for ((shard = 0; shard < num_shards; shard++)); do
  shard_predictions+=("$out_dir/shards/$shard/preds.jsonl")
done
ibench merge "$out_dir/preds.jsonl" "${shard_predictions[@]}"

python - "$out_dir/preds.jsonl" "$items_file" <<'PY'
import json
import sys
from pathlib import Path

predictions = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line]
expected = {line.strip() for line in Path(sys.argv[2]).read_text().splitlines() if line.strip()}
item_ids = [f"{row['video_id']}#{row['item_index']}" for row in predictions]
failed = [row for row in predictions if row.get("failed")]
assert len(predictions) == len(expected), (len(predictions), len(expected))
assert len(set(item_ids)) == len(expected), "duplicate prediction IDs"
assert set(item_ids) == expected, "prediction IDs do not match the selected split"
assert not failed, f"{len(failed)} failed items"
print(
    f"validated rows={len(predictions)} unique={len(set(item_ids))} "
    f"failed={len(failed)} polls={sum(row['n_polls'] for row in predictions)}"
)
PY

if [[ ${RUN_MCQ_EVAL:-1} == 1 ]]; then
  ibench eval "$out_dir/preds.jsonl" \
    --data "$data_dir" \
    --items benchmark/splits/mcq688.txt \
    --mcq-key \
    --out "$out_dir/eval-mcq"
fi

if [[ ${RUN_JUDGE:-1} == 1 ]]; then
  CUDA_VISIBLE_DEVICES=${gpu_ids[0]} ibench eval "$out_dir/preds.jsonl" \
    --data "$data_dir" \
    --items "$items_file" \
    --mcq-key \
    --judge hf:Qwen/Qwen3-14B \
    --judge-cache results/judge_cache/qwen3-14b-v2.jsonl \
    --out "$out_dir/eval-qwen3-14b"
fi

echo "complete: $out_dir"
