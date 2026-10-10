#!/usr/bin/env bash
# Start the JoyAI-VL-Interaction serving stack used by baselines/joyai/run.py.
# Settings are environment variables; see README.md in this directory.

set -uo pipefail

KERNEL=${KERNEL:-native}
PYTHON=${PYTHON:-python}
JOYAI_REPO=${JOYAI_REPO:-external/JoyAI-VL-Interaction}
LOG_DIR=${LOG_DIR:-logs/joyai}
HOST=${HOST:-127.0.0.1}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1800}
SUMM_NAME=${SUMM_NAME:-Qwen3-VL-4B-Instruct}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAIN_GPU_FRACTION=${MAIN_GPU_FRACTION:-0.65}
SUMM_GPU_FRACTION=${SUMM_GPU_FRACTION:-0.20}

case "$KERNEL" in
  native)
    MAIN_MODEL=${MAIN_MODEL:-jdopensource/JoyAI-VL-Interaction-Preview}
    MAIN_NAME=${MAIN_NAME:-JoyAI-VL-Interaction-Preview}
    SUMM_MODEL=${SUMM_MODEL:-Qwen/Qwen3-VL-4B-Instruct}
    MAIN_PORT=${MAIN_PORT:-7060}; SUMM_PORT=${SUMM_PORT:-8065}
    ADAPTER_PORT=${ADAPTER_PORT:-8070} ;;
  open)
    MAIN_MODEL=${MAIN_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
    MAIN_NAME=${MAIN_NAME:-Qwen3-VL-8B-Instruct}
    SUMM_MODEL=${SUMM_MODEL:-Qwen/Qwen3-VL-4B-Instruct}
    MAIN_PORT=${MAIN_PORT:-7160}; SUMM_PORT=${SUMM_PORT:-8165}
    ADAPTER_PORT=${ADAPTER_PORT:-8170} ;;
  api)
    : "${API_BASE:?set API_BASE to the OpenAI-compatible endpoint root}"
    : "${MAIN_MODEL:?set MAIN_MODEL to the model id of the main model}"
    SUMM_MODEL=${SUMM_MODEL:-$MAIN_MODEL}
    API_KEY_ENV=${API_KEY_ENV:-API_KEY}
    export MODEL_API_KEY=${!API_KEY_ENV:-}
    : "${MODEL_API_KEY:?the environment variable named by API_KEY_ENV is empty}"
    export ADAPTER_OAI_STRICT=${ADAPTER_OAI_STRICT-1}
    [ -n "${ADAPTER_MAX_IMAGES:-}" ] && export ADAPTER_MAX_IMAGES
    ADAPTER_PORT=${ADAPTER_PORT:-8071} ;;
  *) echo "unknown KERNEL '$KERNEL' (native, open, api)" >&2; exit 2 ;;
esac

ADAPTER=$JOYAI_REPO/services/webinfer/live_adapter.py
[ -f "$ADAPTER" ] || { echo "live adapter not found: $ADAPTER (set JOYAI_REPO)" >&2; exit 2; }
mkdir -p "$LOG_DIR"

PIDS=()
cleanup() { [ ${#PIDS[@]} -gt 0 ] && kill "${PIDS[@]}" 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT
trap 'exit 130' INT TERM

healthy() {
  curl -s -m 4 "$1" | grep -q "$2"
}

if [ "$KERNEL" != "api" ]; then
  CUDA_VISIBLE_DEVICES=${MAIN_CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-}} \
  VLLM_USE_FLASHINFER_SAMPLER=0 "$PYTHON" -m vllm.entrypoints.cli.main serve "$MAIN_MODEL" \
    --port "$MAIN_PORT" --max-model-len "$MAX_MODEL_LEN" \
    --served-model-name "$MAIN_NAME" \
    --gpu-memory-utilization "$MAIN_GPU_FRACTION" \
    --enable-prefix-caching --enable-chunked-prefill \
    --limit-mm-per-prompt '{"image":2048,"video":1}' \
    >> "$LOG_DIR/main.log" 2>&1 &
  PIDS+=($!)
  CUDA_VISIBLE_DEVICES=${SUMM_CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-}} \
  VLLM_USE_FLASHINFER_SAMPLER=0 "$PYTHON" -m vllm.entrypoints.cli.main serve "$SUMM_MODEL" \
    --port "$SUMM_PORT" --max-model-len "$MAX_MODEL_LEN" \
    --served-model-name "$SUMM_NAME" \
    --gpu-memory-utilization "$SUMM_GPU_FRACTION" \
    >> "$LOG_DIR/summary.log" 2>&1 &
  PIDS+=($!)
  "$PYTHON" "$ADAPTER" \
    --host "$HOST" --port "$ADAPTER_PORT" --adapter-model streaming-infer-adapter \
    --main-api-base "http://127.0.0.1:$MAIN_PORT/v1" \
    --main-model "$MAIN_NAME" \
    --summarizer-api-base "http://127.0.0.1:$SUMM_PORT/v1" \
    --summarizer-model "$SUMM_NAME" \
    --longterm-model "$SUMM_NAME" \
    >> "$LOG_DIR/adapter.log" 2>&1 &
  PIDS+=($!)
  PAUSE=15
else
  "$PYTHON" "$ADAPTER" \
    --host "$HOST" --port "$ADAPTER_PORT" --adapter-model streaming-infer-adapter \
    --main-api-base "$API_BASE" --main-model "$MAIN_MODEL" \
    --summarizer-api-base "$API_BASE" --summarizer-model "$SUMM_MODEL" \
    --longterm-model "$SUMM_MODEL" \
    >> "$LOG_DIR/adapter.log" 2>&1 &
  PIDS+=($!)
  PAUSE=5
fi

echo "[joyai] waiting for the stack (logs in $LOG_DIR) ..."
UP=0
for i in $(seq 1 $((STARTUP_TIMEOUT / PAUSE))); do
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null || { echo "[joyai] a server exited during start-up; see $LOG_DIR" >&2; exit 1; }
  done
  if [ "$KERNEL" != "api" ]; then
    healthy "localhost:$MAIN_PORT/v1/models" "$MAIN_NAME" \
      && healthy "localhost:$SUMM_PORT/v1/models" "$SUMM_NAME" \
      && healthy "localhost:$ADAPTER_PORT/health" ok && { UP=1; break; }
  else
    healthy "localhost:$ADAPTER_PORT/health" ok && { UP=1; break; }
  fi
  sleep "$PAUSE"
done
[ "$UP" = 1 ] || { echo "[joyai] the stack did not become healthy in $STARTUP_TIMEOUT s" >&2; exit 1; }

RUN_ARGS="--base http://127.0.0.1:$ADAPTER_PORT/v1"
[ "$KERNEL" = "api" ] && RUN_ARGS="$RUN_ARGS --served-model $MAIN_MODEL"
cat <<EOF
[joyai] stack is up (kernel: $KERNEL, main model: ${MAIN_NAME:-$MAIN_MODEL}, adapter port: $ADAPTER_PORT)
[joyai] run from a second shell:
  python baselines/joyai/run.py $RUN_ARGS --sample-fps 4 --max-long-side 448 \\
      --mcq data/interactionbench/mcq/mcq_options_v4.jsonl \\
      --video-dir data/interactionbench/videos_h264 --out results/runs/<run name>
EOF
wait
