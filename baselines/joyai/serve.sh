#!/usr/bin/env bash
# Start the JoyAI-VL-Interaction serving stack that baselines/joyai/run.py talks to.
#
# The stack has three processes:
#   main model server      vLLM, OpenAI-compatible (not started when KERNEL=api)
#   summarizer server      vLLM, OpenAI-compatible (not started when KERNEL=api)
#   live adapter           services/webinfer/live_adapter.py of the upstream repository
#
# Upstream repository: https://github.com/jd-opensource/JoyAI-VL-Interaction
#
# Three kernel variants, selected with KERNEL:
#   KERNEL=native  main model = jdopensource/JoyAI-VL-Interaction-Preview (paper run)
#   KERNEL=open    main model = another open-weight model served by vLLM
#                  (default Qwen/Qwen3-VL-8B-Instruct)
#   KERNEL=api     main model and summarizer behind one OpenAI-compatible API endpoint;
#                  no GPU is needed. Requires the patches live_adapter.patch and
#                  memory_summarizer.patch (see README.md).
#
# Every setting is an environment variable. Defaults reproduce the paper runs.
#
#   common
#     PYTHON            python of the environment with vllm, aiohttp, openai   [python]
#     JOYAI_REPO        checkout of the upstream repository   [external/JoyAI-VL-Interaction]
#     LOG_DIR           directory of the three log files                       [logs/joyai]
#     HOST              bind address of the adapter                            [127.0.0.1]
#     STARTUP_TIMEOUT   seconds to wait for the stack to become healthy        [1800]
#     ADAPTER_PORT      native 8070 | open 8170 | api 8071
#   KERNEL=native and KERNEL=open
#     MAIN_MODEL        Hugging Face id or local path of the main model
#     MAIN_NAME         served model name of the main model
#     MAIN_PORT         native 7060 | open 7160
#     SUMM_MODEL        summarizer / long-term memory model   [Qwen/Qwen3-VL-4B-Instruct]
#     SUMM_NAME         served model name of the summarizer   [Qwen3-VL-4B-Instruct]
#     SUMM_PORT         native 8065 | open 8165
#     MAX_MODEL_LEN     context length of both servers                         [262144]
#     MAIN_GPU_FRACTION vLLM --gpu-memory-utilization of the main server       [0.65]
#     SUMM_GPU_FRACTION vLLM --gpu-memory-utilization of the summarizer        [0.20]
#     MAIN_CUDA_DEVICES, SUMM_CUDA_DEVICES
#                       CUDA_VISIBLE_DEVICES of each server [inherited]. The paper runs
#                       put both servers on one 180 GB GPU; the fractions above are
#                       sized for that. Use two GPUs and larger fractions otherwise.
#   KERNEL=api
#     API_BASE          endpoint root, e.g. https://api.anthropic.com/v1       [required]
#     API_KEY_ENV       NAME of the environment variable that holds the key    [API_KEY]
#     MAIN_MODEL        model id of the main model                             [required]
#     SUMM_MODEL        model id of the summarizer / long-term memory model    [MAIN_MODEL]
#     ADAPTER_OAI_STRICT  send only max_tokens to the endpoint                 [1]
#     ADAPTER_MAX_IMAGES  keep at most this many images per request            [unset = no cap]
#
# Long-term memory endpoint: the adapter is started with the same options as in the
# paper runs, which set --summarizer-api-base but not --longterm-api-base. The adapter
# then sends long-term memory compression requests to its built-in default
# http://127.0.0.1:8065/v1. This equals the summarizer endpoint only for KERNEL=native
# with the default SUMM_PORT. The adapter reads the environment variable
# LONGTERM_SUMMARIZER_API_BASE; setting it changes the endpoint and deviates from the
# paper runs of the "open" and "api" variants (see README.md).
#
# Usage:
#   bash baselines/joyai/serve.sh                              # native kernel
#   KERNEL=open bash baselines/joyai/serve.sh                  # Qwen3-VL-8B as main model
#   KERNEL=api API_BASE=https://api.anthropic.com/v1 API_KEY_ENV=ANTHROPIC_API_KEY \
#     MAIN_MODEL=claude-sonnet-5 SUMM_MODEL=claude-haiku-4-5 bash baselines/joyai/serve.sh
#
# The script stays in the foreground while the stack runs. Stop it with Ctrl-C; the
# servers are stopped with it. Run baselines/joyai/run.py from a second shell.

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
    # The adapter reads the key from MODEL_API_KEY; the key is never put on a command line.
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

healthy() {  # healthy <url> <pattern>
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
  PAUSE=15                     # the model servers need several minutes to load
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
