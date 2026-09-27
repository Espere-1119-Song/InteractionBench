# JoyAI-VL-Interaction (native streaming service)

`run.py` drives the live adapter of JoyAI-VL-Interaction. The adapter keeps one session
per item on the server: the video history, the mid-term and long-term memory, and the
standing question. At every tick the runner sends only the frames that are new since
the previous tick. The adapter answers with `</silence>` (stay quiet) or
`</response> <text>` (speak now). The runner records every `</response>` with text as an
emission at the time of the tick.

- B and C items: the question is sent with every tick from t=0. One tick per
  `--interval` seconds (1.0) until the end of the video. A tick without new frames is
  skipped.
- A items: the frames up to `question_time_s` are ingested first in chunks of
  `--ingest-chunk` frames (32) with the text
  `"(No request yet. Keep watching silently.)"` and `max_tokens=8`; the replies are
  discarded. The last frame before the question is held back and sent together with the
  question. Ticks continue until `question_time_s + --a-window`.
- Every frame is declared to the adapter with a time range `t` to `t + 1/interval`
  seconds (`frame_time_ranges`).

Files:

| File | Content |
|---|---|
| `run.py` | the runner |
| `client.py` | HTTP client of the adapter (`JoyAISession`, `parse_marker`) |
| `serve.sh` | starts the serving stack: two vLLM servers and the adapter |
| `live_adapter.patch`, `memory_summarizer.patch` | local changes of the upstream checkout, needed for API kernels only |

## Upstream

| | |
|---|---|
| Code repository | https://github.com/jd-opensource/JoyAI-VL-Interaction |
| Checkpoint | `jdopensource/JoyAI-VL-Interaction-Preview` (https://huggingface.co/jdopensource/JoyAI-VL-Interaction-Preview) |
| Summarizer and long-term memory model | `Qwen/Qwen3-VL-4B-Instruct` |
| Checkout needed | Yes. `serve.sh` starts `services/webinfer/live_adapter.py`. The runner itself imports nothing from the checkout. |
| Upstream commit | `f8bcde678680cdaa7c7430dcb3e504eb30edbefd` (origin `https://github.com/jd-opensource/JoyAI-VL-Interaction`) |
| Local modifications of the checkout | `services/webinfer/live_adapter.py` and `services/webinfer/memory_summarizer.py`, see "Patches" |
| Checkpoint revision of the paper run | Not recorded. Revisions in the local cache when this runner was ported: `ad70e6f0ed63ec1a337bb1c67b82539b42f12d50` (JoyAI-VL-Interaction-Preview), `ebb281ec70b05090aa6165b016eac8ec08e71b17` (Qwen3-VL-4B-Instruct). |

```bash
git clone https://github.com/jd-opensource/JoyAI-VL-Interaction external/JoyAI-VL-Interaction
git -C external/JoyAI-VL-Interaction checkout f8bcde678680cdaa7c7430dcb3e504eb30edbefd
```

## Patches

The checkout of the paper runs differs from the upstream commit in two files. Both
changes are inactive unless an environment variable is set, so the run with the native
kernel and the runs with open-weight kernels behave as the unmodified upstream code.
The patches are required for API kernels.

```bash
git -C external/JoyAI-VL-Interaction apply "$PWD/baselines/joyai/live_adapter.patch"
git -C external/JoyAI-VL-Interaction apply "$PWD/baselines/joyai/memory_summarizer.patch"
```

`live_adapter.patch`:

| Change | Effect |
|---|---|
| `_strict_kwargs`, applied to both calls of `client.chat.completions.create` | When `ADAPTER_OAI_STRICT` is set, the request to the main model carries `max_tokens` only. The upstream code also sends `temperature`, `top_p`, `presence_penalty` and, in `extra_body`, the vLLM fields `top_k`, `repetition_penalty`, `greedy` and `skip_special_tokens`. Commercial OpenAI-compatible endpoints reject some of these fields. With the variable set, sampling follows the default of the provider. |
| `_cap_images`, called at the end of the message assembly | When `ADAPTER_MAX_IMAGES` is greater than 0 and a request holds more images, the oldest images are removed until that number remains. Text parts are kept. A user turn that loses all of its content receives the text `[earlier frames omitted]`. Unset or 0: no cap. |

`memory_summarizer.patch`:

| Change | Effect |
|---|---|
| branch in `SummarizerModel` before `client.chat.completions.create` | When `ADAPTER_OAI_STRICT` is set, the summarizer request carries `model`, `messages` and `max_tokens` only. |

`ADAPTER_OAI_STRICT` is tested for a non-empty value: `ADAPTER_OAI_STRICT=0` enables the
strict mode as well. Leave the variable unset or empty to disable it.

Relative to the working copy of the paper runs, the patch files differ in one docstring
line (a note was removed) and have no `index` header line. The code is unchanged.

## Environment

Versions of the environment of the paper run (Python 3.10):

| Package | Version |
|---|---|
| vllm | 0.28.0 |
| torch | 2.13.0 |
| transformers | 5.16.0 |
| openai | 3.3.1 |
| aiohttp | 3.14.3 |
| pillow | 11.3.0 |
| huggingface-hub | 1.28.0 |

The environment was created without version pins; the table lists what was installed.

```bash
python3.10 -m venv external/joyai-venv
external/joyai-venv/bin/pip install "vllm==0.28.0" aiohttp openai pillow huggingface_hub
external/joyai-venv/bin/pip install -e .
```

- vllm 0.28 compiles kernels when the engine starts and needs the CUDA compiler `nvcc`
  on `PATH` (`CUDA_HOME` set).
- The vLLM servers are started with `VLLM_USE_FLASHINFER_SAMPLER=0`.
- The runner needs Pillow, `interactionbench` and an `ffmpeg` binary on `PATH`. It can
  run in any environment that has these.

## Serving stack

`serve.sh` starts the stack and stays in the foreground. All settings are environment
variables; the header of the script lists them.

| `KERNEL` | Main model | Default ports (main / summarizer / adapter) |
|---|---|---|
| `native` | `jdopensource/JoyAI-VL-Interaction-Preview`, served by vLLM | 7060 / 8065 / 8070 |
| `open` | another open-weight model served by vLLM, default `Qwen/Qwen3-VL-8B-Instruct` | 7160 / 8165 / 8170 |
| `api` | a model behind an OpenAI-compatible API endpoint; the summarizer uses the same endpoint | none / none / 8071 |

```bash
# native kernel (paper run)
PYTHON=external/joyai-venv/bin/python bash baselines/joyai/serve.sh

# open-weight kernel
KERNEL=open PYTHON=external/joyai-venv/bin/python bash baselines/joyai/serve.sh

# API kernel; the key is read from the environment variable named by API_KEY_ENV
KERNEL=api API_BASE=https://api.anthropic.com/v1 API_KEY_ENV=ANTHROPIC_API_KEY \
    MAIN_MODEL=claude-sonnet-5 SUMM_MODEL=claude-haiku-4-5 \
    PYTHON=external/joyai-venv/bin/python bash baselines/joyai/serve.sh
```

Options of the paper runs, as set by `serve.sh`:

- both vLLM servers: `--max-model-len 262144`
- main server: `--gpu-memory-utilization 0.65 --enable-prefix-caching
  --enable-chunked-prefill --limit-mm-per-prompt '{"image":2048,"video":1}'`
- summarizer server: `--gpu-memory-utilization 0.20`
- adapter: `--adapter-model streaming-infer-adapter`, `--main-model`,
  `--summarizer-model` and `--longterm-model` set to the served model names; all other
  adapter options at their upstream defaults
- both servers ran on one GPU with 180 GB of memory; the memory fractions are sized for
  that GPU. Set `MAIN_CUDA_DEVICES`, `SUMM_CUDA_DEVICES`, `MAIN_GPU_FRACTION` and
  `SUMM_GPU_FRACTION` for other hardware.

The adapter can need more than a minute to start. `serve.sh` waits up to
`STARTUP_TIMEOUT` seconds (1800) and then prints the matching runner command.

Long-term memory endpoint. The launch commands of the paper runs set
`--summarizer-api-base` and do not set `--longterm-api-base`. The adapter then sends
the long-term memory compression requests to its default
`http://127.0.0.1:8065/v1`. For `KERNEL=native` this is the summarizer server. For
`KERNEL=open` with the ports above and for `KERNEL=api`, no server of the stack listens
on that port, and the long-term compression requests fail with a connection error.
`serve.sh` keeps this behaviour because the paper runs of these variants were produced
with it. To send the requests to the summarizer instead, set the variable
`LONGTERM_SUMMARIZER_API_BASE`, which the adapter reads; the result then deviates from
the paper runs of these variants.

## Paper run

| | |
|---|---|
| Run directory | `joyai_streaming_4fps_mcqv4` |
| Item set | 1,060-item set (all items) |
| Multiple-choice file | v4, `mcq/mcq_options_v4.jsonl` |
| Predictions stored | 1,060 lines |

```bash
python baselines/joyai/run.py --sample-fps 4 --max-long-side 448 \
    --mcq data/interactionbench/mcq/mcq_options_v4.jsonl \
    --video-dir data/interactionbench/videos_h264 \
    --out results/runs/joyai_streaming_4fps_mcqv4
```

What was checked:

- The launch log of the paper run records this command 49 times (the run was resumed
  after every interruption) and no other command.
- The stored `preds.jsonl` has 1,060 lines. 1,057 lines were written by the runner;
  their `run` field is `joyai_streaming_iv1_mcq`, the run tag that the runner derives
  from `--model-key` and `--interval`. The name of the output directory comes from
  `--out`.
- Three lines were added outside the runner for items that failed in every attempt:
  `OQpMvcvmxx0#0`, `Ol6UI4reKRs#0` and `_rV1rbRxdrg#0` (request larger than the context
  length or than the image limit of the server). They have empty emissions and an
  `error` field. The runner writes no line for a failed item; `ibench eval` scores an
  item without a line as silent, which gives the same score.
- The stored `config.json` of the run was written after the run. It states
  `max_long_side: 512` and a context length of 131072. The launch commands state
  `--max-long-side 448`. The context length of the main server was 131072, 163840 and
  262144 in different sessions of the run; `serve.sh` uses the last value.
- The first 237 items were produced on other hardware (one GPU with 80 GB,
  `--max-model-len 131072`, `--gpu-memory-utilization 0.85`), with the same runner
  options. The upstream commit and the package versions of that part were not recorded.

Runs with other kernels (103-item subset, `benchmark/splits/subset103.txt`, unless
stated otherwise):

| Run directory | Stack | Runner options in addition to the paper command |
|---|---|---|
| `joyai-qwen3vl8b-sub103_streaming_4fps_mcqv4` | `KERNEL=open` | `--base http://127.0.0.1:8170/v1 --items benchmark/splits/subset103.txt` |
| `joyai-claude-opus5-sub103_streaming_4fps_mcqv4` | `KERNEL=api`, `MAIN_MODEL=claude-opus-5`, `SUMM_MODEL=claude-haiku-4-5`, `ADAPTER_MAX_IMAGES=560` | `--base http://127.0.0.1:<port>/v1 --served-model claude-opus-5 --items benchmark/splits/subset103.txt` |
| `joyai-claude-sonnet5_streaming_4fps_mcqv4_merged` | `KERNEL=api`, `MAIN_MODEL=claude-sonnet-5`, `SUMM_MODEL=claude-haiku-4-5` | `--base http://127.0.0.1:<port>/v1 --served-model claude-sonnet-5`; two item lists, merged with `ibench merge` |
| `joyai-qwen4b_streaming_iv1_mcq` | not determined | not determined |

- The run with the Qwen3-VL-8B kernel did not pass `--served-model`. The request field
  `model` was `JoyAI-VL-Interaction-Preview`; the adapter uses its own `--main-model`.
- All runs kept `--model-key joyai`, so the `model` field of their predictions is
  `joyai`.
- The later sessions of the runs with the Opus kernel and with the Qwen3-VL-8B kernel
  used a variant of the runner that writes a prediction line with empty emissions and a
  `failed` field for an item that raised an error, so that the item is not attempted
  again. 39 of the 103 lines of the Opus run are of this kind. `run.py` does not contain
  this variant: a failed item gets no line and is attempted again at the next start.
  Both forms are scored as silent.
- The later sessions of the run with the Qwen3-VL-8B kernel used the memory fractions
  0.60 (main) and 0.30 (summarizer).
- No launch command of `joyai-qwen4b_streaming_iv1_mcq` was available when this runner
  was ported. The directory name equals the default output directory of
  `--model-key joyai-qwen4b --mcq`.
- API models change over time. A later run can differ from the stored predictions.

## Output

`results/runs/<model-key>_streaming_iv<interval>[_mcq]/` or the directory given with
`--out`: `preds.jsonl` (one line per item, appended, items already present are skipped)
and `raw/<item_id>.json` (every tick with the raw reply of the adapter, including the
silent ticks).

## Scoring

```bash
python -m interactionbench eval results/runs/joyai_streaming_4fps_mcqv4/preds.jsonl \
    --data data/interactionbench \
    --mcq-key data/interactionbench/mcq/mcq_key_v4.jsonl \
    --judge hf:Qwen/Qwen3-14B \
    --out results/runs/joyai_streaming_4fps_mcqv4/eval
```

- Without `--judge`, free-form content is scored lexically.
- Add `--items benchmark/splits/subset103.txt` for the runs on the 103-item subset.
- Items without a prediction line are scored as silent. Add `--skip-missing` to leave
  them out.

## Defaults that affect results

| Setting | Default | Paper run |
|---|---|---|
| `--sample-fps` | 2.0 | 4 |
| `--max-long-side` | 512 | 448 |
| `--interval` | 1.0 s | 1.0 s |
| `--a-window` | 10.0 s | 10.0 s |
| `--max-new-tokens` | 96 | 96, not applied by the adapter (see below) |
| `--ingest-chunk` | 32 frames | 32 frames |
| JPEG quality of the frames | 85 | 85 |
| request timeout | 900 s | 900 s |
| adapter `--main-max-tokens` | 128 | 128 |
| adapter sampling of the main model | temperature 0.8, top_p 0.9, top_k 40, repetition penalty 1.0, presence penalty 0.0 | the same (upstream defaults) |
| `--mcq` without a value | `data/interactionbench/mcq/mcq_options_v4.jsonl` | the v4 file was passed explicitly |

## Behaviour to be aware of

- The adapter samples from the main model with temperature 0.8. Two runs of the same
  command give different predictions.
- The runner sends `max_tokens` with every request (`--max-new-tokens`, and 8 for the
  ingest steps of A items). The adapter ignores generation parameters of incoming
  requests unless it is started with `--honor-inbound-generation-params`, which
  `serve.sh` and the paper runs do not do. The reply length is limited by the adapter
  option `--main-max-tokens` (128).
- `--mcq` without a value selects `mcq/mcq_options_v4.jsonl`.
- The items are processed in round-robin order over the capabilities (sorted by
  capability name, then by item id). `--limit N` takes the first N items of that order.
- The declared time range of a frame has the length `1/--interval` seconds and does not
  depend on `--sample-fps`. At 4 fps and an interval of 1 s the declared ranges of
  consecutive frames overlap.
- An item that raises an error (for example a request that exceeds the context length
  of the server) prints `ERROR` and writes no prediction line.
- For a reply that starts with `</response>`, `parse_marker` keeps the text up to the
  first `<delegation>` or `</delegation>` marker. A reply `</response>` without text is
  not recorded as an emission.
