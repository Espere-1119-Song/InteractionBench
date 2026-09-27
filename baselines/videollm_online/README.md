# VideoLLM-online (native streaming with a trigger threshold)

`run.py` drives `LiveInfer` of the VideoLLM-online demo (`demo/inference.py`). The model
reads one frame at a time. After every frame it predicts either the frame-interval
token (stay silent and read the next frame) or the start of a response. `LiveInfer`
sets the probability of the frame-interval token to zero when it is below the trigger
threshold, so the model speaks. The upstream value of the threshold is 0.725. A
response is recorded as an emission at the video time that `LiveInfer` reports.

- B and C items: the question enters the query queue at video time 0.
- A items: the question enters the query queue at the first frame at or after
  `question_time_s`. Frames are processed until `question_time_s + --a-window`.
- Every video is resampled once with the upstream helper `ffmpeg_once` to `--fps`
  frames per second and to the frame resolution of the model (384 x 384, padded). The
  result is stored in `--cache-dir`.
- The text after the first `Assistant:` of a response is the content of the emission.

Files:

| File | Content |
|---|---|
| `run.py` | the runner with the upstream threshold 0.725 (paper run) |
| `run_threshold.py` | the runner with `--threshold` (threshold ablation) |

`run_threshold.py` differs from `run.py` in three points: the required argument
`--threshold` sets `LiveInfer.frame_token_interval_threshold`; the threshold is written
to `config.json`; an item is recorded as failed after one failed attempt instead of
three.

## Upstream

| | |
|---|---|
| Code repository | https://github.com/showlab/videollm-online |
| Checkpoint | LoRA `chenjoya/videollm-online-8b-v1plus` (https://huggingface.co/chenjoya/videollm-online-8b-v1plus) |
| Base LLM | `NousResearch/Meta-Llama-3-8B-Instruct`, a copy of `meta-llama/Meta-Llama-3-8B-Instruct` |
| Vision encoder | `google/siglip-large-patch16-384` |
| Checkout needed | Yes. The runner imports `demo/inference.py` and `data/utils.py`. |
| Upstream commit | `3da632217258b830a9a594bdabd7a40ae99646dc` (origin `https://github.com/showlab/videollm-online`) |
| Local modifications of the checkout | None (`git status --short` is empty). No patch file is needed. |
| Checkpoint revisions of the paper run | Not recorded. Revisions in the local cache when this runner was ported: `b6541f5208f887690856ebb705e2c6c3b71d0095` (LoRA), `53346005fb0ef11d3b6a83b12c895cca40156b6c` (base LLM), `ce005573a40965dfd21fd937fbdeeebf2439fc35` (vision encoder). |

Locations:

| Argument | Environment variable | Default |
|---|---|---|
| `--repo` | `VLLMONLINE_REPO` | `external/videollm-online` |
| `--checkpoint` | none | `chenjoya/videollm-online-8b-v1plus` |
| `--llm` | none | `NousResearch/Meta-Llama-3-8B-Instruct` |
| `--cache-dir` | `VLLMONLINE_CACHE` | `cache/videollm_online` |

```bash
git clone https://github.com/showlab/videollm-online external/videollm-online
git -C external/videollm-online checkout 3da632217258b830a9a594bdabd7a40ae99646dc
```

## Environment

Versions of the environment of the paper run (Python 3.10):

| Package | Version |
|---|---|
| torch | 2.7.1 (CUDA 12.8 build) |
| torchvision | 0.22.1 |
| torchaudio | 2.7.1 |
| transformers | 4.55.4 |
| peft | 0.20.0 |
| accelerate | 1.14.0 |
| av | 12.3.0 |
| pillow | 11.3.0 |

```bash
python3.10 -m venv external/vllmonline-venv
external/vllmonline-venv/bin/pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
external/vllmonline-venv/bin/pip install transformers==4.55.4 accelerate peft editdistance \
    Levenshtein tensorboard gradio moviepy submitit av==12.3.0 pillow huggingface_hub
external/vllmonline-venv/bin/pip install -e .
```

- `av==12.3.0` is a required pin of the paper environment.
- The upstream helper `ffmpeg_once` calls `./ffmpeg/ffmpeg` relative to the working
  directory. Create this path in the directory from which the runner is started:

  ```bash
  mkdir -p ffmpeg && ln -s "$(command -v ffmpeg)" ffmpeg/ffmpeg
  ```

- The runner loads the model with `--attn_implementation sdpa`. `flash-attn` is not
  needed.
- The runner changes the working directory to the checkout while `LiveInfer` is built
  and changes back afterwards. Relative paths in the arguments refer to the directory
  from which the runner is started.
- Use the H.264 copies of the videos (`--video-dir`, default
  `data/interactionbench/videos_h264`, one flat directory with files named
  `<video_id>.mp4`).

## Paper run

| | |
|---|---|
| Run directory | `videollm-online-8b_streaming_8fps_mcq` |
| Item set | 1,060-item set (all items) |
| Multiple-choice file | v4, `mcq/mcq_options_v4.jsonl` |
| Predictions stored | 1,060 lines |

```bash
python baselines/videollm_online/run.py --mcq --fps 8
```

`--mcq` without a value selects `data/interactionbench/mcq/mcq_options_v4.jsonl`. The
default output directory is `results/runs/videollm-online-8b_streaming_8fps_mcq`.

What was checked:

- The stored `config.json` has `"fps": 8`, `"a_window_s": 10.0`, `"attn": "sdpa"`,
  `"n_items_targeted": 1060` and a threshold of 0.725 in its `protocol` text. Its
  `"mcq_options"` entry names `mcq/mcq_options_v4.jsonl`, and its `"video_dir"` entry
  names the `videos_h264` directory.
- The stored `preds.jsonl` has 1,060 lines. 18 lines have empty emissions and the field
  `"error": "deterministic_failure_x3"`: the item failed in three starts of the runner.
- 1,038 of the lines were produced on other hardware with the same command and the
  same recorded pins (torch 2.7.1, transformers 4.55.4, av 12.3.0).

Threshold ablation (103-item subset; stored directories `videollm_threshold0.5_0921`
and `videollm_threshold0.9_0921`):

```bash
python baselines/videollm_online/run_threshold.py --threshold 0.5 --mcq --fps 8 \
    --items benchmark/splits/subset103.txt --out results/runs/videollm_threshold0.5
python baselines/videollm_online/run_threshold.py --threshold 0.9 --mcq --fps 8 \
    --items benchmark/splits/subset103.txt --out results/runs/videollm_threshold0.9
```

Pass `--out`: the default output directory of `run_threshold.py` is the one of `run.py`.
In the stored predictions, 1 of 103 items (threshold 0.5) and 38 of 103 items
(threshold 0.9) are recorded as failed with the field
`"error": "deterministic_failure_x1"`.

## Output

`results/runs/videollm-online-8b_streaming_<fps>fps[_mcq]/` or the directory given with
`--out`: `preds.jsonl` (one line per item, appended, items already present are skipped),
`raw/<item_id>.json` (number of frames and the emissions), `config.json` and
`fail_counts.json` (number of failed attempts per item).

## Scoring

```bash
python -m interactionbench eval results/runs/videollm-online-8b_streaming_8fps_mcq/preds.jsonl \
    --data data/interactionbench \
    --mcq-key data/interactionbench/mcq/mcq_key_v4.jsonl \
    --judge hf:Qwen/Qwen3-14B \
    --out results/runs/videollm-online-8b_streaming_8fps_mcq/eval
```

- Without `--judge`, free-form content is scored lexically.
- Add `--items benchmark/splits/subset103.txt` for the threshold ablation.
- Items without a prediction line are scored as silent. Add `--skip-missing` to leave
  them out.

## Defaults that affect results

| Setting | Default | Paper run |
|---|---|---|
| `--fps` | 8 | 8 |
| `--a-window` | 10.0 s | 10.0 s |
| trigger threshold | 0.725 (set by the upstream `LiveInfer`) | 0.725 |
| frame resolution | 384, from the model configuration | 384 |
| failed attempts before an item is recorded as failed | 3 (`run.py`), 1 (`run_threshold.py`) | the same |

The upstream repository states that the model was trained at 2 fps and that inference
supports up to about 10 fps.

## Behaviour to be aware of

- The outputs have the style of Ego4D narration. The model often does not answer the
  question and describes the scene from the view of the camera wearer.
- An item that raises an exception writes no prediction line in that start.
  `fail_counts.json` counts the failure. A later start of the runner tries the item
  again. When the count has reached the limit, the next start writes a line with empty
  emissions and an `error` field. A run is complete after the runner has been started
  often enough for every item to have a line.
- A failure observed in the paper run: for very long items the assertion on the stream
  token in `LiveInfer._call_for_response` fails.
- `poll_latencies` is empty in every prediction line. The latency of an emission is in
  its `latency_s` field. `n_polls` is the number of processed frames.
- Command line arguments that the runner does not know are ignored without a message.
