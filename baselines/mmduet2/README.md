# MMDuet2 (native proactive streaming)

`run.py` feeds the video to `ProactiveInferenceAPIClient` of the MMDuet2 demo
(`demo/api_server.py`) frame by frame. The client collects frames into a turn. At the
end of every turn the model generates a reply or the text `NO REPLY`. A reply is
recorded as an emission at the video time that the client reports for that turn.

- B and C items: the question is added as text before the first frame.
- A items: frames are fed without a question until `question_time_s`; then the
  question is added. Frames are fed until `question_time_s + --a-window`.
- `--fps` sets the input frame rate. `--decision-hz` sets the number of decisions per
  second: a turn has `max(2, round(fps / decision_hz))` frames. The visual input keeps
  the full frame rate.
- Every frame is scaled to fit into 448 x 448 pixels.
- `--max-frames` limits the number of frames per item. When the limit is reached, the
  rest of the video is not processed.

## Upstream

| | |
|---|---|
| Code repository | https://github.com/yellow-binary-tree/MMDuet2 |
| Checkpoint | `wangyueqian/MMDuet2` (https://huggingface.co/wangyueqian/MMDuet2) |
| Checkout needed | Yes. The runner imports `demo/api_server.py`, which imports from `proactive_eval/`. |
| Upstream commit | `7bd91d5239adcb273df0977bc6c23faff950912c` (origin `https://github.com/yellow-binary-tree/MMDuet2`) |
| Local modifications of the checkout | None (`git status --short` is empty). No patch file is needed. |
| Checkpoint revision of the paper run | Not recorded. Revision `3224235ebb6256e2af7c196f2c265d6cf20c311a` was in the local cache when this runner was ported. |

Locations:

| Argument | Environment variable | Default |
|---|---|---|
| `--repo` | `MMDUET2_REPO` | `external/MMDuet2` |
| `--checkpoint` | none | `wangyueqian/MMDuet2` |

```bash
git clone https://github.com/yellow-binary-tree/MMDuet2 external/MMDuet2
git -C external/MMDuet2 checkout 7bd91d5239adcb273df0977bc6c23faff950912c
```

## Environment

Versions of the environment of the paper run (Python 3.10):

| Package | Version |
|---|---|
| torch | 2.7.1 (CUDA 12.8 build) |
| torchvision | 0.22.1 |
| transformers | 4.49.0 |
| qwen-vl-utils | 0.0.8, with the `decord` extra |
| decord | 0.6.0 |
| accelerate | 1.14.0 |
| gradio | 5.47.2 |
| six | 1.17.0 |
| pillow | 11.3.0 |

```bash
python3.10 -m venv external/mmduet2-venv
external/mmduet2-venv/bin/pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
external/mmduet2-venv/bin/pip install transformers==4.49.0 "qwen-vl-utils[decord]==0.0.8" \
    "six>=1.17.0" gradio==5.47.2 accelerate pillow huggingface_hub
external/mmduet2-venv/bin/pip install -e .
```

- The pins recorded for the first part of the paper run (717 items, other hardware) are
  torch 2.4 and transformers 4.49. The remaining 343 items ran in the environment of the
  table. torch 2.7.1 was chosen because the GPU of that part needs it.
- The runner loads the model with `attn_implementation="sdpa"`. `flash-attn` is not
  needed.
- `decord` is imported after the model is on the GPU. Importing it before the CUDA
  initialisation of torch can end the process with a segmentation fault.
- decord cannot read AV1 video. `--video-dir` defaults to
  `data/interactionbench/videos_h264`, one flat directory with H.264 copies named
  `<video_id>.mp4`.
- If decord reports end-of-file errors, set `DECORD_EOF_RETRY_MAX=20480`.

## Paper run

| | |
|---|---|
| Run directory | `mmduet2-3b_streaming_4fps_mcq_MERGED` |
| Item set | 1,060-item set (all items) |
| Multiple-choice file | v4, `mcq/mcq_options_v4.jsonl` |
| Predictions stored | 1,060 lines |

```bash
python baselines/mmduet2/run.py --mcq --fps 4 --decision-hz 0.5 --max-frames 880
```

`--mcq` without a value selects `data/interactionbench/mcq/mcq_options_v4.jsonl`. The
default output directory is `results/runs/mmduet2-3b_streaming_4fps_mcq`.

The paper run was split into item lists that ran in parallel, each with `--items` and
its own `--out`, and the prediction files were merged:

```bash
python baselines/mmduet2/run.py --mcq --fps 4 --decision-hz 0.5 --max-frames 880 \
    --items shard_0.txt --out results/runs/mmduet2-3b_streaming_4fps_mcq_shard0
python -m interactionbench merge results/runs/mmduet2-3b_streaming_4fps_mcq_MERGED/preds.jsonl \
    results/runs/mmduet2-3b_streaming_4fps_mcq_shard*/preds.jsonl
```

What was checked:

- All launch commands that were available use `--fps 4 --decision-hz 0.5
  --max-frames 880` with the v4 multiple-choice file and the H.264 video directory.
- The stored `preds.jsonl` has 1,060 lines with `run` equal to
  `mmduet2-3b_streaming_4fps_mcq`.
- 343 lines are identical to the lines of stored shard directories whose `config.json`
  states `fps: 4`, `decision_hz: 0.5`, `max_frames: 880`.
- The other 717 lines come from shards that ran on other hardware. Their launch script
  uses the same options. Their output directories were not available, so these lines
  could not be compared with shard files.
- The merged directory has no `config.json` and no `raw/` directory.

## Output

`results/runs/mmduet2-3b_streaming_<fps>fps[_mcq]/` or the directory given with `--out`:
`preds.jsonl` (one line per item, appended, items already present are skipped),
`raw/<item_id>.json` (the frame rate of the item and one entry per frame) and
`config.json`.

## Scoring

```bash
python -m interactionbench eval results/runs/mmduet2-3b_streaming_4fps_mcq_MERGED/preds.jsonl \
    --data data/interactionbench \
    --mcq-key data/interactionbench/mcq/mcq_key_v4.jsonl \
    --judge hf:Qwen/Qwen3-14B \
    --out results/runs/mmduet2-3b_streaming_4fps_mcq_MERGED/eval
```

- Without `--judge`, free-form content is scored lexically.
- Items without a prediction line are scored as silent. Add `--skip-missing` to leave
  them out.

## Defaults that affect results

| Setting | Default | Paper run |
|---|---|---|
| `--fps` | `auto`: the highest of 16, 8, 4 for which duration x fps is at most `--max-frames`, else 4 | 4 |
| `--decision-hz` | 1.0 | 0.5 (8 frames per turn, one decision every 2 s) |
| `--max-frames` | 1200 | 880 |
| `--a-window` | 10.0 s | 10.0 s |
| frame size | fits into 448 x 448 | the same |
| decoding | upstream defaults of the client (`do_sample=False`, `max_new_tokens=512`) | the same |

## Behaviour to be aware of

- At 4 fps, `--max-frames 880` covers the first 220 seconds of a video. For a longer
  video the model sees nothing after that point and cannot respond to later events.
  133 of the 1,060 stored prediction lines reached the limit.
- The upstream client processes the whole history again at every turn. The time per
  item grows with the square of the number of frames.
- `n_polls` counts frames, not decisions: the client returns a value for every frame,
  including the frames that only fill the turn buffer.
- When an exception occurs while an item is processed, the runner prints the error and
  writes a prediction line with the emissions collected up to that point. The item is
  not attempted again at the next start.
- The field `protocol` in `config.json` contains the text `2 frames/turn` for every
  setting. The number of frames per turn is `max(2, round(fps / decision_hz))`.
- The `fps` field of a raw file is taken from the previous item when the current item
  fails before its frame rate is set.
