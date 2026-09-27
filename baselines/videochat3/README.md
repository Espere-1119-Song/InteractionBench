# VideoChat3-4B (native streaming interface)

`run.py` drives the `StreamingSession` class that ships with the VideoChat3 checkpoint.
The video is fed in rounds of one second. In each round the model emits `</Silence>`,
`</Standby>` or `</Response> <text>`. A `</Response>` becomes an emission at the end of
that round. After a `</Standby>` the next round is fed at four times the pixel budget.

- B and C items: the question is a standing request from round 0.
- A items: the question is given at the round of `question_time_s`. Earlier rounds only
  ingest frames. Rounds continue until `question_time_s + --a-window`.

## Upstream

| | |
|---|---|
| Code repository | https://github.com/MCG-NJU/VideoChat3 |
| Checkpoint | `MCG-NJU/VideoChat3-4B` (https://huggingface.co/MCG-NJU/VideoChat3-4B) |
| Checkout needed | No. `inference_fast_vc3.py` and `demo_vc3_proactive.py` are imported from the checkpoint snapshot. |
| Upstream commit | Not applicable (no checkout). |
| Checkpoint revision of the paper run | Not recorded. Revision `37fa901ec5913f84bc31108ebc1e60ad1903634c` was in the local cache when this runner was ported. |

`--model` takes a Hugging Face repository id, because the runner calls
`huggingface_hub.snapshot_download` on it.

## Environment

The original runner records no version pins, and the environment of the paper run was
not preserved. The model card lists these packages:

```bash
python -m venv external/videochat3-venv
external/videochat3-venv/bin/pip install -e .
external/videochat3-venv/bin/pip install torch "transformers<5" accelerate \
    qwen-vl-utils decord opencv-python-headless huggingface_hub
```

- An `ffmpeg` binary must be on `PATH` (frame extraction of `interactionbench.frames`).
- transformers 5.16.0 with torch 2.13.0 does not work: the remote code of the checkpoint
  imports `BASE_VIDEO_PROCESSOR_DOCSTRING` from `transformers.video_processing_utils`,
  and the import fails. The bound `transformers<5` follows from this observation. The
  exact version of the paper run is not determined.

## Paper run

| | |
|---|---|
| Run directory | `videochat3-4b_streaming_iv1_mcq` |
| Item set | 218-item subset, `benchmark/splits/frozen218.txt` |
| Multiple-choice file | An earlier file, `mcq/mcq_options.jsonl`. Not the v4 file. |
| Predictions stored | 247 lines: the 218 items of the subset and 29 items from a supplementary run. Scores use the 218 items. |

The paper run of this system did **not** use the 1,060-item set with the v4
multiple-choice file. What was checked:

- The stored `preds.jsonl` contains all 218 ids of the subset.
- The stored `eval_report.txt` reports 218 items and an answer key with 72 entries. The
  v4 key has 688 entries.
- The stored `summary.json` has no `config` block.
- The launcher command recorded in the original runner passes the 218-item list and
  `--mcq` without a value, which selected `mcq/mcq_options.jsonl` in the original runner.
- A run on the 1,060-item set with the v4 file was launched with
  `--target-fps 4 --mcq <v4 file>`. Its output directory contains no predictions.

Command of the paper run:

```bash
python baselines/videochat3/run.py \
    --items benchmark/splits/frozen218.txt \
    --mcq data/interactionbench/mcq/mcq_options.jsonl
```

In this repository `--mcq` without a value selects `mcq/mcq_options_v4.jsonl`. The earlier multiple-choice file was not
available when this runner was ported, so the stored predictions could not be
regenerated for comparison.

Command for the 1,060-item set with the v4 file (no paper numbers exist for it):

```bash
python baselines/videochat3/run.py \
    --mcq data/interactionbench/mcq/mcq_options_v4.jsonl \
    --out results/runs/videochat3-4b_streaming_4fps_mcqv4_full
```

## Output

`results/runs/videochat3-4b_streaming_iv1[_mcq]/` or the directory given with `--out`:
`preds.jsonl` (one line per item, appended, items already present are skipped) and
`raw/<item_id>.json` (every round with the raw model output).

## Scoring

```bash
python -m interactionbench eval results/runs/videochat3-4b_streaming_iv1_mcq/preds.jsonl \
    --data data/interactionbench \
    --mcq-key data/interactionbench/mcq/mcq_key_v4.jsonl \
    --items benchmark/splits/frozen218.txt \
    --out results/runs/videochat3-4b_streaming_iv1_mcq/eval
```

- Use the answer key that belongs to the options file used for generation.
- Leave out `--items` for the 1,060-item set.
- Add `--judge <spec>` to grade free-form content with a model. Without a judge,
  free-form content is scored lexically.
- One id of the 218-item list (`0afNDynOHdM#0`) is absent from the current annotation
  set, so the command above scores 217 items.
- The stored scores of the paper run were computed with an earlier evaluator
  configuration (judge `Qwen3-VL-4B-Instruct` behind an API endpoint, 72-entry answer
  key, `Delta=5.0s`, `gate=0.3`), followed by a re-scoring of counting items with exact
  match. `ibench eval` with default settings does not reproduce those values.

## Defaults that affect results

| Argument | Default | Meaning |
|---|---|---|
| `--target-fps` | 4.0 | frames extracted per second; at most `round(target_fps)` frames per round |
| `--max-pixels` | 50176 (224 x 224) | pixel budget per frame; 4 x this value after `</Standby>` |
| `--max-rounds` | 32 | passed to `StreamingSession` |
| `--max-new-tokens` | 96 | generation limit per round |
| `--a-window` | 10.0 | seconds of rounds after an A-type question |

Decoding is greedy (`do_sample=False`). The run tag contains `iv1` (one decision per
second) for every value of `--target-fps`.
