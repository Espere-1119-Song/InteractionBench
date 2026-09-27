# LiveCC-7B-Instruct (streaming commentary)

`run.py` drives the `live_cc` loop of the LiveCC demo (`demo/infer.py`). LiveCC is a
commentary model: given a query, it emits a short text fragment for every clip of about
one second. It has no mechanism to stay silent. The runner advances the video timestamp
in steps of 1 second and records every non-empty fragment at the stop time of its clip.
Consecutive fragments are joined into sentences. A sentence carries the time of its
first fragment.

- B and C items: the question is the query from t=0.
- A items: the query is `"Please describe what is happening."` until
  `question_time_s`, then the question. Fragments before `question_time_s` are not
  counted as emissions. The loop continues until `question_time_s + --a-window`.

Because the model comments on every clip, a low Silence Compliance score is expected.

## Upstream

| | |
|---|---|
| Code repository | https://github.com/showlab/livecc |
| Checkpoint | `chenjoya/LiveCC-7B-Instruct` (https://huggingface.co/chenjoya/LiveCC-7B-Instruct) |
| Checkout needed | Yes. The runner imports `demo/infer.py` and `livecc-utils/src/livecc_utils`. |
| Upstream commit of the paper run | Not determined. The checkout used for the paper run was not available when this runner was ported. |
| Local modifications of the checkout | Not determined, for the same reason. No patch file is provided. |
| Checkpoint revision of the paper run | Not recorded. Revision `dbe7415c5d2024c3e2d7f898ecbc58fd9fc8da2f` was in the local cache when this runner was ported. |

Locations:

| Argument | Environment variable | Default |
|---|---|---|
| `--livecc-repo` | `LIVECC_REPO` | `external/livecc` |
| `--model` | none | `chenjoya/LiveCC-7B-Instruct` |

```bash
git clone https://github.com/showlab/livecc external/livecc
```

## Environment

Pins recorded for the paper run: transformers 4.50, qwen-vl-utils 0.0.8,
liger-kernel 0.5.5. The torch version was not recorded, and the environment was not
preserved.

```bash
python -m venv external/livecc-venv
external/livecc-venv/bin/pip install -e .
external/livecc-venv/bin/pip install torch "transformers==4.50.*" "qwen-vl-utils==0.0.8" \
    "liger-kernel==0.5.5" accelerate decord
```

- `flash-attn` is not required. The upstream constructor of `LiveCCDemoInfer` requests
  `flash_attention_2`; the runner builds the same object with `attn_implementation="sdpa"`
  (function `_build_infer`).
- `livecc_utils` is imported from the checkout, so the `livecc-utils` package does not
  need to be installed.
- decord cannot read AV1 video. For a data set that contains AV1 files, create H.264
  copies in one flat directory (`<video_id>.mp4`) and pass it with `--video-dir`.
- If decord reports end-of-file errors, set `DECORD_EOF_RETRY_MAX=20480`.

## Paper run

| | |
|---|---|
| Run directory | `livecc-7b_streaming_iv1_mcq` |
| Item set | 218-item subset, `benchmark/splits/frozen218.txt` |
| Multiple-choice file | An earlier file, `mcq/mcq_options.jsonl`. Not the v4 file. |
| Predictions stored | 218 lines |

The paper run of this system did **not** use the 1,060-item set with the v4
multiple-choice file. What was checked:

- The stored `preds.jsonl` contains exactly the 218 ids of the subset.
- The stored `eval_report.txt` reports 218 items and an answer key with 72 entries. The
  v4 key has 688 entries.
- The stored `summary.json` has no `config` block.
- The launcher command recorded in the original runner passes the 218-item list and
  `--mcq` without a value, which selected `mcq/mcq_options.jsonl` in the original runner.
- A run on the 1,060-item set with the v4 file was launched. Its output directory
  contains no predictions.

Command of the paper run as recorded in the original runner:

```bash
python baselines/livecc/run.py \
    --items benchmark/splits/frozen218.txt \
    --mcq data/interactionbench/mcq/mcq_options.jsonl
```

In this repository `--mcq` without a value selects `mcq/mcq_options_v4.jsonl`. The recorded command has no
`--video-dir`. Whether the paper run read the original videos or the H.264 copies is
not determined. The earlier multiple-choice file was not available when this runner was
ported, so the stored predictions could not be regenerated for comparison.

Command for the 1,060-item set with the v4 file (no paper numbers exist for it):

```bash
python baselines/livecc/run.py \
    --mcq data/interactionbench/mcq/mcq_options_v4.jsonl \
    --video-dir data/interactionbench/videos_h264 \
    --out results/runs/livecc-7b_streaming_mcqv4_full
```

## Output

`results/runs/livecc-7b_streaming_iv1[_mcq]/` or the directory given with `--out`:
`preds.jsonl` (one line per item, appended, items already present are skipped) and
`raw/<item_id>.json` (every fragment before sentence grouping).

## Scoring

```bash
python -m interactionbench eval results/runs/livecc-7b_streaming_iv1_mcq/preds.jsonl \
    --data data/interactionbench \
    --mcq-key data/interactionbench/mcq/mcq_key_v4.jsonl \
    --items benchmark/splits/frozen218.txt \
    --out results/runs/livecc-7b_streaming_iv1_mcq/eval
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

| Setting | Value |
|---|---|
| step of the video timestamp | 1.0 s |
| `--a-window` | 10.0 s |
| decoding | `do_sample=False`, `repetition_penalty=1.05` |
| query before an A-type question | `"Please describe what is happening."` |
| sentence end characters | `.`, `!`, `?` and their full-width forms |
| trailing `...` of a fragment | removed (end-of-stream marker of LiveCC) |

When an exception occurs inside the loop of one item, the runner prints the error and
writes the emissions collected up to that point.
