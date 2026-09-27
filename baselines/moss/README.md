# MOSS-Video-Preview (real-time streaming)

`run.py` feeds frames to `model.real_time_generate` at the pace of the video: a frame
with timestamp `t` is put on the image queue when the wall clock reaches `t`. The model
emits tokens continuously, `<|silence|>` while it observes and text tokens when it
speaks. Consecutive text tokens form one utterance. The emission time of an utterance is
the wall-clock time of its first token, measured from the start of the feed. A video of
60 seconds therefore takes about 68 seconds to evaluate (stream length plus 8 seconds of
listening after the last frame).

- B and C items: the question is pushed about 1 second after the feed starts.
- A items: the question is pushed when the stream reaches `question_time_s`. The feed
  stops at `question_time_s + --a-window`.
- Utterances that start more than 0.2 seconds before the question are dropped.

## Upstream

| | |
|---|---|
| Code repository | https://github.com/OpenMOSS/MOSS-Video-Preview |
| Checkpoint | `OpenMOSS-Team/moss-video-preview-realtime-sft` (https://huggingface.co/OpenMOSS-Team/moss-video-preview-realtime-sft) |
| Checkout needed | No. The model code is loaded from the checkpoint with `trust_remote_code=True`. |
| Upstream commit | Not applicable (no checkout). |
| Checkpoint revision of the paper run | Not recorded. Revision `102d8269fa6f5819cea9fbdcd50f46a55909c48f` was in the local cache when this runner was ported. |

`--model` (environment variable `MOSS_MODEL`) takes a repository id or a local
directory. The default is the repository id above.

## Environment

The original runner states a dedicated environment with torch 2.4 and transformers 4.46.
That environment was not preserved, so no further pins are recorded.

```bash
python -m venv external/moss-venv
external/moss-venv/bin/pip install -e .
external/moss-venv/bin/pip install "torch==2.4.*" "transformers==4.46.*" accelerate av \
    opencv-python-headless
```

- `av` is required by the remote code of the checkpoint. Without it the processor fails
  to load with an `ImportError`.
- An `ffmpeg` binary must be on `PATH` (frame extraction of `interactionbench.frames`).
- The model card lists Python 3.12.4 with PyTorch 2.4.0 (CUDA 12.1) as the tested setup.
- The runner loads the model with `attn_implementation="sdpa"` and `torch.bfloat16`.
- Emission times depend on the wall clock. Run one process per GPU and do not share the
  GPU with other jobs.

## Paper run

| | |
|---|---|
| Run directory | `moss-video-preview_streaming_rt_mcq` |
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

Command of the paper run:

```bash
python baselines/moss/run.py \
    --items benchmark/splits/frozen218.txt \
    --mcq data/interactionbench/mcq/mcq_options.jsonl
```

In this repository `--mcq` without a value selects `mcq/mcq_options_v4.jsonl`. The earlier multiple-choice file was not
available when this runner was ported, so the stored predictions could not be
regenerated for comparison.

Command for the 1,060-item set with the v4 file (no paper numbers exist for it):

```bash
python baselines/moss/run.py \
    --mcq data/interactionbench/mcq/mcq_options_v4.jsonl \
    --out results/runs/moss-video-preview_streaming_rt_mcqv4_full
```

## Output

`results/runs/moss-video-preview_streaming_rt[_mcq]/preds.jsonl` or `<--out>/preds.jsonl`.
One line per item, appended. Items already present are skipped.

The lines of this runner have no `poll_latencies` field, and the emissions have no
`latency_s` field. `n_polls` is the number of frames fed. A line has a `gen_error` field
when the generation thread raised an exception. The evaluator accepts these lines; the
latency columns stay empty.

## Scoring

```bash
python -m interactionbench eval results/runs/moss-video-preview_streaming_rt_mcq/preds.jsonl \
    --data data/interactionbench \
    --mcq-key data/interactionbench/mcq/mcq_key_v4.jsonl \
    --items benchmark/splits/frozen218.txt \
    --out results/runs/moss-video-preview_streaming_rt_mcq/eval
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
| `--fps` | 1.0 | frames extracted and fed per second |
| `--max-long-side` | 448 | frame resize bound |
| `--max-new-tokens` | 86400 | passed to `real_time_generate` |
| `--a-window` | 10.0 | seconds of stream after an A-type question |

Decoding is greedy (`do_sample=False`).
