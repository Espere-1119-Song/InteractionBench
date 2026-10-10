# OneStreamer-4B (native streaming interface)

`run.py` feeds the video to `NativeSession` in rounds of one second, with four new
frames per full round. The model emits `</Silence>`, `</Standby>` or
`</Response> <text>`. A valid `</Response>` becomes an emission at the end of that
round. Standby leaves the pixel budget unchanged.

- B and C items: the question is a standing request from round 0.
- A items: the question and options are revealed at the first decision time at or
  after `question_time_s`. Earlier rounds accumulate a rolling frame history with
  synthetic Silence replies and no model calls. Rounds end at the earlier of the
  video end and `question_time_s + --a-window`.
- A final partial round is retained. For example, a question revealed at the end
  of a 12.6-second video is first given at 12.6 seconds.

The session retains up to 32 rounds, including previous model replies and the
active question. It rebuilds the bounded context at each call; there is no
cross-round KV or visual-feature cache. Timing uses the video clock, without
adding inference latency.

## Upstream

| | |
|---|---|
| Code repository | https://github.com/MCG-NJU/OneStreamer |
| Checkpoint | [MCG-NJU/OneStreamer-4B](https://huggingface.co/MCG-NJU/OneStreamer-4B) |
| Checkout needed | No. `inference.py` contains the model engine and session adapter. |
| Source engine | [`Eval/Proactive_Eval/shared/inference_fast.py`](https://github.com/MCG-NJU/OneStreamer/blob/24ffb1093570d02a76eac5fdf59aff6a763e7481/Eval/Proactive_Eval/shared/inference_fast.py); source Git blob `19314902809b48bb34c0c4efabb3da125b69a2ea` |
| Default checkpoint revision | `fba29e66908b424877951fa8de46459537c8aa23` |

`--model` accepts a Hugging Face repository id or a downloaded model directory.
The default public checkpoint revision is pinned. `--revision` selects a different
Hub revision; local directories are identified by file hashes. Use `--model-name`
to label a different model. The runner loads the processor and weights from the
same resolved snapshot with Transformers' `AutoModelForImageTextToText`.

The engine is adapted from the OneStreamer evaluation code: unused video/demo
utilities are omitted, while message ordering, preprocessing and greedy generation
are retained. The session adds benchmark question scheduling and bounded multi-frame
rounds. Copyright 2026 OneStreamer contributors. These integration files are
licensed under [Apache-2.0](LICENSE). Model checkpoints and dataset media remain
subject to their respective source terms.

## Environment

Use an existing compatible environment. The reference inference environment is:

| Package | Version |
|---|---|
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| transformers | 4.57.6 |
| accelerate | 1.12.0 |
| Pillow | 12.0.0 |
| flash-attn | 2.7.4.post1 |

`huggingface_hub` is needed for Hub downloads. `ffmpeg` and `ffprobe` must be on
`PATH` to build frame caches; the reference cache used ffmpeg 6.1.1. This runner
does not need OpenCV or a separate streaming SDK checkout. It runs directly from
the repository root. Help, frame preparation and result merging do not load a GPU
model. Package installation and version changes are never performed by the runner.

## Reference run

This is an additional baseline, not a run included in the original InteractionBench
paper. The reference configuration covers all 1,060 items with MCQ v4 and the
unchanged [system prompt](system_prompt.txt). Its prompt SHA-256 is
`94d1034356468a84dfd758a7c543aedb82b1d6ba638b02dc9091be4ef9478055`.

This runner was validated on five items covering A/B/C tasks, including a negative
item and a late question after the context window rolled over. Response texts and
emission times matched the corresponding outputs from the completed full-benchmark
reference evaluation. The benchmark test suite and the added integration tests pass.

Command for the full set:

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/onestreamer/run.py \
    --data data/interactionbench \
    --mcq \
    --out results/runs/onestreamer_native
```

`--mcq` without a value reads `<data>/mcq/mcq_options_v4.jsonl`. Use `--items` for
an explicit item list and `--limit` for a small smoke test. Missing annotations,
videos or required MCQ options are errors. Only the formatted question, observed
frames and conversation history reach the model; answer keys, reference responses
and negative labels are excluded.

### Frame cache

Frames are cached under `results/frame_cache/onestreamer_4fps/<domain>/<video_id>/`.
The first frame is available at 0.25 seconds, named `00000_25.jpeg`, followed by
`00000_50.jpeg`, `00000_75.jpeg`, `00001_00.jpeg`, and so on. Each round reads only
frames in `(start, end]`. Precomputing the cache does not expose future frames to
the model.

Missing caches are built automatically. They can also be prepared separately:

```bash
python baselines/onestreamer/run.py --prepare-frames \
    --data data/interactionbench \
    --frames-root results/frame_cache/onestreamer_4fps \
    --frame-workers 8
```

Use `--cache-policy require` to forbid cache construction during inference. A
`--dry-run --cache-policy require` checks data and cache manifests without loading
weights. Source hashes and cache manifests must match; JPEG bytes are checked when
read. Interrupted extraction never publishes a complete cache.

The sampling filter is:

```text
setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=0.25,fps=4:start_time=0.25:round=up:eof_action=pass
```

Extraction stops at `floor(source_duration * 4) / 4`, with JPEG quality 2 at source
resolution, two decoder threads and one encoder thread. A partial final round may
have no new grid frame. Each incoming image is resized once using the processor's
bicubic kernel before entering the rolling history.

This cache uses a different time grid from the shared `interactionbench.frames`
extractor. The shared extractor and other baselines are unchanged. Rebuilding with
different ffmpeg versions can produce different pixels; the manifests record the
preprocessing identity.

### Multiple GPUs

One process per GPU is the default. This command starts eight independent workers,
then validates and merges their predictions; it never starts a judge:

```bash
python baselines/onestreamer/run.py \
    --devices 0,1,2,3,4,5,6,7 \
    --workers-per-gpu 1 \
    --data data/interactionbench \
    --mcq \
    --frames-root results/frame_cache/onestreamer_4fps \
    --cache-policy require \
    --out results/runs/onestreamer_native
```

`--workers-per-gpu` can be increased if memory permits. Video groups are assigned
deterministically using the number of scheduled model calls, without reference
answer information. Any worker failure stops the launcher with a nonzero exit.
Rerun the same command to resume completed items.

For external scheduling, use `--num-shards N --shard-index i` with the same
`--out` for all workers, then run:

```bash
python baselines/onestreamer/run.py --merge-only \
    --out results/runs/onestreamer_native
```

Use this merge mode for its completeness checks. The general `ibench merge`
keeps the first duplicate item and does not perform these checks.

## Output

`results/runs/onestreamer_native/`, or the directory given with `--out`:

- `run.json` and `system_prompt.txt`: public model identity, file hashes, protocol,
  selected inputs, shard assignment and complete prompt. Local model/data paths
  are not serialized.
- `shards/<rank>/items/<item_id>.json` and `raw/<item_id>.jsonl` beneath each shard:
  complete per-item records and every round's raw output, frame times and latency.
- `preds.jsonl` and `validation.json`: published after all selected items pass
  completeness checks, with one official-format prediction per item.

Completed items are reused only when the configuration, raw trace and prediction
agree. Changing weights, prompt, preprocessing, generation parameters or runner
code requires a new output directory. Corrupt or inconsistent records cause an
error. Execution failures never become successful silent predictions.

The parser accepts exactly one leading control tag. Silence and Standby must
have no body; Response must have a nonempty body. Invalid model output is retained
in the raw trace without an emission, retry or repair. All valid responses are kept,
including repeats, for the official scorer to match.

## Scoring

```bash
python -m interactionbench eval results/runs/onestreamer_native/preds.jsonl \
    --data data/interactionbench \
    --mcq-key \
    --judge hf:Qwen/Qwen3-14B \
    --judge-prompt v2 \
    --judge-cache results/judge_cache/onestreamer_qwen3_14b_hf_v2.jsonl \
    --delta 5 --pre-tol 1 \
    --out results/runs/onestreamer_native/eval
```

Use the answer key that belongs to the options file. Add the same `--items` list
when scoring a subset. Content gating is off by default; leave `--use-gate` unset.
Without `--judge`, free-form responses are scored lexically.

The earlier full reference predictions scored 36.703 with the official metrics and
a Qwen3-14B API judge (v2 prompt, non-thinking mode, temperature 0, 128 output tokens).
The command above uses the official local HF judge and may produce different
verdicts. It is not a claim that these scores are interchangeable or that this runner
was part of the original paper. In particular, VideoChat3's stored paper score uses
an earlier 218-item protocol. Comparisons require matching item sets, MCQ versions
and scoring settings.

## Defaults that affect results

| Argument or setting | Default | Meaning |
|---|---|---|
| Sampling / decision interval | 4 fps / 1 second | fixed quarter grid; final partial round retained |
| `--system-prompt` | `system_prompt.txt` next to the runner | full prompt is saved and hashed |
| `--max-rounds` | 32 | at most 128 frames with the default sampling |
| `--min-pixels` / `--max-pixels` | 3136 / 100352 | per-frame pixel budget; unchanged after Standby |
| `--max-new-tokens` | 128 | generation limit per round |
| `--a-window` | 10.0 | maximum video seconds after an A-type question |
| `--seed` | 42 | process random seed |
| `--attn-implementation` | `flash_attention_2` | alternatives are recorded as different configurations |

Weights use bfloat16 and generation is greedy (`do_sample=False`). Control tags
are kept during decoding; only model end-of-turn tokens are stripped.
