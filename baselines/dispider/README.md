# Dispider (offline temporal grounding)

`run.py` evaluates Dispider on the offline track. The inference interface used here,
`videoStream.Run(video, prompt)`, takes the whole video and returns one string. It does
not report the time at which the decision module would respond. The runner therefore
asks for the time in text and uses the stated seconds as emission times. The scores are
comparable to other offline systems. They do not measure the proactive decision module
described in the Dispider paper.

- A items: one call with the question. The answer is one emission at `question_time_s`.
- B and C items: two calls. The first call asks the question and gives the content. The
  second call appends `"At what time in seconds does this happen? If it happens several
  times, state every time in seconds."` and gives the times. Each stated time inside
  the video becomes one emission with the content of the first call.
- Times are parsed from digits and from spelled-out numbers. A range such as
  `57 - 60 seconds` counts as one event at its start.

## Upstream

| | |
|---|---|
| Code repository | https://github.com/Mark12Ding/Dispider |
| Checkpoint | `Mar2Ding/Dispider` (https://huggingface.co/Mar2Ding/Dispider) |
| Checkout needed | Yes. The runner imports `inference.py` (class `videoStream`). |
| Upstream commit of the paper run | Not determined. The checkout used for the paper run was not available when this runner was ported. |
| Local modifications of the checkout | Not determined, for the same reason. No patch file is provided. |
| Checkpoint revision of the paper run | Not recorded. Revision `7c5517eaaa4dc3788602af40e077063792c75f6c` was in the local cache when this runner was ported. |

Locations:

| Argument | Environment variable | Default |
|---|---|---|
| `--dispider-repo` | `DISPIDER_REPO` | `external/Dispider` |
| `--model` | `DISPIDER_MODEL` | `Mar2Ding/Dispider` |

```bash
git clone https://github.com/Mark12Ding/Dispider external/Dispider
huggingface-cli download Mar2Ding/Dispider --local-dir checkpoints/Dispider
```

The paper run loaded a local copy of the checkpoint. In that copy `config.json` was
edited so that the compressor path and the CLIP vision tower pointed to local
directories. The current model card states that the repository id can be loaded
directly. This was not tested with this runner. If loading by repository id fails, pass
the local directory with `--model`.

## Environment

Pins recorded for the paper run: python 3.10, torch 2.2 (cu118), flash-attn 2.5.9
(binary wheel for torch 2.2, cu118, cp310), numpy below 2. The transformers version was
not recorded, and the environment was not preserved. The model card names PyTorch 2.2.0,
FlashAttention 2.5.9.post1 and Transformers 4.41.2.

```bash
python3.10 -m venv external/dispider-venv
external/dispider-venv/bin/pip install -e .
external/dispider-venv/bin/pip install "torch==2.2.*" --index-url https://download.pytorch.org/whl/cu118
external/dispider-venv/bin/pip install "transformers==4.41.2" "numpy<2" decord accelerate
external/dispider-venv/bin/pip install "flash-attn==2.5.9.post1"
```

Install the remaining requirements of the upstream repository as its README describes.

- decord cannot read AV1 video. `--video-dir` defaults to
  `data/interactionbench/videos_h264`, one flat directory with H.264 copies named
  `<video_id>.mp4`.
- If decord reports end-of-file errors, set `DECORD_EOF_RETRY_MAX=20480`.

## Paper run

| | |
|---|---|
| Run directory | `dispider_offline_mcq` |
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
python baselines/dispider/run.py \
    --model checkpoints/Dispider \
    --items benchmark/splits/frozen218.txt \
    --mcq data/interactionbench/mcq/mcq_options.jsonl
```

In this repository `--mcq` without a value selects `mcq/mcq_options_v4.jsonl`. The earlier multiple-choice file was not
available when this runner was ported, so the stored predictions could not be
regenerated for comparison.

Command for the 1,060-item set with the v4 file (no paper numbers exist for it):

```bash
python baselines/dispider/run.py \
    --model checkpoints/Dispider \
    --mcq data/interactionbench/mcq/mcq_options_v4.jsonl \
    --out results/runs/dispider_offline_mcqv4_full
```

## Output

`results/runs/dispider_offline[_mcq]/` or the directory given with `--out`:
`preds.jsonl` (one line per item, appended, items already present are skipped) and
`raw/<item_id>.json` (both model outputs). `n_polls` is 1 for every item.
`poll_latencies` holds the summed time of the calls for that item.

## Scoring

```bash
python -m interactionbench eval results/runs/dispider_offline_mcq/preds.jsonl \
    --data data/interactionbench \
    --mcq-key data/interactionbench/mcq/mcq_key_v4.jsonl \
    --items benchmark/splits/frozen218.txt \
    --out results/runs/dispider_offline_mcq/eval
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

## Behaviour to be aware of

- When `videoStream.Run` raises an exception, the runner prints the error and treats
  the output as empty. The item is written with no emissions.
- Stated times outside `[0, duration_s]` are dropped. Equal times are merged.
