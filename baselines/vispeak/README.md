# ViSpeak-s3 (native proactive streaming)

`run.py` calls `model.streaming_generate` of ViSpeak. The model scores every video
segment with its informative head and starts to speak when a sliding window of three
segments exceeds 0.35. One call returns one response and the video time of the segment
that triggered it. For items with several events the runner calls the model again,
starting at the segment after the last trigger. Earlier responses are not fed back, and
at most 24 responses are collected per item.

- The question is appended to the system prompt of ViSpeak as
  `"\nUser request (answer at the right moment): <question>"`.
- B and C items: inference starts at the first segment, with `proactive=True`.
- A items: inference starts at the first segment at or after `question_time_s`, with
  `proactive=False` (the model must answer). Segments up to
  `question_time_s + --a-window` are used.
- Frame sampling is fixed by the model: at most `MAX_IMAGE_LENGTH * pooling_size^2`
  segments (64 for ViSpeak-s3), spread uniformly over the video. The input frame rate
  depends on the video length and cannot be set.
- The audio input is a zero tensor.

## Upstream

| | |
|---|---|
| Code repository | https://github.com/HumanMLLM/ViSpeak |
| Checkpoint | `fushh7/ViSpeak-s3` (https://huggingface.co/fushh7/ViSpeak-s3) |
| Further weights | audio encoder from `VITA-MLLM/VITA-1.5`, visual encoder `OpenGVLab/InternViT-300M-448px` |
| Checkout needed | Yes. The runner imports the `vispeak` package. |
| Upstream commit | `b6755dbd09b7eb8de03f305e10617b9018e6ca48` (origin `https://github.com/HumanMLLM/ViSpeak`) |
| Local modifications of the checkout | None (`git status --short` is empty). No patch file is needed. |
| Checkpoint revision of the paper run | Not recorded. |

The commit above is the one of the checkout that was available when this runner was
ported. The launcher of the paper run pointed to a checkout in another location, which
was not available. Whether both checkouts are at the same commit is not determined.

Locations:

| Argument | Environment variable | Default |
|---|---|---|
| `--vispeak-repo` | `VISPEAK_REPO` | `external/ViSpeak` |
| `--model` | `VISPEAK_MODEL` | `fushh7/ViSpeak-s3` |

```bash
git clone https://github.com/HumanMLLM/ViSpeak external/ViSpeak
git -C external/ViSpeak checkout b6755dbd09b7eb8de03f305e10617b9018e6ca48
huggingface-cli download fushh7/ViSpeak-s3 --local-dir checkpoints/ViSpeak-s3
huggingface-cli download VITA-MLLM/VITA-1.5 --local-dir checkpoints/VITA-1.5
huggingface-cli download OpenGVLab/InternViT-300M-448px --local-dir checkpoints/InternViT-300M-448px
```

Then edit `checkpoints/ViSpeak-s3/config.json` so that the paths of the audio encoder
and of the visual encoder point to the downloaded directories, as the upstream README
requires. The paper run used such a local copy. Pass it with `--model`. Keep the
directory name `ViSpeak-s3`: the loader of ViSpeak derives the model name from the last
path component.

## Environment

Pins recorded for the paper run: python 3.10, transformers 4.44.2,
xformers 0.0.27.post2, torchaudio 2.4, six. The upstream README lists
`pytorch==2.4.0+cu121`, `transformers==4.44.2` and `numpy==1.23.5`. The environment of
the paper run was not preserved, so no further pins are recorded.

```bash
python3.10 -m venv external/vispeak-venv
external/vispeak-venv/bin/pip install -e .
external/vispeak-venv/bin/pip install "torch==2.4.0" "torchaudio==2.4.0" \
    --index-url https://download.pytorch.org/whl/cu121
external/vispeak-venv/bin/pip install "transformers==4.44.2" "xformers==0.0.27.post2" \
    "numpy==1.23.5" six decord accelerate
```

Install the remaining packages listed in the upstream README.

- decord cannot read AV1 video. `--video-dir` defaults to
  `data/interactionbench/videos_h264`, one flat directory with H.264 copies named
  `<video_id>.mp4`.
- If decord reports end-of-file errors, set `DECORD_EOF_RETRY_MAX=20480`.

## Paper run

| | |
|---|---|
| Run directory | `vispeak-s3_streaming_native_mcq` |
| Item set | 1,060-item set (all items) |
| Multiple-choice file | v4, `mcq/mcq_options_v4.jsonl` |
| Predictions stored | 1,060 lines |

The paper run of this system used the 1,060-item set with the v4 multiple-choice file.
What was checked:

- The stored `config.json` of the run has `"n_items_targeted": 1060`,
  `"a_window_s": 10.0` and `"max_triggers": 24`. Its `"mcq_options"` entry names
  `mcq/mcq_options_v4.jsonl`, and its `"video_dir"` entry names the `videos_h264`
  directory.
- The stored `preds.jsonl` has 1,060 lines with 1,060 different item ids.
- The stored `eval/summary.json` reports 1,060 items. It has no `config` block, so the
  evaluator settings of the stored scores are not determined.

Command of the paper run:

```bash
python baselines/vispeak/run.py \
    --vispeak-repo external/ViSpeak \
    --model checkpoints/ViSpeak-s3 \
    --mcq
```

`--mcq` without a value selects `data/interactionbench/mcq/mcq_options_v4.jsonl`.

## Output

`results/runs/vispeak-s3_streaming_native[_mcq]/` or the directory given with `--out`:
`preds.jsonl` (one line per item, appended, items already present are skipped),
`raw/<item_id>.json` (every call with its trigger time and the first 120 characters of
the text) and `config.json`.

## Scoring

```bash
python -m interactionbench eval results/runs/vispeak-s3_streaming_native_mcq/preds.jsonl \
    --data data/interactionbench \
    --mcq-key data/interactionbench/mcq/mcq_key_v4.jsonl \
    --out results/runs/vispeak-s3_streaming_native_mcq/eval
```

- Add `--judge <spec>` to grade free-form content with a model. Without a judge,
  free-form content is scored lexically.
- Add `--items benchmark/splits/frozen218.txt` to score the 218-item subset.
- Items without a prediction line are scored as silent. Add `--skip-missing` to leave
  them out.

## Behaviour to be aware of

- An item is skipped without a prediction line in two cases: the number of image
  patches is not a multiple of the number of sampled frames, or an exception occurs
  while the item is processed. A later start of the runner tries these items again.
- Generation uses `temperature=0.01`, `max_new_tokens=256` and `padding_size=128`.
- A leading character `☞`, `☜` or `☟` is removed from the response.
