# Flash-VStream-Qwen-7b (streaming encoder, polled once per second)

Flash-VStream answers questions about a stream and has no mechanism to decide when to
speak. `run.py` therefore applies the polling protocol. The flash memory of the model
ingests the stream at `--fps` frames per second (`embed_new_video_clip`). Every
`--interval` seconds the model is asked, on the current memory, whether to speak or to
wait. The prompt has its own wording (constants `SYSTEM` and `FORMAT` in `run.py`); it
is not the prompt of `ibench run`.

- B and C items: one poll per interval from the first interval to the end of the video.
- A items: no poll before `question_time_s`. During that time a query with the text
  `"Keep watching. Reply WAIT."` runs whenever the integer part of the stream time is a
  multiple of 30; its output is discarded. It keeps the memory of the model bounded.
  Polls with the question run from `question_time_s` to `question_time_s + --a-window`.
- A clip with an odd number of frames gets its last frame twice.

Files:

| File | Content |
|---|---|
| `run.py` | the runner; parses the replies with the strict format `DECISION:` / `RESPONSE:` |
| `reparse.py` | builds the predictions again from the raw replies with a lenient parser |

The model does not follow the requested format. In the paper run the strict parser
matched no reply, so `run.py` alone yields empty emissions for every item. The paper
numbers use the output of `reparse.py`.

## Upstream

| | |
|---|---|
| Code repository | https://github.com/IVGSZ/Flash-VStream (sub-directory `Flash-VStream-Qwen`) |
| Checkpoint | `zhang9302002/Flash-VStream-Qwen-7b` (https://huggingface.co/zhang9302002/Flash-VStream-Qwen-7b) |
| Checkout needed | Yes. The runner imports the package `models` of `Flash-VStream-Qwen`. |
| Upstream commit | `8f6bde2f397f4846505df4d03364d97617fc02ee` (origin `https://github.com/IVGSZ/Flash-VStream`) |
| Local modifications of the checkout | None. `git status --short` lists only an untracked `__pycache__` directory. No patch file is needed. |
| Checkpoint revision of the paper run | Not recorded. Revision `063585f19888dae72c17eb49beb15ab0b77523b9` was in the local cache when this runner was ported. |

Locations:

| Argument | Environment variable | Default |
|---|---|---|
| `--repo` | `FVSTREAM_REPO` | `external/Flash-VStream/Flash-VStream-Qwen` |
| `--checkpoint` | `FVSTREAM_CKPT` | the newest directory under `$HF_HOME/hub/models--zhang9302002--Flash-VStream-Qwen-7b/snapshots/`; `HF_HOME` defaults to `~/.cache/huggingface` |

```bash
git clone https://github.com/IVGSZ/Flash-VStream external/Flash-VStream
git -C external/Flash-VStream checkout 8f6bde2f397f4846505df4d03364d97617fc02ee
huggingface-cli download zhang9302002/Flash-VStream-Qwen-7b
```

`--checkpoint` takes a local directory. The runner passes a directory to
`from_pretrained`, not a repository id.

## Environment

Versions of the environment of the paper run (Python 3.10):

| Package | Version |
|---|---|
| torch | 2.7.1 (CUDA 12.8 build) |
| torchvision | 0.22.1 |
| transformers | 4.45.0 |
| flash-attn | 2.8.3, built from source |
| decord | 0.6.0 |
| accelerate | 1.14.0 |
| peft | 0.20.0 |
| opencv-python-headless | 5.0.0.93 |
| pillow | 12.3.0 |

```bash
python3.10 -m venv external/fvstream-venv
external/fvstream-venv/bin/pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
external/fvstream-venv/bin/pip install transformers==4.45.0 opencv-python-headless accelerate \
    decord pillow openai peft pandas ninja packaging psutil setuptools wheel huggingface_hub
MAX_JOBS=32 FLASH_ATTENTION_FORCE_BUILD=TRUE \
    external/fvstream-venv/bin/pip install flash-attn==2.8.3 --no-build-isolation
external/fvstream-venv/bin/pip install -e .
```

- `flash-attn` is required. The runner loads the model with
  `attn_implementation="flash_attention_2"`. With `sdpa` the attention needs several
  hundred GB of memory.
- The build of the paper environment set `TORCH_CUDA_ARCH_LIST="10.0"` for its GPU. Set
  the value that matches your GPU, or install a prebuilt wheel of flash-attn.
- The pins recorded for the first part of the paper run (45 items, other hardware) are
  torch 2.6, transformers 4.45 and the prebuilt wheel flash-attn 2.7.4.post1.
- `decord` is imported after the model is on the GPU. Importing it before the CUDA
  initialisation of torch can end the process with a segmentation fault.
- decord cannot read AV1 video. `--video-dir` defaults to
  `data/interactionbench/videos_h264`, one flat directory with H.264 copies named
  `<video_id>.mp4`.
- If decord reports end-of-file errors, set `DECORD_EOF_RETRY_MAX=20480`.

## Paper run

| | |
|---|---|
| Run directory | `fvstream-7b_polling_iv1_8fps_mcq_lenientparse` (re-parsed from `fvstream-7b_polling_iv1_8fps_mcq`) |
| Item set | 1,060-item set (all items) |
| Multiple-choice file | v4, `mcq/mcq_options_v4.jsonl` |
| Predictions stored | 1,060 lines |

Step 1, generation. The runner exits with code 17 after a GPU out-of-memory error,
because the memory is not released inside the process. Start the same command again
until it exits with code 0:

```bash
until python baselines/flash_vstream/run.py --mcq --fps 8 --interval 1.0; do
    [ $? -eq 17 ] || break
done
```

`--mcq` without a value selects `data/interactionbench/mcq/mcq_options_v4.jsonl`. The
default output directory is `results/runs/fvstream-7b_polling_iv1_8fps_mcq`.

The paper run was split into six item lists. Each list ran with `--items` and its own
`--out`, on one GPU with 180 GB of memory per process. The prediction files were merged
into the run directory; the raw dumps stayed in the shard directories:

```bash
python baselines/flash_vstream/run.py --mcq --fps 8 --interval 1.0 \
    --items shard_0.txt --out results/runs/fvstream-7b_polling_iv1_8fps_mcq_shard0
python -m interactionbench merge results/runs/fvstream-7b_polling_iv1_8fps_mcq/preds.jsonl \
    results/runs/fvstream-7b_polling_iv1_8fps_mcq_shard*/preds.jsonl
```

Step 2, lenient re-parse:

```bash
python baselines/flash_vstream/reparse.py \
    --src results/runs/fvstream-7b_polling_iv1_8fps_mcq
```

`reparse.py` reads `<src>/preds.jsonl` and the raw dumps in `<src>_*shard*/raw/` and
`<src>/raw/`, and writes `<src>_lenientparse/preds.jsonl`.

Rule of the lenient parser for one reply (the first match applies):

| Reply | Result |
|---|---|
| contains `DECISION: WAIT` | silent |
| contains `DECISION: SPEAK` | emission: first line after `RESPONSE:` |
| starts with `WAIT` | silent |
| starts with `SPEAK` (any letter case, also with further letters attached) | emission: first line of the text after that word |
| is one option letter `A` to `E`, optionally followed by `.` or `)` | emission: the reply |
| is empty | silent |
| anything else | emission: first line of the reply |

An emission with empty content is not recorded. Counts in the raw dumps of the paper
run: 27,167 replies of the form `SPEAk` followed by content on the next line, 25,764
replies with a bare option letter, 2,678 replies `WAIT`.

What was checked:

- `reparse.py`, applied to the stored predictions and raw dumps of the paper run,
  gives the stored `fvstream-7b_polling_iv1_8fps_mcq_lenientparse/preds.jsonl` in every
  field except the value of the field `reparse` (a label of the parser version).
- The stored predictions have 1,060 lines: 701 re-parsed lines, 314 lines with the
  field `"error": "oom_x2_on_180GB"` (the item ran out of memory in two processes;
  empty emissions), and 45 lines without a raw dump.
- The 45 lines without a raw dump were produced on other hardware (GPU with 46 GB of
  memory) and their raw dumps were not available. These lines keep the result of the
  strict parser, which is empty emissions.
- The `config.json` files of the six shard directories state `fps: 8`,
  `interval_s: 1.0` and the v4 multiple-choice file.

## Output

`results/runs/fvstream-7b_polling_iv<interval>_<fps>fps[_mcq]/` or the directory given
with `--out`: `preds.jsonl` (one line per item, appended, items already present are
skipped), `raw/<item_id>.json` (every poll with the first 120 characters of the reply),
`config.json`, `oom_restarts.json` (number of out-of-memory failures per item) and
`current_item.txt` (the item in progress; present only while an item runs or after a
process was killed).

## Scoring

```bash
python -m interactionbench eval \
    results/runs/fvstream-7b_polling_iv1_8fps_mcq_lenientparse/preds.jsonl \
    --data data/interactionbench \
    --mcq-key data/interactionbench/mcq/mcq_key_v4.jsonl \
    --judge hf:Qwen/Qwen3-14B \
    --out results/runs/fvstream-7b_polling_iv1_8fps_mcq_lenientparse/eval
```

- Without `--judge`, free-form content is scored lexically.
- Items without a prediction line are scored as silent. Add `--skip-missing` to leave
  them out.

## Defaults that affect results

| Setting | Default | Paper run |
|---|---|---|
| `--fps` | 8 | 8 |
| `--interval` | 1.0 s | 1.0 s |
| `--a-window` | 10.0 s | 10.0 s |
| `--max-new-tokens` | 48 | 48 |
| decoding | `do_sample=False` | the same |
| weights | bfloat16 | the same |
| time limit per item | 5,400 s, then the item counts as an out-of-memory failure | the same |
| out-of-memory failures before an item is recorded as failed | 2 | 2 |

## Behaviour to be aware of

- An item that ran out of memory in two processes is recorded with empty emissions and
  the field `error`. The text of that field names 180 GB for every GPU size.
- A process that is killed while an item runs leaves `current_item.txt`. The next start
  counts this as one out-of-memory failure of that item.
- An exception that is not an out-of-memory error prints `ERROR` and writes no
  prediction line. A later start tries the item again. There is no limit for these
  attempts.
- The raw dump keeps the first 120 characters of a reply. `reparse.py` works on this
  text.
- `run.py` has no `--overwrite` option. Delete the output directory to start again.
- The memory need depends on the length of the video. The longest items of the
  benchmark did not fit into 180 GB.
