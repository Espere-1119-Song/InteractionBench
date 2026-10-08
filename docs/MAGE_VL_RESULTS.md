# Mage-VL-4B on InteractionBench

![Mage-VL-4B InteractionBench results](assets/mage_vl_interactionbench.svg)

Mage-VL-4B was evaluated on all 1,060 InteractionBench items with the default
turn-based polling protocol: a one-second polling interval, a two-FPS frame sampler,
and a 16-frame sliding window. Free-form answers were scored by Qwen3-14B using the
v2 judge prompt; multiple-choice answers were option-scored.

| Evaluation | Items | Total | Accuracy | Timing Accuracy | Silence Compliance |
|---|---:|---:|---:|---:|---:|
| Full benchmark | 1,060 | **33.668** | 46.856 | 65.454 | 21.034 |
| MCQ subset | 688 | **34.903** | 57.916 | 73.912 | 15.364 |

The full run produced 1,060 unique prediction rows with zero failures across
119,412 polling decisions. Under the identical default sliding-window configuration,
Mage-VL-4B ranks third among the models currently listed in `paper_runs.json`, behind
Qwen3-VL-8B (34.8) and LLaVA-OV2-8B (34.5).

## Reproduce

The complete run needs roughly 54 GB for benchmark videos, 9.5 GB for Mage-VL, and
29.5 GB for the optional Qwen3-14B judge. The reported run used eight NVIDIA H800
GPUs and took approximately nine active hours. Fewer GPUs work because each shard is
resumable, but take proportionally longer.

Create a Python 3.12 environment and install the known-compatible versions:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[data]'
pip install 'torch==2.9.1' 'torchvision==0.24.1' \
  'transformers==5.7.0' accelerate opencv-python decord imageio-ffmpeg
pip install --no-build-isolation 'mamba-ssm==2.3.2.post1'
```

InteractionBench videos include AV1. Confirm that `ffmpeg -decoders` lists AV1. If
the system FFmpeg lacks it, expose the decoder-enabled binary installed above:

```bash
ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" \
  .venv/bin/ffmpeg
export PATH="$PWD/.venv/bin:$PATH"
```

Download and verify the benchmark, then launch the resumable runner:

```bash
python scripts/prepare_data.py --data data/interactionbench download
python scripts/prepare_data.py --data data/interactionbench check
GPUS=0,1,2,3,4,5,6,7 bash scripts/reproduce_mage_vl.sh
```

The script runs one shard per listed GPU, merges exactly one prediction per item,
checks for duplicates and failures, scores the 688 MCQ items mechanically, and runs
the paper's Qwen3-14B v2 judge over all 1,060 items. Re-running the same command
resumes completed shard items. Set `RUN_JUDGE=0` to skip the 29.5 GB judge model.

Expected final files:

```text
results/runs/mage-vl-sliding-all1060/
  preds.jsonl
  eval-mcq/summary.json
  eval-qwen3-14b/summary.json
  logs/{0..7}.log
  shards/{0..7}/preds.jsonl
```

## Scope

This is an apples-to-apples InteractionBench evaluation through its generic
frame-sampled, multi-image VLM adapter. It does not measure Mage-VL's native codec
processor or visual-only proactive gate; those require a separate native-streaming
runner and should be reported as a distinct protocol.
