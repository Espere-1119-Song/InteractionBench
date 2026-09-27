# InteractionBench

Evaluation code for **InteractionBench: A Real-Time Interaction Benchmark for Streaming Video Systems**.

[Code (GitHub)](https://github.com/Espere-1119-Song/InteractionBench) ·
[Dataset (Hugging Face)](https://huggingface.co/datasets/InteractionBench/InteractionBench) ·
[Blog](https://www.enxinsong.com/blog/interactionbench/) ·
arXiv (to be added)

A real-time interaction system watches a video stream and decides, at every moment,
whether to speak or to stay silent. InteractionBench scores three things per item:
what the system said (Accuracy), when it said it (Timing Accuracy, TA), and whether it
stayed silent when no response was warranted (Silence Compliance, SC).

This repository contains

- the scorer and the judges (`ibench eval`),
- the protocols that drive turn-based models as real-time systems (`ibench run`),
- adapters for local Hugging Face models and for OpenAI-compatible APIs,
- runners for nine streaming video systems (`baselines/`) and for tool-using agents (`agents/`),
- the analysis scripts behind the supplementary tables (`analysis/`),
- the configuration of every run in the paper (`configs/paper_runs.json`).

Videos, questions and ground truth are on the Hugging Face Hub:
[InteractionBench/InteractionBench](https://huggingface.co/datasets/InteractionBench/InteractionBench).
See [docs/DATA.md](docs/DATA.md).

## Install

```bash
git clone https://github.com/Espere-1119-Song/InteractionBench.git
cd InteractionBench
pip install -e .            # scoring, API models, API judges
pip install -e ".[hf]"      # adds local Hugging Face models and judges
```

Frame extraction calls an `ffmpeg` binary on `PATH`. Python 3.10 or newer.

## Data

```bash
python scripts/prepare_data.py --data data/interactionbench download   # 812 videos, about 54 GB
python scripts/prepare_data.py --data data/interactionbench check
```

`download --no-videos` fetches the questions and the ground truth only, which is enough
for scoring existing predictions.

## Quick start

```bash
# 1. generate predictions: Qwen3-VL-8B, polled once per second with a sliding window
ibench run --model qwen3vl-8b --protocol sliding --mcq --out results/runs/qwen3vl-8b_sliding

# 2. score them with the paper protocol
ibench eval results/runs/qwen3vl-8b_sliding/preds.jsonl --mcq-key \
    --judge hf:Qwen/Qwen3-14B --out results/runs/qwen3vl-8b_sliding/eval
```

`ibench` is also available as `python -m interactionbench`. A run is resumable: start
the same command again and it continues after the last finished item. `--limit 10` runs
the first ten items.

## Evaluate your own system

| Your system is | Do this | Code needed |
|---|---|---|
| a Hugging Face chat VLM | `--model hf:<repo-or-path> --model-arg model_cls=... --model-arg image_style=payload` | none |
| a fine-tuned copy of a listed model | `--model qwen3vl-8b --model-path /path/to/checkpoint` | none |
| behind an OpenAI-compatible endpoint (vLLM, SGLang, a commercial API) | `--model api:<name> --api-base http://host:port/v1 --api-key-env MY_KEY` | none |
| anything callable from Python | subclass `ChatModel`, implement `chat`, load with `--plugin` | one class |
| natively streaming, an agent, or a human | write `preds.jsonl` yourself and score it | your runner |

Details and templates: [docs/ADD_A_MODEL.md](docs/ADD_A_MODEL.md),
[examples/custom_model.py](examples/custom_model.py),
[examples/write_predictions.py](examples/write_predictions.py).

API keys are read from environment variables only.

## Test methods

| `--protocol` | What the system sees at decision time t | Real-time |
|---|---|---|
| `sliding` | the most recent `--max-frames` frames up to t; every poll is a new conversation | yes |
| `cumulative` | all frames from 0 to t, subsampled to `--max-frames` | yes |
| `interleaved` | one growing conversation with the new frames and the system's own earlier replies | yes |
| `offline` | the whole video in one call; the system lists timed responses | no, reference only |

`--blind` removes all frames and measures what the text alone gives away.
`--interval`, `--sample-fps`, `--max-frames` and `--hint-set` cover the ablations of the
paper. New protocols are plugins: [docs/PROTOCOLS.md](docs/PROTOCOLS.md).

## Judges

Multiple-choice items are scored by option match. Free-form content is scored by a judge:

| `--judge` | Backend |
|---|---|
| `hf:<model_id>` | local Hugging Face model; the paper uses `hf:Qwen/Qwen3-14B` |
| `api:<model>` | OpenAI-compatible endpoint (`IBENCH_JUDGE_BASE_URL`, `IBENCH_JUDGE_API_KEY`) |
| `cache:<files>` | stored verdicts, no model |
| `ensemble:<a>,<b>,<c>` | majority vote over stored verdicts |
| `vdc:<judge>` | decomposition scoring on top of any judge |
| omitted | lexical scoring (token F1, numeric match) |

Adding a judge: [docs/ADD_A_JUDGE.md](docs/ADD_A_JUDGE.md).

## Reproduce the paper

```bash
python scripts/reproduce_paper.py list                      # every run and its command
python scripts/reproduce_paper.py eval --runs-root results/runs --judge hf:Qwen/Qwen3-14B
```

`eval` scores every run it finds and prints the difference from the reference scores.
See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the environments, the natively streaming
systems and the agents.

## Metric

Per item, on a 0 to 100 scale ([docs/METRICS.md](docs/METRICS.md) has the full definition):

- **Accuracy**: is the content right.
- **TA** = `100/N * sum_n max(0, 1 - d_n / Delta)`, where `d_n` is the delay of the response
  matched to event `n` and `Delta` = 5 s. An unanswered event contributes 0.
- **SC** = `100 * max(0, 1 - |V| / max(N, 1))`, where `V` counts premature, redundant and
  spurious responses.
- **Total** = `mean(Accuracy, harmonic_mean(TA, SC))`. Commentary items use Accuracy
  alone, items that require silence use SC alone.

A system that never speaks gets no SC credit on items that require a response, and a
system that always speaks gets SC 0, so neither policy scores well.

## Repository layout

```
interactionbench/   package: data, metrics, models, judges, protocols, run, evaluate, cli
baselines/          runners for natively streaming systems
agents/             runners for tool-using agents
analysis/           scripts behind the supplementary tables and figures
benchmark/splits/   item lists: all1060, subset103, mcq688, frozen218
configs/            paper_runs.json: every run of the paper with its command and reference scores
scripts/            reproduce_paper.py, prepare_data.py, run_with_watchdog.sh, slurm_example.sbatch
examples/           plugin templates for models, protocols and judges
tests/              unit and end-to-end tests (no GPU, no data needed): pytest
docs/               DATA, METRICS, PROTOCOLS, ADD_A_MODEL, ADD_A_JUDGE, REPRODUCE
```

## Citation

The citation entry will be added when the paper is released.
