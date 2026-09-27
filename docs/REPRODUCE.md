# Reproduce the paper

`configs/paper_runs.json` lists every run: the system, the setting, the command, the
item list and the reference scores.

```bash
python scripts/reproduce_paper.py list
```

## 1. Turn-based models under polling and offline grounding

```bash
ibench run --model qwen3vl-8b --protocol sliding     --mcq --out results/runs/qwen3vl-8b_sliding_iv1_mcqv4_full
ibench run --model qwen3vl-8b --protocol interleaved --mcq --out results/runs/qwen3vl-8b_interleaved_iv1_mcqv4_full
ibench run --model qwen3vl-8b --protocol offline     --mcq --out results/runs/qwen3vl-8b_sliding_offline_iv1_mcqv4_full
ibench run --model qwen3vl-8b --protocol sliding --blind --mcq --out results/runs/qwen3vl-8b_blind_iv1_mcqv4_full
```

Ablations on Qwen3-VL-8B: `--max-frames 64`, `--max-frames 256`, `--sample-fps 0.5`,
`--sample-fps 1`, `--sample-fps 4`, and `--hint-set v2` or `v3` on
`--items benchmark/splits/subset103.txt`.

Parallel execution: `--num-shards N --shard-index i`, then `ibench merge`.
`scripts/slurm_example.sbatch` shows a job array.

Environment of the paper runs: one NVIDIA B200 GPU per run, bfloat16 weights, greedy
decoding. MiniCPM-o-4.5 needs its own environment with transformers 4.51; all other
listed models ran in one environment.

An item that does not fit the GPU memory is recorded as silent with a `failed` field in
`preds.jsonl` and the run continues. `--no-skip-oom` stops instead.

## 2. API models

```bash
export ANTHROPIC_API_KEY=...
ibench run --model claude-api --protocol sliding --mcq \
    --items benchmark/splits/subset103.txt --out results/runs/claude-bare_sliding_sub103
export GOOGLE_API_KEY=...
ibench run --model gemini-api --model-arg model=gemini-2.5-flash --protocol sliding --mcq \
    --items benchmark/splits/subset103.txt --out results/runs/gemini-bare_sliding_sub103
```

API models change over time, so later runs can differ from the reference scores.

## 3. Natively streaming systems

Each system has its own runner, environment and instructions in `baselines/<system>/`.
The runners write the same `preds.jsonl`.

## 4. Agents

See `agents/README.md`.

## 5. Scoring

```bash
python scripts/reproduce_paper.py eval --runs-root results/runs \
    --judge hf:Qwen/Qwen3-14B --judge-cache results/judge_cache/qwen3-14b_v2.jsonl \
    --write-eval eval_paper --out results/paper_scores.csv
```

This applies the paper protocol to every run found under `--runs-root`, prints the
scores next to the reference scores and marks runs that differ by more than
`--tolerance`.

## What to expect

| Step | Expected agreement with the reference |
|---|---|
| Scoring given predictions and stored judge verdicts | exact |
| Scoring given predictions, judge run again | the same verdicts on the same software and GPU type |
| Generating predictions with an open-weight model | most items identical; on a different GPU type or library version a small share of decisions differs, because greedy decoding in bfloat16 is sensitive to numerical differences |
| Generating predictions with an API model | depends on the provider's current model |

## Supplementary analyses

`analysis/README.md` lists the script behind each supplementary table and figure.
