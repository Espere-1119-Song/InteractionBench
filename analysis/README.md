# Analysis scripts

Scripts that produce the supplementary tables and figures of the paper from prediction
files and per-item evaluation records. None of them needs a GPU, except where a judge
model or an oracle run is requested explicitly.

## Layout and common arguments

```
<runs_root>/<run_name>/preds.jsonl                                  predictions of one run
<runs_root>/<run_name>/<eval_name>/{summary.json,records.jsonl}     one evaluation of the run
```

An evaluation directory is written by `python -m interactionbench eval PREDS --out DIR`
or by `python scripts/reproduce_paper.py eval --write-eval NAME`.

Every script accepts the first four arguments:

| argument | default | meaning |
|---|---|---|
| `--runs-root` | `results/runs` | directory with one sub-directory per run |
| `--data` | `data/interactionbench` | benchmark root |
| `--mcq-key` | `<data>/mcq/mcq_key_v4.jsonl` | multiple-choice answer key |
| `--out` | `results/analysis` | output directory |
| `--eval-name` | `eval_nojudge` or `eval_judge` | evaluation directory name, where a script reads or writes one |
| `--runs` | the list used for the paper | only where a script has a run list: run names, or a file with one name per line; entries may be `label=run_name` where a script uses display labels |

Scripts that score predictions call the evaluator as a library
(`interactionbench.evaluate.evaluate`). `--suffix` is appended to output file names and
to the default evaluation directory name; the paper numbers use a pre-anchor tolerance
of 1 s and the suffix `_pretol1`.

Run the commands from the repository root. The commands below are the ones used for
the paper numbers.

## scripted_policies.py

Purpose: predictions of seven scripted policies that never look at the video
(always-fire, never-fire, periodic, chatter, burst repeater, parrot, anticipatory),
and their scores. Used to audit whether a timing or silence score can be raised
without understanding the stream.

Inputs: benchmark annotations, answer key.

```
python analysis/scripted_policies.py --pre-tol 1.0 --suffix _pretol1
python analysis/scripted_policies.py --pre-tol 1.0 --suffix _pretol1 \
    --policies anticipatory --summary-name scripted_anticipatory
```

Outputs: `<runs_root>/scripted_<policy>/preds.jsonl`,
`<runs_root>/scripted_<policy>/eval_nojudge_pretol1/`,
`<out>/scripted_policies_pretol1.json`, `<out>/scripted_anticipatory_pretol1.json`.

Paper: `scripted_policies_pretol1.json` is the data of the figure with label
`fig:audit` (six policies). For the anticipatory policy: not stated in the source.

## violation_anatomy.py

Purpose: (1) per run, the rate of premature, redundant and spurious emissions per
reference event, the false-alarm rate on negative items, mean timing accuracy and
silence compliance on positive items, latency figures, and the hit rate by item
position; (2) LaTeX rows of the per-task table.

Inputs: `<runs_root>/<run>/<eval_name>/records.jsonl` and `summary.json`. A missing
evaluation is created from `<runs_root>/<run>/preds.jsonl` without a judge.

```
python analysis/violation_anatomy.py --pre-tol 1.0 --suffix _pretol1
```

Outputs: `<out>/violation_anatomy_pretol1.json`, `<out>/fullcap_rows_pretol1.tex`.

Paper: `violation_anatomy_pretol1.json` is one of the two data files of the figure
with label `fig:silence` (fields `prem`, `redun`, `neg_fa`). `fullcap_rows_pretol1.tex`
has the row format of the per-task table with label `tab:fullcap`; the source does not
state that the table of the paper was built from this file.

## sensitivity_and_thinning.py

Purpose: (1) total score of every run under other values of the delay bound Delta
(2, 3, 8, 10 s) and of the content-gate threshold (0.2, 0.4, 0.5); (2) timing accuracy
and silence compliance after random deletion of emissions at keep rates 1.0, 0.8, 0.6,
0.4, 0.2 and 0.1.

Inputs: `<runs_root>/<run>/preds.jsonl`. Presets: `paper16` (16 runs, curves for four
of them) and `all` (22 runs, curves for all).

```
python analysis/sensitivity_and_thinning.py --preset paper16 --pre-tol 1.0 \
    --suffix _pretol1 --jobs 8
python analysis/sensitivity_and_thinning.py --preset all --pre-tol 1.0 \
    --suffix _pretol1_all --jobs 8
```

Outputs: `<out>/sensitivity<suffix>.json`, `<out>/thinning_curves<suffix>.json`, and
the thinned prediction files in `<out>/sweep_work<suffix>/`.

Paper: `thinning_curves_pretol1.json` is the random-deletion data of the figure with
label `fig:tasc`. `sensitivity*.json`: not stated in the source.

Note: the content gate is disabled in the paper protocol, so the three gate settings
return the total of the default setting.

## pairsel.py

Purpose: matched-suite selectivity (PairSel). A suite passes when its positive item
earns timing credit and the run emits nothing near any near-miss event of the suite.

Inputs: the suite file (`--suites`, default `<data>/suites/nearmiss_verified.jsonl`,
schema in the docstring of the script), `preds.jsonl` and judged records of every run
under `--runs-root` (several roots may be given).

```
python analysis/pairsel.py score --eval-name eval_judge_pretol1 \
    --fallback-eval-name eval_nojudge_pretol1 --suffix _pretol1
python analysis/pairsel.py strict --pairsel results/analysis/pairsel_pretol1.json
```

Outputs: `<out>/pairsel_pretol1.json`, `<out>/pairsel_strict.json`.

Paper: `pairsel_pretol1.json` is one of the two data files of the figure with label
`fig:silence` (field `pairsel` of each run). `pairsel_strict.json`: not stated in the
source.

## bootstrap_ci.py

Purpose: item-level bootstrap 95% confidence interval of the mean total score of each
run (2000 resamples, one shared random stream, seed 20260923). The intervals depend
on the order of the run list.

Inputs: `<runs_root>/<run>/<eval_name>/records.jsonl`; `--runs` is required (names, a
text file, or a JSON list of `{"run", "dir"}`).

```
python analysis/bootstrap_ci.py --runs runs.txt --eval-name eval_judge_pretol1
```

Output: `<out>/ci_overall.csv`.

Paper: the 95% interval column of the table with label `tab:system_scores`.

## rank_stability.py

Purpose: Kendall tau between the system ranking at Delta = 5 s and the rankings at
Delta = 2, 3, 8 and 10 s, largest change of rank, and whether the three best systems
stay the same.

Inputs: `<sweep_dir>/<run>_d<Delta>/summary.json` for the five values of Delta. The
evaluations of the runs named in `--runs` are created when they are missing.

```
python analysis/rank_stability.py --runs full_runs.txt --pre-tol 1.0 \
    --judge ensemble:qwen_qwen3-14b
python analysis/rank_stability.py --runs subset_runs.txt --pre-tol 1.0 \
    --judge ensemble:qwen_qwen3-14b --items benchmark/splits/subset103.txt
```

`full_runs.txt` lists the runs scored on all items and `subset_runs.txt` the runs
scored on the 103-item subset. `ensemble:<slug>` reads stored verdicts from
`<judge_cache_dir>/<slug>_v2*.jsonl`.

Outputs: `<out>/delta_sweep/<run>_d<Delta>/`, `<out>/kendall.csv`, `<out>/per_run.csv`.

Paper: `kendall.csv` is the table with label `tab:delta`; `per_run.csv` is the data
of the figure with label `fig:delta`.

## judge_calibration.py

Purpose: calibration of candidate judges against stored reference verdicts (binary
verdicts of an earlier Gemini judge; field `gemini`).

| sub-command | purpose | needs |
|---|---|---|
| `extract` | rebuild the (question, reference, prediction) triples of two runs | verdict files of the reference judge in `<run>/eval/` |
| `run` | score the labelled triples with candidate judges; agreement, kappa, TPR, TNR | a judge (GPU or API), or stored verdicts through `cache:<file>` |
| `votes` | one greedy and K = 5 sampled verdicts per triple (temperature 0.7, top-p 0.95) | GPU |
| `analyze` | majority-vote table, kappa between judges, ensembles | vote files |
| `tables` | print the per-task rows of all judged evaluations | judged evaluations |

```
python analysis/judge_calibration.py extract
python analysis/judge_calibration.py run hf:Qwen/Qwen3-14B --judge-prompt v2
python analysis/judge_calibration.py votes hf:Qwen/Qwen3-14B --judge-prompt v2
python analysis/judge_calibration.py analyze
python analysis/judge_calibration.py tables --eval-name eval_judge
```

Outputs: in `<out>/judge_calib/`: `triples.jsonl`, `cache_<slug>*.jsonl`,
`report_<slug>.json`, `disagree_<slug>.jsonl`, `votes_<slug>_prompt<version>.jsonl`,
`vote_table.csv`, `vote_report.md`; and `<out>/judge_vote_table.tex`.

Paper: `judge_vote_table.tex` is the judge vote table (file `tab_judge_vote.tex` of
the paper source, according to its header). The table with label `tab:judge` reports
the same quantities; the source does not state which file it was built from. The
output of `tables` is named as the origin of the per-task table with label
`tab:fullcap` in the header of that table.

Note: `extract` depends on the matching rules of the scorer, because only emissions
that are matched to a reference event are graded. With the current scorer it returns
a different set of triples than the set used for the paper. Use the stored
`triples.jsonl` to reproduce the calibration numbers.

## human_reference.py

Purpose: the human reference. Conversion of the export of the annotation tool,
scoring variants, and the comparison with systems on the same items. The human run is
named `human_reference`.

Allowances of the human protocol (each stated with its value in the docstring):
multiple choice graded by option text; A-type emissions shifted back by the measured
reaction latency of 2.48 s; optional unbounded pre-anchor tolerance (`--pre-tol inf`).
The human reference of the paper uses the first two with a pre-anchor tolerance of
1.0 s.

```
python analysis/human_reference.py convert export.json
python analysis/human_reference.py reaction-latency --raw export.json
python analysis/human_reference.py shift-a
python analysis/human_reference.py score results/runs/human_reference/preds_shiftedA.jsonl \
    --judge hf:Qwen/Qwen3-14B --judge-cache results/judge_cache/qwen3-14b_v2.jsonl \
    --pre-tol 1.0 --eval-name eval_judge_mcqtext_shiftedA_pretol1
python -m interactionbench eval results/runs/human_reference/preds.jsonl \
    --items results/runs/human_reference/items.txt --mcq-key \
    --out results/runs/human_reference/eval_nojudge
python analysis/human_reference.py compare
```

Variants with unbounded tolerance: add `--pre-tol inf` to `score` (or to
`python -m interactionbench eval` for scoring without a judge).

Outputs: `<runs_root>/human_reference/{preds.jsonl,items.txt,human_wall_latency.jsonl,
preds_shiftedA.jsonl}`, `<runs_root>/human_reference/<eval_name>/`,
`<out>/reaction_offset.json`, `<out>/human_first300.json`.

Paper: the evaluation `eval_judge_mcqtext_shiftedA_pretol1` is the human reference row
of the system score table (label `tab:system_scores`) and the human reference line of
the mechanism figure. `human_first300.json`: not stated in the source.

## oracles.py

Purpose: three diagnostics that use the reference annotations.

| part | what it does | needs |
|---|---|---|
| `restraint` | deletes every emission that the scorer counts as a timing violation; kept emissions are unchanged | stored predictions |
| oracle timing | queries the model only at the reference response times and requires an answer | GPU |
| oracle perception | replaces the images by the reference facts that are already available | GPU |
| `manifest` | writes `oracle_manifest.json` of a timing or perception run | - |
| `score` | scores a finished run on the 103-item subset with the paper protocol | stored verdicts, or a GPU with `--gpu-judge` |
| `collect` | status of a list of runs | - |

```
python analysis/oracles.py restraint --preset subset103
python analysis/oracles.py restraint --preset all --target-suffix _oracle_restraint_all

python -m interactionbench run --plugin analysis/oracles.py --protocol oracle-timing \
    --model qwen3vl-8b --interval 1 --max-frames 16 --sample-fps 2 --mcq \
    --items benchmark/splits/subset103.txt --out results/runs/qwen3vl-8b_oracle_timing
python -m interactionbench run --plugin analysis/oracles.py --protocol oracle-perception \
    --model qwen3vl-8b --interval 1 --max-frames 16 --sample-fps 2 --mcq \
    --items benchmark/splits/subset103.txt --out results/runs/qwen3vl-8b_oracle_perception

python analysis/oracles.py manifest --oracle timing --run qwen3vl-8b_oracle_timing
python analysis/oracles.py score qwen3vl-8b_oracle_timing --gpu-judge
python analysis/oracles.py score qwen3vl-8b_sliding_iv1_mcqv4_full_oracle_restraint --gpu-judge
python analysis/oracles.py collect
```

The filtered runs of `restraint --preset all` are scored without a judge:

```
python -m interactionbench eval \
    results/runs/<run>_oracle_restraint_all/preds.jsonl --mcq-key --pre-tol 1.0 \
    --out results/runs/<run>_oracle_restraint_all/eval_nojudge_pretol1
```

Outputs: `<runs_root>/<run>_oracle_restraint*/{preds.jsonl,config.json}`,
`<runs_root>/<run>/oracle_manifest.json`, `<runs_root>/<run>/eval_paper/`,
`<out>/status.json`.

Paper: `restraint --preset subset103` with `score` gives the table with label
`tab:oracle_restraint`. `restraint --preset all` gives the oracle points of the figure
with label `fig:tasc`. Oracle timing and oracle perception: not stated in the source.

## export_tables.py

Purpose: one CSV and LaTeX tables with the overall scores of every evaluation found
under `--runs-root`, a count of prediction lines per run, and the accuracy of the agent
multiple-choice runs. The script reads files only.

Inputs: `<runs_root>/<run>/<eval_name>/summary.json` for the evaluation names given
with `--eval-name directory_name=label` (default: the names of the evaluations of the
paper).

```
python analysis/export_tables.py --out results/export
```

Outputs in `<out>`: `scores_provisional.csv`, `scores_table.tex`,
`scores_table_judge.tex`, `progress_overview.csv`, `mcqv4_agent_accuracy.csv`,
`mcqv4_agent_table.tex`; `ANOMALY.txt` when a consistency check fails.

Paper: `scores_table_judge.tex` is named as the origin of the table file
`tab_scores_judge.tex` in the header of that file. The other files: not stated in the
source.
