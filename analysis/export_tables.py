#!/usr/bin/env python3
"""Collect the summaries of all evaluations into one CSV and LaTeX tables.

Every run directory under ``--runs-root`` is searched for the evaluation directories
listed in ``--eval-name`` (entries ``directory_name=label``). One row is written per
(run, evaluation) with the number of items, total, accuracy, timing accuracy and
silence compliance of ``overall`` in summary.json. The label records which judge
produced the scores; an evaluation without a judge scores multiple choice and timing
mechanically and free-form content by token F1.

Also written:
  progress_overview.csv     number of prediction lines of every run
  mcqv4_agent_accuracy.csv  accuracy of the agent multiple-choice runs. These runs are
                            the directories matching ``--mcq-runs-glob``; they store
                            results.jsonl with one row per item ({item_id, letter,
                            correct, ...}). The last row of an item is used, an
                            unanswered item counts as wrong, and the first root that
                            holds a run takes precedence.
  ANOMALY.txt               written only when a consistency check fails (see below);
                            removed when all checks pass

Consistency checks on the evaluations without a judge (label of the first entry of
``--eval-name``): the total of ``scripted_never_fire`` has to be within 0.05 of 6.509,
and the accuracy of ``qwen3vl-8b_sliding_fps05_iv1_mcqv4_full`` has to be at least 40.
A failure indicates an evaluation that was run without the answer key.

Upstream repository or checkpoint: none. Environment: Python >= 3.10; no GPU. The
script reads files only.

Command used for the paper tables:

  python analysis/export_tables.py --out results/export

Outputs in <out>:
  scores_provisional.csv, scores_table.tex, scores_table_judge.tex,
  progress_overview.csv, mcqv4_agent_accuracy.csv, mcqv4_agent_table.tex
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_common_args, resolve_common  # noqa: E402

EVAL_DIRS = [
    ("nojudge(provisional)", "eval_nojudge"),
    ("qwen3-14b-binary(v1)", "eval_qwen_qwen3-14b_v1"),
    ("qwen3-14b-binary(v2)", "eval_qwen_qwen3-14b_v2"),
    ("qwen3-32b-binary(v2)", "eval_qwen_qwen3-32b_v2"),
    ("gpt-oss-20b-binary(v2)", "eval_openai_gpt-oss-20b_v2"),
    ("ensemble3-majority(v2)", "eval_ensemble3_v2"),
    ("qwen3-14b-binary(v2,nogate)", "eval_qwen_qwen3-14b_v2_nogate"),
    ("gemini-binary(retired)", "eval_final"),
    ("gemini-vdc(retired)", "eval_final_vdc"),
]
JUDGE_FINAL = "qwen3-14b-binary(v2)"


def parse_eval_names(values: list[str] | None) -> list[tuple[str, str]]:
    """[(label, directory_name)] from entries 'directory_name=label'."""
    if not values:
        return list(EVAL_DIRS)
    out = []
    for v in values:
        name, sep, label = v.partition("=")
        out.append((label if sep else name, name))
    return out


def cnt(p):
    try:
        return sum(1 for l in open(p) if l.strip())
    except FileNotFoundError:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, multi_root=True)
    ap.add_argument("--eval-name", nargs="*", default=None,
                    help="evaluation directories as 'directory_name=label' "
                         "(default: " + ", ".join(f"{n}={l}" for l, n in EVAL_DIRS) + ")")
    ap.add_argument("--final-label", default=JUDGE_FINAL,
                    help="label of the rows of scores_table_judge.tex (default: %(default)s)")
    ap.add_argument("--progress-roots", nargs="*", default=None,
                    help="roots counted in progress_overview.csv (default: --runs-root)")
    ap.add_argument("--mcq-runs-glob", default="*_qwenmm_mcqv4",
                    help="run-directory pattern of the agent multiple-choice runs "
                         "(default: %(default)s)")
    args = ap.parse_args()
    resolve_common(args)
    out = args.out
    eval_dirs = parse_eval_names(args.eval_name)

    ROWS = []
    for base in args.runs_root:
        for run_dir in sorted(glob.glob(f"{base}/*")):
            for judge, ev in eval_dirs:
                fp = f"{run_dir}/{ev}/summary.json"
                if not os.path.exists(fp):
                    continue
                o = json.load(open(fp)).get("overall", {})
                ROWS.append({
                    "run": os.path.basename(run_dir), "judge": judge,
                    "n": o.get("n_items"), "total": o.get("total_score"),
                    "acc": o.get("accuracy"), "TA": o.get("timing_accuracy"),
                    "SC": o.get("silence_compliance"),
                })

    with open(f"{out}/scores_provisional.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "judge", "n", "total", "acc", "TA", "SC"])
        w.writeheader(); w.writerows(ROWS)

    with open(f"{out}/scores_table.tex", "w") as f:
        f.write("% auto-generated by export_tables.py — provisional scores\n"
                "\\begin{tabular}{llrrrrr}\n\\toprule\n"
                "Run & Judge & $n$ & Total & Acc & TA & SC \\\\\n\\midrule\n")
        for r in ROWS:
            vals = [f"{r[k]:.1f}" if isinstance(r[k], (int, float)) and k != "n"
                    else str(r[k] if r[k] is not None else "--")
                    for k in ("n", "total", "acc", "TA", "SC")]
            name = r["run"].replace("_", "\\_")
            f.write(f"{name} & {r['judge']} & " + " & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # rows of the final judge only
    final = args.final_label
    with open(f"{out}/scores_table_judge.tex", "w") as f:
        f.write(f"% auto-generated by export_tables.py — final open-judge scores ({final}); MCQ items option-scored\n"
                "\\begin{tabular}{lrrrrr}\n\\toprule\n"
                "Run & $n$ & Total & Acc & TA & SC \\\\\n\\midrule\n")
        for r in ROWS:
            if r["judge"] != final:
                continue
            vals = [f"{r[k]:.1f}" if isinstance(r[k], (int, float)) and k != "n" else str(r[k] if r[k] is not None else "--")
                    for k in ("n", "total", "acc", "TA", "SC")]
            f.write(f"{r['run'].replace('_', chr(92) + '_')} & " + " & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    prog = []
    progress_roots = args.progress_roots if args.progress_roots else args.runs_root
    run_dirs = []
    for base in progress_roots:
        run_dirs += glob.glob(f"{base}/*")
    for run_dir in sorted(run_dirs):
        if not os.path.isdir(run_dir):
            continue
        n = cnt(f"{run_dir}/preds.jsonl") or cnt(f"{run_dir}/results.jsonl")
        if n:
            prog.append({"run": os.path.basename(run_dir), "items_done": n})
    with open(f"{out}/progress_overview.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "items_done"])
        w.writeheader(); w.writerows(prog)

    # Agent multiple-choice accuracy; results.jsonl rows carry gt / letter / correct.
    # Duplicate rows of an item are resolved by the last occurrence.
    mcq_rows = []
    for base in args.runs_root:
        for fp in sorted(glob.glob(f"{base}/{args.mcq_runs_glob}/results.jsonl")):
            run = os.path.basename(os.path.dirname(fp))
            if any(r["run"] == run for r in mcq_rows):
                continue  # the first root takes precedence
            last = {}
            for l in open(fp):
                if l.strip():
                    d = json.loads(l); last[d["item_id"]] = d
            rows = list(last.values())
            answered = sum(1 for d in rows if str(d.get("letter") or "").strip() not in ("", "None"))
            correct = sum(1 for d in rows if str(d.get("correct")) == "True")
            mcq_rows.append({"run": run, "n": len(rows), "answered": answered, "correct": correct,
                             "acc": round(100.0 * correct / len(rows), 3) if rows else None})
    with open(f"{out}/mcqv4_agent_accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "n", "answered", "correct", "acc"])
        w.writeheader(); w.writerows(mcq_rows)
    with open(f"{out}/mcqv4_agent_table.tex", "w") as f:
        f.write("% auto-generated by export_tables.py — agent MCQ v4 accuracy (688 items, unanswered counted wrong)\n"
                "\\begin{tabular}{lrrrr}\n\\toprule\nRun & $n$ & Answered & Correct & Acc (\\%) \\\\\n\\midrule\n")
        for r in mcq_rows:
            f.write(f"{r['run'].replace('_', chr(92) + '_')} & {r['n']} & {r['answered']} & {r['correct']} & {r['acc']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"export built: {len(ROWS)} score rows, {len(prog)} progress rows, {len(mcq_rows)} mcqv4 rows")

    # ---- consistency checks: detect evaluations that were run without the answer key
    nojudge_label = eval_dirs[0][0]
    _c = {r["run"]: r for r in ROWS if r["judge"] == nojudge_label}
    _warn = []
    _nf = _c.get("scripted_never_fire"); _fp = _c.get("qwen3vl-8b_sliding_fps05_iv1_mcqv4_full")
    if _nf and abs((_nf["total"] or 0) - 6.509) > 0.05: _warn.append(f"never_fire total {_nf[chr(39) + 'total' + chr(39)]} != 6.509")
    if _fp and (_fp["acc"] or 0) < 40: _warn.append("fps05 acc <40: MCQ key likely missing in some eval — DO NOT EXPORT")
    if _warn:
        open(f"{out}/ANOMALY.txt", "w").write(chr(10).join(_warn) + chr(10))
        print("!!! EXPORT ANOMALY !!!"); [print("  ", w) for w in _warn]
    else:
        import contextlib
        with contextlib.suppress(FileNotFoundError): os.remove(f"{out}/ANOMALY.txt")


if __name__ == "__main__":
    main()
