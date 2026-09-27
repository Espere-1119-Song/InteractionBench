#!/usr/bin/env python3
"""Reproduce the paper tables.

  python scripts/reproduce_paper.py list
      Print every configuration of the paper and the command that generates it.

  python scripts/reproduce_paper.py eval --runs-root results/runs --judge hf:Qwen/Qwen3-14B
      Score every run found under <runs-root>/<name>/preds.jsonl with the paper
      protocol and compare with the reference scores in configs/paper_runs.json.

The paper protocol is: multiple-choice items scored by option, free-form content judged
by Qwen3-14B with grading prompt v2, Delta = 5 s, pre-anchor tolerance 1 s, no content
gate. Use ``--judge cache:<verdict files>`` to replay stored verdicts without a GPU, or
omit ``--judge`` for lexical scoring of free-form content (numbers then differ from the
paper on free-form items).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from interactionbench.evaluate import EvalConfig, evaluate, write_outputs  # noqa: E402
from interactionbench.registry import load_plugins  # noqa: E402

KEYS = [("total", "total_score"), ("accuracy", "accuracy"),
        ("timing_accuracy", "timing_accuracy"), ("silence_compliance", "silence_compliance")]


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_list(args) -> None:
    cfg = load_config(args.config)
    for r in cfg["runs"]:
        ref = r.get("reference_scores") or {}
        stored = r.get("stored_scores") or {}
        score = (f"reference total: {ref['total']} (n={ref['n_items']})" if ref else
                 f"stored total, earlier protocol: {stored.get('total', '-')} (n={stored.get('n_items', '-')})")
        print(f"{r['name']}\n    system:   {r['system']}  [{r['setting']}]\n"
              f"    items:    {r['items']}   {score}\n    generate: {r['command']}"
              + (f" --out {args.runs_root}/{r['name']}" if r["generator"] == "ibench" else "")
              + (f"\n    note:     {r['note']}" if r.get("note") else "") + "\n")


def cmd_eval(args) -> None:
    load_plugins(args.plugin)
    cfg = load_config(args.config)
    proto = cfg["protocol"]
    judge = None
    if args.judge:
        from interactionbench.judges import make_judge
        judge = make_judge(args.judge, cache_path=args.judge_cache,
                           prompt_version=proto["judge_prompt"])
    mcq_key = args.mcq_key or f"{args.data}/mcq/mcq_key_v4.jsonl"
    rows = []
    for r in cfg["runs"]:
        if args.only and r["name"] not in args.only:
            continue
        if r.get("protocol") == "human":      # scored by analysis/human_reference.py
            continue
        preds = Path(args.runs_root) / r["name"] / "preds.jsonl"
        if not preds.exists():
            continue
        items = r["items"]
        ec = EvalConfig(
            predictions=str(preds), data=args.data, mcq_key=mcq_key,
            items=None if items in ("all1060", "partial") else str(REPO / "benchmark/splits" / f"{items}.txt"),
            skip_missing=(items == "partial"),
            delta=proto["delta_s"], pre_tol=proto["pre_tol_s"], use_gate=proto["content_gate"])
        with contextlib.redirect_stderr(io.StringIO()):
            agg, records = evaluate(ec, judge=judge)
        if args.write_eval:
            write_outputs(preds.parent / args.write_eval, agg, records)
        o, ref = agg["overall"], (r.get("reference_scores") or {})
        row = {"run": r["name"], "system": r["system"], "setting": r["setting"],
               "n_items": o["n_items"], "ref_n_items": ref.get("n_items")}
        for short, key in KEYS:
            row[short] = round(o[key], 1) if o.get(key) is not None else None
            row[f"ref_{short}"] = ref.get(short)
        row["max_abs_diff"] = max(
            (abs(row[s] - row[f"ref_{s}"]) for s, _ in KEYS
             if row[s] is not None and row[f"ref_{s}"] is not None), default=None)
        rows.append(row)
        flag = "" if row["max_abs_diff"] is None or row["max_abs_diff"] <= args.tolerance else "   <-- differs"
        print(f"{r['name']:<58} n={o['n_items']:<5} total {row['total']:>5}  (ref {row['ref_total']})"
              f"  acc {row['accuracy']:>5} TA {row['timing_accuracy']:>5} SC {row['silence_compliance']:>5}{flag}")
    if not rows:
        sys.exit(f"no predictions found under {args.runs_root}/<run name>/preds.jsonl")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")
    n_diff = sum(1 for r in rows if r["max_abs_diff"] is not None and r["max_abs_diff"] > args.tolerance)
    print(f"\n{len(rows)} runs scored; {len(rows) - n_diff} within {args.tolerance} of the reference, "
          f"{n_diff} differ")
    if judge is not None and getattr(judge, "n_missing", 0):
        print(f"warning: {judge.n_missing} judge verdicts were missing from the cache and counted as 0")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs/paper_runs.json"))
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list")
    p.add_argument("--runs-root", default="results/runs")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("eval")
    p.add_argument("--runs-root", default="results/runs")
    p.add_argument("--data", default="data/interactionbench")
    p.add_argument("--mcq-key", default=None)
    p.add_argument("--judge", default=None)
    p.add_argument("--judge-cache", default="results/judge_cache/qwen3-14b_v2.jsonl")
    p.add_argument("--only", nargs="*", default=None, help="run names to score")
    p.add_argument("--write-eval", default=None, metavar="NAME",
                   help="also write <run>/<NAME>/{summary.json,records.jsonl}")
    p.add_argument("--out", default="results/paper_scores.csv")
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="largest difference from the reference that counts as reproduced")
    p.add_argument("--plugin", action="append", default=[])
    p.set_defaults(func=cmd_eval)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
