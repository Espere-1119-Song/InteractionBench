#!/usr/bin/env python3
"""Rank stability of the system ranking under other delay bounds (Kendall tau)."""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_common_args, parse_runs, resolve_common, run_eval  # noqa: E402

DELTAS = [2, 3, 5, 8, 10]


def tau(a, b):
    n = len(a); conc = disc = 0
    for i, j in itertools.combinations(range(n), 2):
        s = (a[i] - a[j]) * (b[i] - b[j])
        if s > 0: conc += 1
        elif s < 0: disc += 1
    return (conc - disc) / (conc + disc) if conc + disc else float("nan")


def rank(vals):
    order = sorted(range(len(vals)), key=lambda i: -vals[i]); rk = [0]*len(vals)
    for pos, i in enumerate(order): rk[i] = pos + 1
    return rk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--sweep-dir", default=None,
                    help="directory with <run>_d<Delta>/summary.json "
                         "(default <out>/delta_sweep)")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run names, or a file with one name per line, whose missing "
                         "evaluations are created before the analysis (default: none)")
    ap.add_argument("--items", default=None,
                    help="file of item_ids used when evaluations are created")
    ap.add_argument("--pre-tol", type=float, default=None,
                    help="pre-anchor tolerance used when evaluations are created "
                         "(default: the evaluator default)")
    ap.add_argument("--judge", default=None,
                    help="judge spec used when evaluations are created, e.g. "
                         "ensemble:qwen_qwen3-14b (default: no judge)")
    ap.add_argument("--judge-cache-dir", default="results/judge_cache",
                    help="directory of the stored verdicts of an ensemble judge")
    ap.add_argument("--judge-prompt", default="v2", help="grading prompt version")
    args = ap.parse_args()
    resolve_common(args)
    sweep = args.sweep_dir or f"{args.out}/delta_sweep"

    for _, run in parse_runs(args.runs, []):
        for d in DELTAS:
            target = "%s/%s_d%d" % (sweep, run, d)
            if os.path.exists(f"{target}/summary.json"):
                continue
            kw = {}
            if args.judge:
                kw = {"judge": args.judge, "judge_prompt": args.judge_prompt}
                if args.judge.startswith("ensemble:"):
                    kw["judge_args"] = {"cache_dir": args.judge_cache_dir}
            run_eval(f"{args.runs_root}/{run}/preds.jsonl", args.data, args.mcq_key,
                     out_dir=target, items=args.items, pre_tol=args.pre_tol,
                     delta=float(d), **kw)
            print("scored", target, flush=True)

    runs = sorted({os.path.basename(p).rsplit("_d", 1)[0] for p in glob.glob(f"{sweep}/*_d5")})
    deltas = DELTAS
    score = {}
    for r in runs:
        for d in deltas:
            f = "%s/%s_d%d/summary.json" % (sweep, r, d)
            if os.path.exists(f):
                o = json.load(open(f))["overall"]
                score[(r, d)] = (o["total_score"], o["timing_accuracy"], o["silence_compliance"])
    rs = [r for r in runs if all((r, d) in score for d in deltas)]
    base = [score[(r, 5)][0] for r in rs]; base_rk = rank(base)
    out = open(f"{args.out}/kendall.csv", "w"); out.write("delta,n_runs,tau_overall,tau_timing,max_rank_shift,top3_same,mean_overall\n")
    top3_base = set(sorted(range(len(rs)), key=lambda i: -base[i])[:3])
    for d in deltas:
        ov = [score[(r, d)][0] for r in rs]; ta = [score[(r, d)][1] for r in rs]
        ta5 = [score[(r, 5)][1] for r in rs]
        rk = rank(ov); shift = max(abs(a - b) for a, b in zip(rk, base_rk))
        top3 = set(sorted(range(len(rs)), key=lambda i: -ov[i])[:3])
        out.write("%d,%d,%.3f,%.3f,%d,%s,%.2f\n" % (d, len(rs), tau(ov, base), tau(ta, ta5), shift, top3 == top3_base, sum(ov)/len(ov)))
    out.close()
    with open(f"{args.out}/per_run.csv", "w") as f:
        f.write("run," + ",".join("overall_d%d" % d for d in deltas) + "," + ",".join("ta_d%d" % d for d in deltas) + "\n")
        for r in rs:
            f.write(r + "," + ",".join("%.2f" % score[(r, d)][0] for d in deltas) + "," + ",".join("%.2f" % score[(r, d)][1] for d in deltas) + "\n")
    print(open(f"{args.out}/kendall.csv").read())


if __name__ == "__main__":
    main()
