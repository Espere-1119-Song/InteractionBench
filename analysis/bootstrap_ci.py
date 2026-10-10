#!/usr/bin/env python3
"""Bootstrap 95% confidence interval of the item-level mean total score of each run."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import EVAL_JUDGE, add_common_args, parse_runs, resolve_common  # noqa: E402

SEED = 20260923
B = 2000


def load_run_list(args) -> list[dict]:
    if len(args.runs) == 1 and args.runs[0].endswith(".json") and os.path.isfile(args.runs[0]):
        return json.load(open(args.runs[0]))
    return [{"run": run, "dir": os.path.join(args.runs_root, run, args.eval_name)}
            for _, run in parse_runs(args.runs, [])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run names, a file with one name per line, or a JSON file with "
                         "a list of {run, dir} objects")
    ap.add_argument("--eval-name", default=EVAL_JUDGE,
                    help="evaluation directory name (default: %(default)s)")
    ap.add_argument("--out-file", default=None,
                    help="output CSV (default <out>/ci_overall.csv)")
    args = ap.parse_args()
    resolve_common(args)

    random.seed(SEED)
    runs = load_run_list(args)
    out = open(args.out_file or f"{args.out}/ci_overall.csv", "w")
    out.write("run,n,overall,ci_lo,ci_hi,dir\n")
    for r in runs:
        f = os.path.join(r["dir"], "records.jsonl")
        if not os.path.exists(f):
            out.write("%s,,,,,%s MISSING\n" % (r["run"], r["dir"])); continue
        xs = [json.loads(l).get("total_score") for l in open(f)]
        xs = [float(x) for x in xs if x is not None]
        n = len(xs); mean = sum(xs) / n
        means = []
        for _ in range(B):
            s = 0.0
            for _ in range(n):
                s += xs[random.randrange(n)]
            means.append(s / n)
        means.sort()
        lo, hi = means[int(0.025 * B)], means[int(0.975 * B) - 1]
        out.write("%s,%d,%.2f,%.2f,%.2f,%s\n" % (r["run"], n, mean, lo, hi, r["dir"]))
        print(r["run"], n, round(mean, 2), round(lo, 2), round(hi, 2), flush=True)
    out.close()


if __name__ == "__main__":
    main()
