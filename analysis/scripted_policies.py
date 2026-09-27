#!/usr/bin/env python3
"""Scripted, video-free policies for the metric audit.

Each policy writes a predictions file without looking at any video. The policies probe
whether a timing or silence score can be raised without understanding the stream.

  always_fire      one emission every second, t = 1, 2, ..., floor(duration)
  never_fire       no emission
  periodic_10s     one emission every 10 s, t = 10, 20, ...
  chatter          Poisson process, rate 0.5 emissions per minute (exponential gaps,
                   random.Random(7)), times rounded to 0.01 s
  burst_repeater   every 20 s a burst of three emissions at t, t + 0.4, t + 0.8
  parrot           one emission 0.5 s after the question time of the item
                   (on every item, including items that require silence)
  anticipatory     one emission 0.5 s before every reference response time
                   (question time for time type A, merged answer times otherwise);
                   nothing on items that require silence. With a pre-anchor tolerance
                   of 0 every emission is premature or redundant; with a tolerance of
                   at least 0.5 s every emission is a match with delay 0.

Every emission carries the same text, "Something is happening now.".

Upstream repository or checkpoint: none. Environment: Python >= 3.10 and the
``interactionbench`` package; no GPU.

Command used for the paper numbers (pre-anchor tolerance 1 s):

  python analysis/scripted_policies.py --pre-tol 1.0 --suffix _pretol1
  python analysis/scripted_policies.py --pre-tol 1.0 --suffix _pretol1 \
      --policies anticipatory --summary-name scripted_anticipatory

Outputs:
  <policies_root>/scripted_<policy>/preds.jsonl
  <policies_root>/scripted_<policy>/<eval_name>/{summary.json,records.jsonl}
  <out>/scripted_policies<suffix>.json     total, accuracy, timing accuracy and silence
                                           compliance of every policy
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import EVAL_NOJUDGE, add_common_args, resolve_common, run_eval  # noqa: E402

TXT = "Something is happening now."
LEAD = 0.5  # anticipatory: seconds before the reference response time
POLICIES = ["always_fire", "never_fire", "periodic_10s", "chatter", "burst_repeater", "parrot"]
ALL_POLICIES = POLICIES + ["anticipatory"]


def generate(policy: str, items: list) -> list[dict]:
    from interactionbench.metrics import merge_coincident

    rng = random.Random(7)
    rows = []
    for it in items:
        dur = float(it.duration_s or 0); em = []
        q = it.question_time_s
        if policy == "always_fire":
            em = [{"t": float(t), "content": TXT} for t in range(1, int(dur) + 1)]
        elif policy == "never_fire":
            em = []
        elif policy == "periodic_10s":
            em = [{"t": float(t), "content": TXT} for t in range(10, int(dur) + 1, 10)]
        elif policy == "chatter":
            t = 0.0
            while True:
                t += rng.expovariate(0.5 / 60.0)
                if t >= dur: break
                em.append({"t": round(t, 2), "content": TXT})
        elif policy == "burst_repeater":
            for t in range(20, int(dur) + 1, 20):
                em += [{"t": float(t), "content": TXT}, {"t": t + 0.4, "content": TXT}, {"t": t + 0.8, "content": TXT}]
        elif policy == "parrot":
            if q is not None:
                em = [{"t": float(q) + 0.5, "content": TXT}]
        elif policy == "anticipatory":
            if not it.should_remain_silent:
                if it.time_type == "A":
                    times = [it.question_time_s]
                else:
                    times = [t for t, _ in merge_coincident(it.timed_answers)]
                em = [{"t": round(max(0.0, float(t) - LEAD), 2), "content": TXT} for t in times]
        rows.append({"video_id": it.video_id, "item_index": it.item_index,
                     "model": policy, "run": f"scripted_{policy}", "emissions": em})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--policies", nargs="+", default=POLICIES, choices=ALL_POLICIES,
                    help="policies to generate and score (default: %(default)s)")
    ap.add_argument("--policies-root", default=None,
                    help="directory that receives scripted_<policy>/ (default: --runs-root)")
    ap.add_argument("--pre-tol", type=float, default=None,
                    help="pre-anchor tolerance passed to the evaluator (default: its default)")
    ap.add_argument("--suffix", default="",
                    help="suffix of the evaluation directory and of the output file, "
                         "e.g. _pretol1")
    ap.add_argument("--eval-name", default=None,
                    help="evaluation directory name (default " + EVAL_NOJUDGE + "<suffix>)")
    ap.add_argument("--summary-name", default="scripted_policies",
                    help="output file stem: <out>/<summary-name><suffix>.json")
    args = ap.parse_args()
    resolve_common(args)
    eval_name = args.eval_name or f"{EVAL_NOJUDGE}{args.suffix}"
    root = args.policies_root or args.runs_root

    from interactionbench.data import iter_items, load_benchmark

    items = list(iter_items(load_benchmark(args.data)))
    res = {}
    for p in args.policies:
        d = f"{root}/scripted_{p}"; os.makedirs(d, exist_ok=True)
        with open(f"{d}/preds.jsonl", "w") as f:
            for r in generate(p, items): f.write(json.dumps(r) + "\n")
        agg, _ = run_eval(f"{d}/preds.jsonl", args.data, args.mcq_key,
                          out_dir=f"{d}/{eval_name}", pre_tol=args.pre_tol)
        o = agg["overall"]
        res[p] = {k: o.get(k) for k in ("total_score", "accuracy", "timing_accuracy", "silence_compliance")}
    out = f"{args.out}/{args.summary_name}{args.suffix}.json"
    json.dump(res, open(out, "w"), indent=1)
    print("policies:", json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
