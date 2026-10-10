#!/usr/bin/env python3
"""Delta / gate sensitivity grid and emission-thinning TA-SC curves."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_common_args, parse_runs, resolve_common, run_eval  # noqa: E402

PRESETS = {
    "paper16": {
        "runs": ["joyai_streaming_4fps_mcqv4", "mmduet2-3b_streaming_4fps_mcq_MERGED",
                 "videollm-online-8b_streaming_8fps_mcq", "fvstream-7b_polling_iv1_8fps_mcq_lenientparse",
                 "llava-ov2-8b_sliding_iv1_mcqv4_full", "llava-ov2-8b_interleaved_iv1_mcqv4_full",
                 "qwen3vl-8b_sliding_iv1_mcqv4_full", "qwen3vl-8b_sliding_fps05_iv1_mcqv4_full",
                 "qwen3vl-8b_sliding_fps1_iv1_mcqv4_full", "qwen3vl-8b_sliding_fps4_iv1_mcqv4_full",
                 "qwen3vl-8b_sliding_win64_iv1_mcqv4_full", "qwen3vl-8b_interleaved_iv1_mcqv4_full",
                 "qwen3vl-4b_sliding_iv1_mcqv4_full", "qwen3vl-4b_interleaved_iv1_mcqv4_full",
                 "qwen3vl-8b_blind_iv1_mcqv4_full", "qwen3vl-8b_sliding_offline_iv1_mcqv4_full"],
        "curves": [("JoyAI-VL (native)", "joyai_streaming_4fps_mcqv4"),
                   ("Qwen3-VL-8B (L=64)", "qwen3vl-8b_sliding_win64_iv1_mcqv4_full"),
                   ("MMDuet2-3B", "mmduet2-3b_streaming_4fps_mcq_MERGED"),
                   ("LLaVA-OV2-8B (sliding)", "llava-ov2-8b_sliding_iv1_mcqv4_full")],
    },
    "all": {
        "runs": ["joyai_streaming_4fps_mcqv4", "mmduet2-3b_streaming_4fps_mcq_MERGED",
                 "fvstream-7b_polling_iv1_8fps_mcq_lenientparse", "videollm-online-8b_streaming_8fps_mcq",
                 "qwen3vl-8b_sliding_iv1_mcqv4_full", "llava-ov2-8b_sliding_iv1_mcqv4_full",
                 "qwen3vl-4b_sliding_iv1_mcqv4_full", "gemma-3n-e4b_sliding_iv1_mcqv4_full",
                 "minicpm-o-4.5_sliding_iv1_mcqv4_full", "qwen2.5-omni-7b_sliding_iv1_mcqv4_full",
                 "qwen3vl-8b_sliding_offline_iv1_mcqv4_full", "qwen3vl-4b_sliding_offline_iv1_mcqv4_full",
                 "llava-ov2-8b_sliding_offline_iv1_mcqv4_full", "minicpm-o-4.5_sliding_offline_iv1_mcqv4_full",
                 "qwen2.5-omni-7b_sliding_offline_iv1_mcqv4_full", "gemma-3n-e4b_sliding_offline_iv1_mcqv4_full",
                 "qwen3vl-8b_blind_iv1_mcqv4_full", "qwen3vl-8b_sliding_win64_iv1_mcqv4_full",
                 "qwen3vl-8b_interleaved_iv1_mcqv4_full", "qwen3vl-4b_interleaved_iv1_mcqv4_full",
                 "llava-ov2-8b_interleaved_iv1_mcqv4_full", "qwen3-8b-text_blind_iv1_mcqv4_full"],
        "curves": None,
    },
}
CFGS = [("delta", d) for d in (2, 3, 8, 10)] + [("gate", g) for g in (0.2, 0.4, 0.5)] + [("base", None)]
KEEPS = (1.0, 0.8, 0.6, 0.4, 0.2, 0.1)


def ev(pred: str, extra: dict, common: dict) -> dict:
    agg, _ = run_eval(pred, common["data"], common["mcq_key"],
                      pre_tol=common["pre_tol"], **extra)
    return agg["overall"]


def sens_task(task):
    run, tag, pred, extra, common = task
    return ev(pred, extra, common)["total_score"]


def thin(job):
    run, keep, common = job
    src = [json.loads(l) for l in open(f"{common['runs_root']}/{run}/preds.jsonl")]
    rng = random.Random(11)
    tmp = f"{common['work_dir']}/thin_{run}_{int(keep*100)}.jsonl"
    with open(tmp, "w") as f:
        for r in src:
            r2 = dict(r); r2["emissions"] = [e for e in r["emissions"] if rng.random() < keep]
            f.write(json.dumps(r2) + "\n")
    s = ev(tmp, {}, common)
    return {"keep": keep, "TA": s["timing_accuracy"], "SC": s["silence_compliance"], "total": s["total_score"]}


def pool_map(fn, jobs, n_workers):
    if n_workers <= 1:
        return [fn(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(fn, jobs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="paper16",
                    help="run list (default: %(default)s)")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run names of the sensitivity grid, or a file with one name per "
                         "line; replaces the list of the preset")
    ap.add_argument("--curve-runs", nargs="*", default=None,
                    help="runs of the thinning curves: entries 'label=run_name' or "
                         "'run_name', or a file; replaces the list of the preset")
    ap.add_argument("--pre-tol", type=float, default=None,
                    help="pre-anchor tolerance passed to the evaluator (default: its default)")
    ap.add_argument("--suffix", default="", help="output-file suffix, e.g. _pretol1")
    ap.add_argument("--jobs", type=int, default=1, help="parallel evaluation processes")
    ap.add_argument("--work-dir", default=None,
                    help="directory for the thinned prediction files "
                         "(default <out>/sweep_work<suffix>)")
    args = ap.parse_args()
    resolve_common(args)
    preset = PRESETS[args.preset]
    runs = [r for _, r in parse_runs(args.runs, [(r, r) for r in preset["runs"]])]
    if args.curve_runs:
        curves = parse_runs(args.curve_runs, [])
    elif args.runs:
        curves = [(l, r) for l, r in (preset["curves"] or [(r, r) for r in runs]) if r in runs]
    else:
        curves = preset["curves"] or [(r, r) for r in runs]
    work = args.work_dir or f"{args.out}/sweep_work{args.suffix}"
    os.makedirs(work, exist_ok=True)
    common = {"data": args.data, "mcq_key": args.mcq_key, "pre_tol": args.pre_tol,
              "runs_root": args.runs_root, "work_dir": work}

    tasks = []
    for run in runs:
        pred = f"{args.runs_root}/{run}/preds.jsonl"
        if not os.path.exists(pred):
            print(f"note: {pred} not found, run left out", file=sys.stderr)
            continue
        for kind, val in CFGS:
            extra = {} if kind == "base" else ({"delta": float(val)} if kind == "delta" else {"gate_thresh": val})
            tag = "base" if kind == "base" else f"{kind}{val}"
            tasks.append((run, tag, pred, extra, common))
    outs = pool_map(sens_task, tasks, args.jobs)
    sens = {}
    for (run, tag, _, _, _), tot in zip(tasks, outs): sens.setdefault(run, {})[tag] = tot
    json.dump(sens, open(f"{args.out}/sensitivity{args.suffix}.json", "w"), indent=1)
    print("sensitivity done")

    jobs = []
    for label, run in curves:
        if not os.path.exists(f"{args.runs_root}/{run}/preds.jsonl"):
            print(f"note: predictions of {run} not found, curve left out", file=sys.stderr)
            continue
        jobs += [(run, keep, common) for keep in KEEPS]
    label_of = {run: label for label, run in curves}
    pts = pool_map(thin, jobs, args.jobs)
    cur = {}
    for (run, keep, _), p in zip(jobs, pts): cur.setdefault(label_of[run], []).append(p)
    json.dump(cur, open(f"{args.out}/thinning_curves{args.suffix}.json", "w"), indent=1)
    print("curves done")


if __name__ == "__main__":
    main()
