#!/usr/bin/env python3
"""Aggregations over per-item evaluation records: violation anatomy and the per-task table."""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (EVAL_NOJUDGE, add_common_args, parse_runs, resolve_common,  # noqa: E402
                     run_eval)

RUNS = [("JoyAI-VL (native, 4fps)", "joyai_streaming_4fps_mcqv4"),
        ("Qwen3-VL-8B (sliding, L=64)", "qwen3vl-8b_sliding_win64_iv1_mcqv4_full"),
        ("MMDuet2-3B", "mmduet2-3b_streaming_4fps_mcq_MERGED"),
        ("LLaVA-OV2-8B (sliding)", "llava-ov2-8b_sliding_iv1_mcqv4_full"),
        ("Qwen3-VL-8B (blind)", "qwen3vl-8b_blind_iv1_mcqv4_full"),
        ("Qwen3-VL-8B (offline)", "qwen3vl-8b_sliding_offline_iv1_mcqv4_full")]

ORDER = ["PTR", "TOA", "CST", "IVQA", "LCG", "LVM"]
FULLRUNS = [("JoyAI-VL (native, 4 fps)", "joyai_streaming_4fps_mcqv4"),
            ("VideoLLM-online-8B", "videollm-online-8b_streaming_8fps_mcq"),
            ("MMDuet2-3B", "mmduet2-3b_streaming_4fps_mcq_MERGED"),
            ("Qwen3-VL-8B (sliding, L=64)", "qwen3vl-8b_sliding_win64_iv1_mcqv4_full"),
            ("Qwen3-VL-8B (sliding, 0.5 fps)", "qwen3vl-8b_sliding_fps05_iv1_mcqv4_full"),
            ("Qwen3-VL-8B (sliding, 1 fps)", "qwen3vl-8b_sliding_fps1_iv1_mcqv4_full"),
            ("Qwen3-VL-8B (sliding, 2 fps)", "qwen3vl-8b_sliding_iv1_mcqv4_full"),
            ("Qwen3-VL-8B (sliding, 4 fps)", "qwen3vl-8b_sliding_fps4_iv1_mcqv4_full"),
            ("Qwen3-VL-8B (interleave)", "qwen3vl-8b_interleaved_iv1_mcqv4_full"),
            ("LLaVA-OV2-8B (sliding)", "llava-ov2-8b_sliding_iv1_mcqv4_full"),
            ("LLaVA-OV2-8B (interleave)", "llava-ov2-8b_interleaved_iv1_mcqv4_full"),
            ("Qwen3-VL-4B (sliding)", "qwen3vl-4b_sliding_iv1_mcqv4_full"),
            ("Qwen3-VL-4B (interleave)", "qwen3vl-4b_interleaved_iv1_mcqv4_full"),
            ("Qwen3-VL-8B (offline replay)", "qwen3vl-8b_sliding_offline_iv1_mcqv4_full"),
            ("Qwen3-VL-4B (offline replay)", "qwen3vl-4b_sliding_offline_iv1_mcqv4_full"),
            ("LLaVA-OV2-8B (offline replay)", "llava-ov2-8b_sliding_offline_iv1_mcqv4_full"),
            ("Qwen3-VL-8B (blind)", "qwen3vl-8b_blind_iv1_mcqv4_full")]


def anatomy(recs: list[dict]) -> dict:
    pos = [r for r in recs if r["family"] != "negative"]
    neg = [r for r in recs if r["family"] == "negative"]
    def nrm(r, k): return r[k] / max(r["n_gt"], 1)
    a = {"prem": st.mean(nrm(r, "v_premature") for r in pos),
         "redun": st.mean(nrm(r, "v_redundant") for r in pos),
         "spur": st.mean(nrm(r, "v_spurious") for r in pos),
         "neg_fa": st.mean(1.0 if r["silence_compliance"] < 100 else 0.0 for r in neg) if neg else None,
         "TA": st.mean(r["timing_accuracy"] for r in pos if r["timing_accuracy"] is not None),
         "SC": st.mean(r["silence_compliance"] for r in pos if r["silence_compliance"] is not None),
         "rtf": st.mean(r["realtime_factor"] for r in recs if r.get("realtime_factor") is not None) if any(r.get("realtime_factor") is not None for r in recs) else None,
         "epm": st.mean(r["emissions_per_min"] for r in recs if r.get("emissions_per_min") is not None),
         "lat": st.mean(r["poll_latency_mean"] for r in recs if r.get("poll_latency_mean") is not None) if any(r.get("poll_latency_mean") is not None for r in recs) else None}
    byord = collections.defaultdict(list)
    for r in pos:
        if r["family"] in ("B_trigger", "C_counting"):
            byord[min(int(r["item_id"].split("#")[1]), 2)].append(1.0 if r["n_matched"] > 0 else 0.0)
    a["ord_hit"] = {k: round(st.mean(v), 3) for k, v in sorted(byord.items())}
    tot = a["prem"] + a["redun"] + a["spur"]
    a["dom"] = max(("premature", a["prem"]), ("redundant", a["redun"]), ("spurious", a["spur"]), key=lambda x: x[1])[0] if tot else None
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in a.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--runs", nargs="*", default=None,
                    help="runs of the violation anatomy: entries 'label=run_name' or "
                         "'run_name', or a file with one entry per line "
                         "(default: the six runs of the paper)")
    ap.add_argument("--table-runs", nargs="*", default=None,
                    help="runs of the per-task table, same format "
                         "(default: the 17 runs of the paper)")
    ap.add_argument("--pre-tol", type=float, default=None,
                    help="pre-anchor tolerance used when an evaluation has to be created "
                         "(default: the evaluator default)")
    ap.add_argument("--suffix", default="",
                    help="suffix of the evaluation directory and of the output files, "
                         "e.g. _pretol1")
    ap.add_argument("--eval-name", default=None,
                    help="evaluation directory name (default " + EVAL_NOJUDGE + "<suffix>)")
    args = ap.parse_args()
    resolve_common(args)
    evd = args.eval_name or f"{EVAL_NOJUDGE}{args.suffix}"
    root = args.runs_root

    agg = {}
    for name, run in parse_runs(args.runs, RUNS):
        fp = f"{root}/{run}/{evd}/records.jsonl"
        if not os.path.exists(fp):
            run_eval(f"{root}/{run}/preds.jsonl", args.data, args.mcq_key,
                     out_dir=f"{root}/{run}/{evd}", pre_tol=args.pre_tol)
        recs = [json.loads(l) for l in open(fp)]
        agg[name] = anatomy(recs)
    json.dump(agg, open(f"{args.out}/violation_anatomy{args.suffix}.json", "w"), indent=1)
    print("anatomy:", json.dumps(agg, indent=1))

    lines = []
    for name, run in parse_runs(args.table_runs, FULLRUNS):
        fp = f"{root}/{run}/{evd}/summary.json"
        if not os.path.exists(fp) and os.path.exists(f"{root}/{run}/preds.jsonl"):
            run_eval(f"{root}/{run}/preds.jsonl", args.data, args.mcq_key,
                     out_dir=f"{root}/{run}/{evd}", pre_tol=args.pre_tol)
        if not os.path.exists(fp): continue
        s = json.load(open(fp)); bc = s["by_capability"]; o = s["overall"]
        cells = [f"{o['total_score']:.1f}"]
        for c in ORDER:
            d = bc.get(c, {})
            for k in ("accuracy", "timing_accuracy", "silence_compliance"):
                v = d.get(k); cells.append("--" if v is None else f"{v:.1f}")
        nm = name.replace("=", "{=}")
        lines.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    open(f"{args.out}/fullcap_rows{args.suffix}.tex", "w").write("\n".join(lines) + "\n")
    print("fullcap rows:", len(lines))


if __name__ == "__main__":
    main()
