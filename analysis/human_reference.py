#!/usr/bin/env python3
"""Human reference: conversion, scoring variants and comparison with systems.

A human participant watched the streams in an annotation tool and typed free-text
answers while the video played. The tool export is converted into the prediction
schema and scored with the same metrics as the systems. The human run is named
``human_reference``.

Allowances of the human protocol. Each one is a departure from the protocol applied to
the systems, and is applied only when the corresponding sub-command or option is used:

  1. Multiple choice scored by option text (``score``). Systems see the options and
     answer by letter; their multiple-choice items are scored by option match. The
     participant did not see the options and answered in free text, so the answer is
     graded by the judge (same judge, same grading prompt) against the text of the
     correct option (``answer_text`` of the answer key). Free-form items are graded
     as usual. No numeric parameter.
  2. Reaction latency removed on A-type items (``shift-a``). Every emission of an
     item with time type A is moved back by the measured human reaction latency,
     2.48 s, and is not moved before the question time: t' = max(t - 2.48, question
     time, 0), rounded to 0.001 s. The value is the median delay between the question
     time and the first emission at or after it, over the 53 answered IVQA items
     with time type A of the human run (``reaction-latency`` recomputes it; key
     ``shift_s_used``). Items with time type B or C are not shifted.
  3. Unbounded pre-anchor tolerance (``score --pre-tol inf``). Systems are scored
     with a pre-anchor tolerance of 1.0 s. With ``inf`` no early emission on a
     positive item counts as a violation: emissions are attributed greedily in time
     order, an emission before the first reference time answers the first event with
     delay 0, and only repeated answers to an event that is already answered count
     as redundant. Items that require silence are scored as for systems.

Unchanged for the human run: the acceptable-delay bound Delta = 5.0 s, the disabled
content gate, and the aggregation.

The human reference reported in the paper uses allowances 1 and 2 with the pre-anchor
tolerance of the systems (1.0 s). The variants with ``--pre-tol inf`` are additional.

Sub-commands:

  convert           tool export -> preds.jsonl, items.txt, human_wall_latency.jsonl
  reaction-latency  delay statistics of the human run and the shift used by shift-a
  shift-a           preds.jsonl -> preds_shiftedA.jsonl (allowance 2)
  score             scoring with allowance 1, optionally allowance 3
  compare           human run and systems aggregated over the items of the human run

Input of ``convert``: a JSON object with the key ``results`` that maps item_id to
{item_id, video_id, capability, time_type, interaction_type, question_time_s,
updates: [{video_time_s, wall_ms_since_question, answer, at}], ...}.
Conversion rules: every update becomes one emission at t = video_time_s (the stream
clock the participant was watching) with the answer text unchanged; updates with an
empty answer are dropped; emissions are sorted by t; an item without any non-empty
update is written with an empty emission list (silent); no latency_s is attached
(wall_ms_since_question is kept in the side file human_wall_latency.jsonl). The name
of the participant in the export is not copied into the output.

Upstream checkpoint: the judge, e.g. https://huggingface.co/Qwen/Qwen3-14B, or stored
verdicts replayed with ``--judge cache:<files>``.
Environment: Python >= 3.10 and the ``interactionbench`` package; ``score`` with an
``hf:`` judge also needs torch and transformers and a GPU.

Commands used for the paper numbers:

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

Outputs:
  <runs_root>/human_reference/{preds.jsonl,items.txt,human_wall_latency.jsonl}
  <runs_root>/human_reference/preds_shiftedA.jsonl
  <runs_root>/human_reference/<eval_name>/{summary.json,records.jsonl}
  <out>/reaction_offset.json
  <out>/human_first300.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import EVAL_JUDGE, EVAL_NOJUDGE, add_common_args, parse_runs, resolve_common  # noqa: E402

HUMAN_RUN = "human_reference"
SYSTEMS = ["qwen3vl-8b_sliding_win64_iv1_mcqv4_full", "joyai_streaming_4fps_mcqv4",
           "qwen3vl-8b_sliding_offline_iv1_mcqv4_full", "llava-ov2-8b_sliding_iv1_mcqv4_full",
           "mmduet2-3b_streaming_4fps_mcq_MERGED"]
METRICS = ["total_score", "accuracy", "timing_accuracy", "silence_compliance"]


def human_dir(args) -> Path:
    return Path(args.runs_root) / args.human_run


# ------------------------------------------------------------------ convert

def cmd_convert(args) -> None:
    d = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    results = d["results"]
    out = Path(args.run_dir) if args.run_dir else human_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    n_items = n_upd = n_empty = n_silent = n_pre = n_broken = 0
    lines, items, side = [], [], []
    for iid, r in results.items():
        vid, idx = iid.rsplit("#", 1)
        assert r.get("video_id", vid) == vid, iid
        qt = r.get("question_time_s")
        ems = []
        for u in r.get("updates", []):
            ans = (u.get("answer") or "").strip()
            if not ans:
                n_empty += 1; continue
            t = float(u["video_time_s"])
            if qt is not None and t < float(qt):
                n_pre += 1
            ems.append({"t": t, "content": ans})
            side.append({"item_id": iid, "t": t, "wall_ms_since_question": u.get("wall_ms_since_question"), "at": u.get("at")})
        ems.sort(key=lambda e: e["t"])
        n_upd += len(ems)
        if not ems: n_silent += 1
        if r.get("broken"): n_broken += 1
        lines.append(json.dumps({"video_id": vid, "item_index": int(idx), "model": args.model, "run": args.human_run,
                                 "emissions": ems}, ensure_ascii=False))
        items.append(iid); n_items += 1
    (out / "preds.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "items.txt").write_text("\n".join(items) + "\n", encoding="utf-8")
    (out / "human_wall_latency.jsonl").write_text("\n".join(json.dumps(s) for s in side) + "\n", encoding="utf-8")
    print(f"items={n_items} emissions={n_upd} empty_dropped={n_empty} silent_items={n_silent} "
          f"pre_query_updates={n_pre} broken_flag={n_broken}", file=sys.stderr)


# ------------------------------------------------------------------ reaction latency

def _load_human(args):
    from interactionbench.data import iter_items, load_benchmark

    H = human_dir(args)
    preds = {}
    for l in (H / args.preds_name).read_text().splitlines():
        if l.strip():
            p = json.loads(l); preds[f'{p["video_id"]}#{p["item_index"]}'] = p
    keep = {l.strip() for l in (H / "items.txt").read_text().splitlines() if l.strip()}
    items = {it.item_id: it for it in iter_items(load_benchmark(args.data)) if it.item_id in keep}
    return H, preds, keep, items


def cmd_reaction_latency(args) -> None:
    from interactionbench.metrics import MetricConfig, decision_timing, merge_coincident

    H, preds, keep, items = _load_human(args)
    raw = json.loads(Path(args.raw or H / "human_raw.json").read_text())["results"]
    print("items in benchmark:", len(items), "of", len(keep))
    def q(xs):
        xs = sorted(xs); n = len(xs)
        if not n: return None
        return {"n": n, "median": round(st.median(xs), 2), "mean": round(st.mean(xs), 2),
                "p25": round(xs[n//4], 2), "p75": round(xs[(3*n)//4], 2), "min": round(xs[0], 2), "max": round(xs[-1], 2)}
    res = {}
    # question-answer items: first emission after the question
    for label, caps in (("IVQA_A", {"IVQA"}), ("IVQA+LVM_A", {"IVQA", "LVM"}), ("CIR_B", {"CIR"})):
        d_stream, d_wall, n_silent, n_pre = [], [], 0, 0
        for iid, it in items.items():
            if it.capability not in caps or it.should_remain_silent: continue
            if label != "CIR_B" and it.time_type != "A": continue
            qt = it.question_time_s if label != "CIR_B" else it.timed_answers[0].time_s
            ems = sorted(preds.get(iid, {}).get("emissions", []), key=lambda e: e["t"])
            post = [e for e in ems if e["t"] >= qt]
            n_pre += len(ems) - len(post)
            if not post: n_silent += 1; continue
            d_stream.append(post[0]["t"] - qt)
            ups = [u for u in raw[iid]["updates"] if (u.get("answer") or "").strip()]
            if ups and label != "CIR_B":
                d_wall.append(min(u["wall_ms_since_question"] for u in ups) / 1000.0)
        res[label] = {"first_emission_delay_stream_s": q(d_stream), "first_update_wall_s": q(d_wall),
                      "silent_items": n_silent, "pre_query_emissions": n_pre}
    # trigger items: delay of the matched emission of every event (no gate, as in the scorer)
    cfg = MetricConfig()
    for label, sel in (("PTR_INS", lambda it: it.capability == "PTR" and it.interaction_type == "INS"),
                       ("PTR_all", lambda it: it.capability == "PTR"),
                       ("TOA_QA_B", lambda it: it.capability == "TOA" and it.time_type == "B"),
                       ("CST_INS", lambda it: it.capability == "CST" and it.interaction_type == "INS"),
                       ("all_INS", lambda it: it.interaction_type == "INS" and it.capability not in ("LCG",)),
                       ("all_B_C_trigger", lambda it: it.time_type in ("B", "C") and it.capability not in ("LCG", "BRC"))):
        first, alld, miss, tot = [], [], 0, 0
        for iid, it in items.items():
            if not sel(it) or it.should_remain_silent: continue
            ev = merge_coincident(it.timed_answers); gt_times = [t for t, _ in ev]
            ems = sorted((e for e in preds.get(iid, {}).get("emissions", []) if (e.get("content") or "").strip()), key=lambda e: e["t"])
            dt = decision_timing(gt_times, ems, cfg)
            m = dt["matched_events"]; tot += len(gt_times); miss += len(gt_times) - len(m)
            ds = {n: m[n]["t"] - gt_times[n] for n in m}
            alld += list(ds.values())
            if 0 in ds: first.append(ds[0])
        res[label] = {"first_event_matched_delay_s": q(first), "all_matched_event_delay_s": q(alld),
                      "events": tot, "missed_events": miss}
    shift = res["IVQA_A"]["first_emission_delay_stream_s"]["median"]
    res["shift_s_used"] = shift
    print(json.dumps(res, indent=1, ensure_ascii=False))
    out = Path(args.out) / "reaction_offset.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("wrote", out)


# ------------------------------------------------------------------ shift-a

def cmd_shift_a(args) -> None:
    H, _, keep, items = _load_human(args)
    if args.shift is not None:
        shift = args.shift
    else:
        fp = args.reaction_offset or f"{args.out}/reaction_offset.json"
        shift = json.loads(Path(fp).read_text())["shift_s_used"]
    lines = []; nA = 0
    for l in (H / args.preds_name).read_text().splitlines():
        if not l.strip(): continue
        p = json.loads(l); iid = f'{p["video_id"]}#{p["item_index"]}'; it = items.get(iid)
        if it is not None and it.time_type == "A":
            nA += 1; qt = it.question_time_s
            p["emissions"] = sorted([{**e, "t": round(max(e["t"] - shift, qt, 0.0), 3), "t_orig": e["t"]} for e in p["emissions"]], key=lambda e: e["t"])
            p["shift_s"] = shift
        p["run"] = f"{args.human_run}_shiftedA"; lines.append(json.dumps(p, ensure_ascii=False))
    (H / "preds_shiftedA.jsonl").write_text("\n".join(lines) + "\n"); print("A-type items shifted:", nA, "shift", shift)


# ------------------------------------------------------------------ score

def cmd_score(args) -> None:
    from interactionbench.data import iter_items, load_benchmark
    from interactionbench.judges import make_judge
    from interactionbench.metrics import MetricConfig, aggregate, score_item

    pred_fp = Path(args.predictions)
    items_fp = args.items or str(pred_fp.parent / "items.txt")
    out = Path(args.eval_dir) if args.eval_dir else pred_fp.parent / args.eval_name

    judge = make_judge(args.judge, cache_path=args.judge_cache,
                       prompt_version=args.judge_prompt)
    print(f"judge: {judge.name} (cache: {args.judge_cache})", file=sys.stderr)

    key = {}
    for line in Path(args.mcq_key).read_text(encoding="utf-8").splitlines():
        if line.strip():
            k = json.loads(line)
            key[k["item_id"]] = k["answer_text"]

    def mcq_text_judge(answer_text: str):
        # same judge, same prompt; the reference is the text of the correct option
        # (the option matcher also ignores the annotated reference answer)
        return lambda question, gt, pred: judge(question, answer_text, pred)

    cfg = MetricConfig() if args.pre_tol is None else MetricConfig(pre_tol_s=args.pre_tol)
    if args.delta is not None:
        cfg.delta_s = args.delta
    videos = load_benchmark(args.data)
    keep = {l.strip() for l in Path(items_fp).read_text().splitlines() if l.strip()}
    items = {it.item_id: it for it in iter_items(videos) if it.item_id in keep}
    preds = {}
    for line in pred_fp.read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            preds[f"{p['video_id']}#{p.get('item_index', 0)}"] = p

    records, n_mcq = [], 0
    for iid, item in items.items():
        p = preds.get(iid) or {"emissions": []}
        if iid in key:
            n_mcq += 1
            item_judge = mcq_text_judge(key[iid])
        else:
            item_judge = judge
        rec = score_item(item, p.get("emissions") or [], cfg, judge=item_judge)
        rec["mcq"] = iid in key
        rec["mcq_scoring"] = "judge_vs_option_text" if iid in key else None
        for k in ("model", "run"):
            if k in p:
                rec[k] = p[k]
        records.append(rec)
    print(f"scored {len(records)} items ({n_mcq} MCQ judged against option text)", file=sys.stderr)
    if hasattr(judge, "n_missing"):
        print(f"judge cache misses: {judge.n_missing}/{judge.n_calls}", file=sys.stderr)

    agg = aggregate(records)
    agg["variant"] = {"mcq_scoring": "judge_vs_option_text", "judge": judge.name,
                      "n_mcq": n_mcq, "n_items": len(records), "pre_tol_s": cfg.pre_tol_s}
    agg["config"] = {"delta_s": cfg.delta_s, "gate_thresh": cfg.gate_thresh,
                     "use_gate": cfg.use_gate, "pre_tol_s": cfg.pre_tol_s,
                     "mcq_key": args.mcq_key, "items": items_fp, "judge": judge.name,
                     "judge_cache_misses": int(getattr(judge, "n_missing", 0) or 0)}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    o = agg["overall"]
    print(f"overall n={o['n_items']} total={o['total_score']} acc={o['accuracy']} "
          f"TA={o['timing_accuracy']} SC={o['silence_compliance']}")
    print(f"wrote {out}/records.jsonl and {out}/summary.json")


# ------------------------------------------------------------------ compare

def cmd_compare(args) -> None:
    HUMAN = human_dir(args)
    ev_nj, ev_j = args.eval_name, args.judge_eval_name
    ev_hj = args.human_judge_eval_name or ev_j
    items = [l.strip() for l in (HUMAN / "items.txt").read_text().splitlines() if l.strip()]
    iset = set(items)

    def load(p):
        recs = {}
        for l in Path(p).read_text(encoding="utf-8").splitlines():
            if not l.strip(): continue
            r = json.loads(l)
            if r["item_id"] in iset: recs[r["item_id"]] = r
        return recs

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    def agg(recs, keyfn):
        groups = collections.defaultdict(list)
        for r in recs.values(): groups[keyfn(r)].append(r)
        out = {}
        for g, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
            out[g] = {"n_items": len(rs), **{m: mean([r.get(m) for r in rs]) for m in METRICS}}
        return out

    def full(recs):
        return {"overall": {"n_items": len(recs), **{m: mean([r.get(m) for r in recs.values()]) for m in METRICS}},
                "by_capability": agg(recs, lambda r: r.get("capability")),
                "by_family": agg(recs, lambda r: r.get("family"))}

    res = {"items_file": str(HUMAN / "items.txt"), "n_items": len(items),
           "items_by_capability": dict(collections.Counter(json.loads(l)["capability"] for l in (HUMAN / ev_nj / "records.jsonl").read_text().splitlines() if l.strip())),
           "human": {}, "systems": {}}
    hrec = load(HUMAN / ev_nj / "records.jsonl")
    res["human"][ev_nj] = full(hrec)
    res["human"]["summary_json_overall"] = json.load(open(HUMAN / ev_nj / "summary.json"))["overall"]
    jd = HUMAN / ev_hj / "records.jsonl"
    if jd.exists(): res["human"][ev_hj] = full(load(jd))

    # multiple choice: answered by letter or by text (same letter pattern as the option scorer)
    key = {}
    for l in Path(args.mcq_key).read_text(encoding="utf-8").splitlines():
        if l.strip():
            k = json.loads(l); key[k["item_id"]] = k
    preds = {}
    for l in (HUMAN / "preds.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            p = json.loads(l); preds[f'{p["video_id"]}#{p["item_index"]}'] = p
    mcq_items = [i for i in items if i in key]
    by_letter, by_text, silent_mcq, article_a_risk, examples = 0, 0, 0, 0, []
    for i in mcq_items:
        k = key[i]; opts = list(k["distractors"]); opts.insert(k["correct_index"], k["answer_text"])
        letters = "ABCDEF"[:len(opts)]
        lre = re.compile(rf"(?:^|[^a-z0-9])([{letters}{letters.lower()}])(?:[^a-z0-9]|$)")
        ems = preds[i]["emissions"]
        if not ems: silent_mcq += 1; continue
        fired = [lre.search(e["content"] if len(e["content"]) <= 40 else e["content"][:40]) for e in ems]
        bare = [e["content"] for e, m in zip(ems, fired) if m and re.fullmatch(r"\s*\(?[A-Fa-f][\.\)]?\s*(.*)?", e["content"]) and len(e["content"].split()) <= 3]
        if any(fired):
            if bare: by_letter += 1
            else:
                article_a_risk += 1
                if len(examples) < 12:
                    examples.append({"item_id": i, "answer": ems[-1]["content"][:80], "regex_letter": next(m.group(1) for m in fired if m), "correct": k["correct_letter"], "answer_text": k["answer_text"]})
        else: by_text += 1
    res["human"]["mcq_diag"] = {"n_mcq_items_in_subset": len(mcq_items), "answered_by_bare_letter": by_letter,
                                "answered_by_text_only": by_text, "silent_mcq": silent_mcq,
                                "free_text_where_letter_regex_fires_on_a_stray_letter": article_a_risk,
                                "examples_stray_letter": examples}
    for _, s in parse_runs(args.runs, [(r, r) for r in SYSTEMS]):
        d = Path(args.runs_root) / s
        res["systems"][s] = {}
        for ev in [ev_j, ev_nj]:
            p = d / ev / "records.jsonl"
            if p.exists():
                r = load(p); res["systems"][s][ev] = full(r)
                res["systems"][s][ev]["overall"]["n_missing_in_records"] = len(iset) - len(r)
    out = Path(args.out) / "human_first300.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    # compact console table
    def row(name, o): return f'{name:48s} n={o["n_items"]:3d} Total={o["total_score"]} Acc={o["accuracy"]} TA={o["timing_accuracy"]} SC={o["silence_compliance"]}'
    print(row("HUMAN nojudge", res["human"][ev_nj]["overall"]))
    for s in res["systems"]:
        for ev, o in res["systems"][s].items(): print(row(f"{s} [{ev[:12]}]", o["overall"]))
    print("MCQ diag", json.dumps(res["human"]["mcq_diag"], ensure_ascii=False)[:1500])
    print("wrote", out)


# ------------------------------------------------------------------ cli

def _base(p: argparse.ArgumentParser) -> None:
    add_common_args(p)
    p.add_argument("--human-run", default=HUMAN_RUN,
                   help="name of the human run directory under --runs-root "
                        "(default: %(default)s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="tool export -> prediction schema")
    _base(p)
    p.add_argument("raw", help="export of the annotation tool (JSON)")
    p.add_argument("--run-dir", default=None,
                   help="output directory (default <runs_root>/<human_run>)")
    p.add_argument("--model", default="human")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("reaction-latency", help="measure the human reaction latency")
    _base(p)
    p.add_argument("--raw", default=None,
                   help="export of the annotation tool "
                        "(default <runs_root>/<human_run>/human_raw.json)")
    p.add_argument("--preds-name", default="preds.jsonl")
    p.set_defaults(func=cmd_reaction_latency)

    p = sub.add_parser("shift-a", help="shift A-type emissions back by the reaction latency")
    _base(p)
    p.add_argument("--shift", type=float, default=None,
                   help="seconds; default: shift_s_used of --reaction-offset "
                        "(2.48 for the human run of the paper)")
    p.add_argument("--reaction-offset", default=None,
                   help="output of reaction-latency (default <out>/reaction_offset.json)")
    p.add_argument("--preds-name", default="preds.jsonl")
    p.set_defaults(func=cmd_shift_a)

    p = sub.add_parser("score", help="score with multiple choice graded by option text")
    _base(p)
    p.add_argument("predictions")
    p.add_argument("--items", default=None,
                   help="file of item_ids (default items.txt next to the predictions)")
    p.add_argument("--judge", required=True, help="judge spec, e.g. hf:Qwen/Qwen3-14B")
    p.add_argument("--judge-cache", default="results/judge_cache/judge_cache.jsonl",
                   help="verdict file of the judge; keep one file per judge and prompt")
    p.add_argument("--judge-prompt", default="v2", help="grading prompt version: v1, v2")
    p.add_argument("--eval-name", default=f"{EVAL_JUDGE}_mcqtext",
                   help="evaluation directory name, created next to the predictions "
                        "(default: %(default)s)")
    p.add_argument("--eval-dir", default=None,
                   help="evaluation directory; overrides --eval-name")
    p.add_argument("--delta", type=float, default=None,
                   help="acceptable-delay bound Delta of the linear timing decay, "
                        "stream seconds (default 5.0)")
    p.add_argument("--pre-tol", type=float, default=None,
                   help="pre-anchor tolerance, seconds; default MetricConfig.pre_tol_s "
                        "(1.0); 'inf' selects the unbounded tolerance")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("compare", help="human run and systems on the same items")
    _base(p)
    p.add_argument("--runs", nargs="*", default=None,
                   help="system run names, or a file with one name per line "
                        "(default: the five systems of the paper)")
    p.add_argument("--eval-name", default=EVAL_NOJUDGE,
                   help="evaluation without a judge (default: %(default)s)")
    p.add_argument("--judge-eval-name", default=EVAL_JUDGE,
                   help="evaluation with the judge (default: %(default)s)")
    p.add_argument("--human-judge-eval-name", default=None,
                   help="judged evaluation of the human run (default: --judge-eval-name)")
    p.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    resolve_common(args)
    args.func(args)


if __name__ == "__main__":
    main()
