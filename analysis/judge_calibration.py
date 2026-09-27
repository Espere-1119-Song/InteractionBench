#!/usr/bin/env python3
"""Judge calibration: candidate open judges against stored reference verdicts.

A judge grades a (question, reference answer, model answer) triple with 0 or 1. The
reference labels of this calibration are the binary verdicts of an earlier reference
judge (Gemini), stored with two runs as verdict files keyed by
sha256(question, reference, answer). The field that holds them is named ``gemini``.

Sub-commands:

  extract   Rebuild the triples. The verdict files store only the hash of a triple, so
            the triples are recovered by scoring the predictions of the runs with a
            recording judge. Writes <calib_dir>/triples.jsonl with
            {k, question, gt, pred, gemini, run}; ``gemini`` is null when the triple
            has no reference verdict.
  run       Score the labelled triples with each candidate judge (one verdict file per
            candidate) and write agreement, Cohen's kappa, true positive and true
            negative rate, agreement per run and seconds per call.
  votes     For one local Hugging Face judge: one greedy verdict and K sampled verdicts
            per labelled triple (default K = 5, temperature 0.7, top-p 0.95, at most
            384 new tokens). Needs a GPU. Resumable.
  analyze   The majority-vote table: per judge the agreement and kappa of the greedy
            verdict and of the majority of K votes (a tie falls back to the greedy
            verdict), the share of triples with unanimous votes, kappa between judges,
            and ensembles over judges. Only vote files with at least 900 rows are
            used, and only triples present in all of them.
  tables    Print, for every run with a judged evaluation, the per-task rows of its
            summary, and the mean scores of selected full-set runs restricted to an
            item subset.

Judges are built with ``interactionbench.judges.make_judge``; see that module for the
spec syntax (hf:<model_id>, api:<model>, ...).

Upstream checkpoints: the candidate judges are loaded from the Hugging Face Hub by
model id, e.g. https://huggingface.co/Qwen/Qwen3-14B (the paper judge). API judges
read their key from the environment variable named by the judge
(IBENCH_JUDGE_API_KEY by default).
Environment: Python >= 3.10 and the ``interactionbench`` package; ``run`` and ``votes``
with an ``hf:`` judge also need torch and transformers and a GPU.

Commands used for the paper numbers (grading prompt v2):

  python analysis/judge_calibration.py extract
  python analysis/judge_calibration.py run hf:Qwen/Qwen3-14B --judge-prompt v2
  python analysis/judge_calibration.py votes hf:Qwen/Qwen3-14B --judge-prompt v2
  python analysis/judge_calibration.py analyze
  python analysis/judge_calibration.py tables --eval-name eval_judge

Outputs (calib_dir defaults to <out>/judge_calib):
  <calib_dir>/triples.jsonl
  <calib_dir>/cache_<slug>[_prompt<version>].jsonl      verdicts of a candidate
  <calib_dir>/report_<slug>.json, disagree_<slug>.jsonl
  <calib_dir>/votes_<slug>_prompt<version>.jsonl
  <calib_dir>/vote_table.csv, vote_report.md
  <out>/judge_vote_table.tex
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import re
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import EVAL_JUDGE, add_common_args, parse_runs, resolve_common  # noqa: E402

EXTRACT_RUNS = ["videollm-online-8b_streaming_8fps_mcq", "joyai_streaming_4fps_mcqv4"]
LABEL_CACHES = "eval/judge_cache_final_*.jsonl"
TABLE_RUNS = ["qwen3vl-8b_sliding_iv1_mcqv4_full", "qwen3vl-8b_sliding_win64_iv1_mcqv4_full",
              "qwen3vl-8b_sliding_fps05_iv1_mcqv4_full"]
MIN_VOTE_ROWS = 900


def slug_of(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", spec.split(":", 1)[-1])


# ------------------------------------------------------------------ extract

class Recorder:
    def __init__(self, cache): self.cache = cache; self.rows = {}
    def __call__(self, q, gt, pred):
        from interactionbench.judges.base import cache_key
        gt, pred = (gt or "").strip(), (pred or "").strip()
        if not gt or not pred:
            return 1.0 if (not gt and not pred) else 0.0
        k = cache_key(q or "", gt, pred)
        if k not in self.rows:
            self.rows[k] = {"k": k, "question": q or "", "gt": gt, "pred": pred, "gemini": self.cache.get(k)}
        s = self.cache.get(k)
        return s if s is not None else 0.0


def cmd_extract(args) -> None:
    from interactionbench.data import iter_items, load_benchmark
    from interactionbench.metrics import MetricConfig, score_item

    items = {it.item_id: it for it in iter_items(load_benchmark(args.data))}
    cfg = MetricConfig()
    out = Path(args.calib_dir) / "triples.jsonl"; n_all = n_lab = 0
    with out.open("w") as f:
        for _, run in parse_runs(args.runs, [(r, r) for r in EXTRACT_RUNS]):
            cache = {}
            for p in sorted(glob.glob(f"{args.runs_root}/{run}/{args.label_caches}")):
                for l in Path(p).read_text().splitlines():
                    if l.strip():
                        d = json.loads(l); cache.setdefault(d["k"], d["s"])
            rec = Recorder(cache)
            for l in Path(f"{args.runs_root}/{run}/preds.jsonl").read_text().splitlines():
                if not l.strip(): continue
                p = json.loads(l); iid = f"{p['video_id']}#{p.get('item_index', 0)}"
                if iid not in items: continue
                extra = {k: p[k] for k in ("n_polls", "poll_latencies") if k in p}
                score_item(items[iid], p.get("emissions") or [], cfg, judge=rec, extra=extra or None)
            for r in rec.rows.values():
                r["run"] = run; f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_all += 1; n_lab += r["gemini"] is not None
            print(f"{run}: cache={len(cache)} triples={len(rec.rows)} labeled={sum(r['gemini'] is not None for r in rec.rows.values())}")
    print(f"total triples {n_all}, with gemini label {n_lab} -> {out}")


# ------------------------------------------------------------------ run

def cmd_run(args) -> None:
    from interactionbench.judges import make_judge

    cd = args.calib_dir
    rows = [json.loads(l) for l in Path(f"{cd}/triples.jsonl").read_text().splitlines() if l.strip()]
    lab = [r for r in rows if r.get("gemini") is not None]
    print(f"labeled triples: {len(lab)} (pos={sum(r['gemini']==1.0 for r in lab)})", flush=True)

    for spec in args.specs:
        slug = slug_of(spec)
        pv = args.judge_prompt
        if pv != "v1":
            slug += f"_prompt{pv}"
        judge = make_judge(spec, cache_path=f"{cd}/cache_{slug}.jsonl", prompt_version=pv)
        t0 = time.time(); n_calls0 = judge.n_calls
        pred = []
        for i, r in enumerate(lab):
            pred.append(judge(r["question"], r["gt"], r["pred"]))
            if (i + 1) % 100 == 0:
                print(f"  [{slug}] {i+1}/{len(lab)} elapsed {time.time()-t0:.0f}s", flush=True)
        dt = time.time() - t0; calls = judge.n_calls - n_calls0
        y = [r["gemini"] for r in lab]
        tp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 1); tn = sum(1 for a, b in zip(pred, y) if a == 0 and b == 0)
        fp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 0); fn = sum(1 for a, b in zip(pred, y) if a == 0 and b == 1)
        n = len(y); acc = (tp + tn) / n
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
        kappa = (acc - pe) / (1 - pe) if pe < 1 else 0.0
        per_run = {}
        for run in sorted({r["run"] for r in lab}):
            idx = [i for i, r in enumerate(lab) if r["run"] == run]
            per_run[run] = round(sum(pred[i] == y[i] for i in idx) / len(idx), 4)
        rep = {"judge": spec, "n": n, "agreement": round(acc, 4), "kappa": round(kappa, 4),
               "tpr": round(tp / (tp + fn), 4) if tp + fn else None, "tnr": round(tn / (tn + fp), 4) if tn + fp else None,
               "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, "per_run_agreement": per_run,
               "judge_positive_rate": round(sum(pred) / n, 4), "gemini_positive_rate": round(sum(y) / n, 4),
               "sec_per_call": round(dt / calls, 3) if calls else None, "uncached_calls": calls}
        Path(f"{cd}/report_{slug}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print(json.dumps(rep, ensure_ascii=False), flush=True)
        # disagreements, for inspection
        with Path(f"{cd}/disagree_{slug}.jsonl").open("w") as f:
            for p, r in zip(pred, lab):
                if p != r["gemini"]:
                    f.write(json.dumps({**{k: r[k] for k in ("question", "gt", "pred", "run")}, "gemini": r["gemini"], "judge": p}, ensure_ascii=False) + "\n")
        del judge
        try:  # release GPU memory before the next candidate
            import gc
            import torch
            gc.collect(); torch.cuda.empty_cache()
        except ImportError:
            pass


# ------------------------------------------------------------------ votes

def cmd_votes(args) -> None:
    import torch
    from interactionbench.judges import make_judge
    from interactionbench.judges.base import parse_score

    cd = args.calib_dir
    slug = slug_of(args.spec)
    pv = args.judge_prompt
    out = Path(f"{cd}/votes_{slug}_prompt{pv}.jsonl")
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip(): done.add(json.loads(l)["k"])
    rows = [json.loads(l) for l in Path(f"{cd}/triples.jsonl").read_text().splitlines() if l.strip()]
    lab = [r for r in rows if r.get("gemini") is not None and r["k"] not in done]
    print(f"{args.spec}: {len(lab)} triples to do ({len(done)} already)", flush=True)
    if not lab:
        sys.exit(0)

    # a local Hugging Face judge exposes .model and .processor
    judge = make_judge(args.spec, cache_path=f"{cd}/cache_{slug}_prompt{pv}.jsonl", prompt_version=pv)
    model, proc = judge.model, judge.processor

    def build_inputs(prompt):
        messages = [{"role": "user", "content": prompt}]
        for tpl_kw in ({"enable_thinking": False, "reasoning_effort": "low"}, {"enable_thinking": False}, {"reasoning_effort": "low"}, {}):
            try:
                return proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                return_dict=True, return_tensors="pt", **tpl_kw).to(model.device)
            except (TypeError, ValueError):
                continue
        raise RuntimeError("chat template failed")

    def decode(out_ids, in_len):
        texts = proc.batch_decode(out_ids[:, in_len:], skip_special_tokens=True)
        res = []
        for t in texts:
            if "</think>" in t: t = t.rsplit("</think>", 1)[1]
            if "final" in t and "assistantfinal" in t.replace(" ", ""):  # channel markers left by gpt-oss
                t = t.split("final", 1)[-1]
            res.append(t.strip())
        return res

    t0 = time.time()
    with out.open("a") as f, torch.inference_mode():
        for i, r in enumerate(lab):
            prompt = judge.prompt.format(question=r["question"] or "(none)", gt=r["gt"], pred=r["pred"])
            inputs = build_inputs(prompt); in_len = inputs["input_ids"].shape[1]
            g = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            raw_greedy = decode(g, in_len)[0]
            if args.sequential_votes:
                votes = []
                for _ in range(args.votes):
                    s = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                                       temperature=args.temperature, top_p=args.top_p)
                    votes.append(parse_score(decode(s, in_len)[0]))
            else:
                s = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                                   temperature=args.temperature, top_p=args.top_p, num_return_sequences=args.votes)
                votes = [parse_score(t) for t in decode(s, in_len)]
            f.write(json.dumps({"k": r["k"], "run": r["run"], "gemini": r["gemini"], "greedy": parse_score(raw_greedy),
                                "votes": votes, "raw_greedy": raw_greedy[:200]}, ensure_ascii=False) + "\n"); f.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(lab)}  {time.time()-t0:.0f}s", flush=True)
    print(f"done {len(lab)} in {time.time()-t0:.0f}s -> {out}", flush=True)


# ------------------------------------------------------------------ analyze

def kappa(a, b):
    n = len(a); acc = sum(x == y for x, y in zip(a, b)) / n
    pa = sum(a) / n; pb = sum(b) / n; pe = pa * pb + (1 - pa) * (1 - pb)
    return acc, (acc - pe) / (1 - pe) if pe < 1 else 0.0


def maj(v):
    return 1.0 if sum(v) * 2 > len(v) else 0.0 if sum(v) * 2 < len(v) else None


def cmd_analyze(args) -> None:
    cd = args.calib_dir
    judges = {}
    for fp in sorted(glob.glob(f"{cd}/votes_*_prompt*.jsonl")):
        name = re.sub(r"^votes_|_prompt.*$", "", os.path.basename(fp)[:-6])
        rows = {json.loads(l)["k"]: json.loads(l) for l in open(fp) if l.strip()}
        if len(rows) >= MIN_VOTE_ROWS:
            judges[name] = rows
    common = set.intersection(*(set(r) for r in judges.values())) if judges else set()
    common = sorted(common)
    print(f"judges: {list(judges)}; common triples: {len(common)}")
    gem = [judges[next(iter(judges))][k]["gemini"] for k in common]

    table = []
    for name, rows in judges.items():
        greedy = [rows[k]["greedy"] for k in common]
        votes = [rows[k]["votes"] for k in common]
        m = [maj(v) if maj(v) is not None else rows[k]["greedy"] for v, k in zip(votes, common)]  # tie -> greedy
        a_g, k_g = kappa(greedy, gem); a_m, k_m = kappa(m, gem)
        unanimous = sum(len(set(v)) == 1 for v in votes) / len(votes)
        table.append({"judge": name, "greedy_agree": a_g, "greedy_kappa": k_g, "maj_agree": a_m, "maj_kappa": k_m,
                      "unanimous": unanimous, "pos_rate": sum(m) / len(m)})
    # ensembles
    names = list(judges)
    mj = {n: [maj(judges[n][k]["votes"]) if maj(judges[n][k]["votes"]) is not None else judges[n][k]["greedy"] for k in common] for n in names}
    if len(names) >= 3:
        ens_judges = [maj([mj[n][i] for n in names]) for i in range(len(common))]
        ens_judges = [e if e is not None else mj[names[0]][i] for i, e in enumerate(ens_judges)]
        a, kp = kappa(ens_judges, gem)
        table.append({"judge": f"ENSEMBLE majority over {len(names)} judges (each majority-of-K)", "greedy_agree": None, "greedy_kappa": None, "maj_agree": a, "maj_kappa": kp, "unanimous": None, "pos_rate": sum(ens_judges) / len(ens_judges)})
        allv = [maj(list(itertools.chain.from_iterable(judges[n][k]["votes"] for n in names))) for k in common]
        allv = [e if e is not None else gem[i] * 0 for i, e in enumerate(allv)]
        a, kp = kappa(allv, gem)
        table.append({"judge": f"ENSEMBLE majority over all votes ({len(names)} judges x K)", "greedy_agree": None, "greedy_kappa": None, "maj_agree": a, "maj_kappa": kp, "unanimous": None, "pos_rate": sum(allv) / len(allv)})
    # ensemble over the three judges with the highest greedy kappa
    single = [r for r in table if not r["judge"].startswith("ENSEMBLE")]
    top3 = [r["judge"] for r in sorted(single, key=lambda r: -r["greedy_kappa"])[:3]]
    if len(top3) == 3:
        e3 = [maj([mj[n][i] for n in top3]) for i in range(len(common))]
        e3 = [e if e is not None else mj[top3[0]][i] for i, e in enumerate(e3)]
        a, kp = kappa(e3, gem)
        table.append({"judge": "ENSEMBLE top-3 by kappa (" + ", ".join(t.split("_", 1)[-1] for t in top3) + ")", "greedy_agree": None, "greedy_kappa": None, "maj_agree": a, "maj_kappa": kp, "unanimous": None, "pos_rate": sum(e3) / len(e3)})
        allv3 = [maj(list(itertools.chain.from_iterable(judges[n][k]["votes"] for n in top3))) for k in common]
        allv3 = [e if e is not None else mj[top3[0]][i] for i, e in enumerate(allv3)]
        a, kp = kappa(allv3, gem)
        table.append({"judge": "ENSEMBLE top-3, majority over all votes (3 x K)", "greedy_agree": None, "greedy_kappa": None, "maj_agree": a, "maj_kappa": kp, "unanimous": None, "pos_rate": sum(allv3) / len(allv3)})
    if len(top3) == 3:  # one greedy verdict per judge, majority of 3 (3 calls per triple)
        g3 = [maj([judges[n][k]["greedy"] for n in top3]) for k in common]
        a, kp = kappa(g3, gem)
        table.append({"judge": "ENSEMBLE top-3, majority of greedy verdicts (3 calls/triple)", "greedy_agree": None, "greedy_kappa": None, "maj_agree": a, "maj_kappa": kp, "unanimous": None, "pos_rate": sum(g3) / len(g3)})
    # inter-judge kappa
    pairs = [(a, b, kappa(mj[a], mj[b])[1]) for a, b in itertools.combinations(names, 2)]

    with open(f"{cd}/vote_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
    fmt = lambda x: "--" if x is None else f"{100*x:.1f}" if abs(x) <= 1 and "kappa" not in str(x) else f"{x:.3f}"
    with open(f"{cd}/vote_report.md", "w") as f:
        f.write(f"# Judge voting calibration (n={len(common)} gemini-labelled triples)\n\n| judge | greedy agree | greedy kappa | maj-of-K agree | maj-of-K kappa | unanimous | pos rate |\n|---|---|---|---|---|---|---|\n")
        for r in table:
            gk = "--" if r["greedy_kappa"] is None else f"{r['greedy_kappa']:.3f}"
            f.write(f"| {r['judge']} | {fmt(r['greedy_agree'])} | {gk} | {fmt(r['maj_agree'])} | {r['maj_kappa']:.3f} | {fmt(r['unanimous'])} | {fmt(r['pos_rate'])} |\n")
        f.write("\n## Inter-judge kappa (majority-of-K verdicts)\n\n")
        for a, b, kp in pairs: f.write(f"- {a} vs {b}: {kp:.3f}\n")
    with open(f"{args.out}/judge_vote_table.tex", "w") as f:
        f.write("% auto-generated by judge_calibration.py analyze — judge calibration vs gemini-binary labels\n\\begin{tabular}{lrrrrr}\n\\toprule\nJudge & Greedy agree & Greedy $\\kappa$ & Maj-of-K agree & Maj-of-K $\\kappa$ & Unanimous \\\\\n\\midrule\n")
        for r in table:
            cells = [fmt(r['greedy_agree']), '--' if r['greedy_kappa'] is None else f"{r['greedy_kappa']:.3f}", fmt(r['maj_agree']), f"{r['maj_kappa']:.3f}", fmt(r['unanimous'])]
            f.write(f"{r['judge'].replace('_', chr(92)+'_')} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(open(f"{cd}/vote_report.md").read())


# ------------------------------------------------------------------ tables

def cmd_tables(args) -> None:
    ev = args.eval_name
    print("== by_capability (judge v2) ==")
    shown_keys = False
    for base in args.runs_root:
        for d in sorted(glob.glob(f"{base}/*")):
            fp = f"{d}/{ev}/summary.json"
            if not os.path.exists(fp): continue
            s = json.load(open(fp)); o = s["overall"]; bc = s.get("by_capability", {})
            if not shown_keys:
                print("keys:", list(bc.keys()))
                k0 = next(iter(bc)); print("fields:", list(bc[k0].keys())); shown_keys = True
            row = [os.path.basename(d), o.get("n_items"), o.get("total_score")]
            for k in bc:
                v = bc[k]; row += [k, v.get("n_items"), v.get("accuracy"), v.get("timing_accuracy"), v.get("silence_compliance")]
            print("ROW", json.dumps(row))
    print("== sub103 baseline from judge records ==")
    ids = [l.strip() for l in open(args.items) if l.strip()]
    print("n_ids", len(ids), "example", ids[:2])
    for _, run in parse_runs(args.runs, [(r, r) for r in TABLE_RUNS]):
        fp = next((f"{base}/{run}/{ev}/records.jsonl" for base in args.runs_root
                   if os.path.exists(f"{base}/{run}/{ev}/records.jsonl")), None)
        if fp is None: print(run, "no records"); continue
        recs = [json.loads(l) for l in open(fp)]
        if run.endswith("sliding_iv1_mcqv4_full"): print("record keys:", list(recs[0].keys())[:30])
        idset = set(ids)
        def rid(r):
            for k in ("item_id", "id", "interaction_id", "qid", "uid"):
                if k in r: return str(r[k])
            return None
        sub = [r for r in recs if rid(r) in idset]
        def mean(key):
            vals = [r[key] for r in sub if r.get(key) is not None]
            return (round(st.mean(vals), 2), len(vals)) if vals else None
        print(run, "matched", len(sub), {k: mean(k) for k in ("total_score", "accuracy", "timing_accuracy", "silence_compliance")})


# ------------------------------------------------------------------ cli

def _base(p: argparse.ArgumentParser, multi_root: bool = False) -> None:
    add_common_args(p, multi_root=multi_root)
    p.add_argument("--calib-dir", default=None,
                   help="working directory of the calibration (default <out>/judge_calib)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="rebuild the (question, reference, prediction) triples")
    _base(p)
    p.add_argument("--runs", nargs="*", default=None,
                   help="run names, or a file with one name per line "
                        "(default: " + " ".join(EXTRACT_RUNS) + ")")
    p.add_argument("--label-caches", default=LABEL_CACHES,
                   help="glob, relative to the run directory, of the verdict files of the "
                        "reference judge; files are read in sorted order and the first "
                        "verdict of a triple is kept (default: %(default)s)")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("run", help="score the labelled triples with candidate judges")
    _base(p)
    p.add_argument("specs", nargs="+", help="judge specs, e.g. hf:Qwen/Qwen3-14B")
    p.add_argument("--judge-prompt", default="v2", help="grading prompt version: v1, v2")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("votes", help="greedy and sampled verdicts of one local judge (GPU)")
    _base(p)
    p.add_argument("spec", help="judge spec of a local model, e.g. hf:Qwen/Qwen3-14B")
    p.add_argument("--judge-prompt", default="v2", help="grading prompt version: v1, v2")
    p.add_argument("--votes", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--sequential-votes", action="store_true",
                   help="one sequence per generate call (for architectures that do not "
                        "fit K parallel sequences in memory)")
    p.set_defaults(func=cmd_votes)

    p = sub.add_parser("analyze", help="agreement, kappa and the majority-vote table")
    _base(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("tables", help="print per-task rows of the judged evaluations")
    _base(p, multi_root=True)
    p.add_argument("--eval-name", default=EVAL_JUDGE,
                   help="evaluation directory name (default: %(default)s)")
    p.add_argument("--runs", nargs="*", default=None,
                   help="runs of the subset baseline (default: " + " ".join(TABLE_RUNS) + ")")
    p.add_argument("--items", default="benchmark/splits/subset103.txt",
                   help="item subset of the baseline (default: %(default)s)")
    p.set_defaults(func=cmd_tables)

    args = ap.parse_args()
    resolve_common(args)
    if args.calib_dir is None:
        args.calib_dir = f"{args.out}/judge_calib"
    os.makedirs(args.calib_dir, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
