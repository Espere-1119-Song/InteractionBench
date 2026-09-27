#!/usr/bin/env python3
"""PairSel: matched-suite selectivity over the verified near-miss suites.

Each suite j pairs one positive item with one to three near-miss events on the same
video. A near-miss event is a moment that resembles the target event without
satisfying the request; it is given as a single timestamp t1 (seconds).

  h+_j = 1  iff the evaluation record of the positive item has timing_accuracy > 0
            (some event received nonzero timing credit).
  s-_j = 1  iff the run emits nothing (emissions with non-empty content only, as in
            the scorer, which drops blank emissions) inside any near-miss window of
            the suite.
  PairSel = 100/G * sum_j h+_j * s-_j,   G = suites whose positive item is in the run.

Near-miss windows: t1 is a point, so the primary window is symmetric,
[t1 - DELTA, t1 + DELTA] with DELTA = 5 s (= MetricConfig.delta_s). Two sensitivity
variants are also stored: "asym" = [t1 - 1, t1 + 5] and "excl_pos" = the symmetric
window, ignoring emissions that fall inside a positive response window [r*, r* + DELTA].

Suite file (``--suites``), one JSON object per line. Fields read by this script:

  suite_id           str    identifier of the suite
  item_id            str    positive item, "<video_id>#<item_index>"
  positive_windows   list   one object per reference event of the positive item:
                            earliest_response  float  reference response time r* (s)
                            evidence           str    reference text; its last number
                                                      is the running count (strict)
  near_miss          list   one object per near-miss event:
                            t1                 float  time of the near miss (s)

Sub-commands:

  score    PairSel of every run found under --runs-root. Timing of the positive item
           is read from <run>/<eval_name>/records.jsonl; runs named scripted_* fall
           back to <run>/<fallback_eval_name>/records.jsonl, because their timing does
           not depend on a judge.
  strict   event-level variants computed from the output of ``score``. They keep the
           near-miss condition and tighten the positive one:
             recall     at least half of the reference events of the positive item
                        are matched (records n_matched / n_gt);
             final      nonzero timing credit and the exact final count
                        (records final_count_abs_err == 0);
             event      at least half of the reference events receive a matched reply
                        that states the correct running count; event_all requires
                        every event. A reply is matched to event n when it is the
                        first non-empty emission in [r_n - 1, r_{n+1} - 1), the last
                        window running to the end, and its count is the last number
                        it states, in digits or in words up to twenty.

Upstream repository or checkpoint: none. Environment: Python >= 3.10; no GPU. This
script reads files only and does not import the scorer.

Commands used for the paper numbers (evaluations with pre-anchor tolerance 1 s):

  python analysis/pairsel.py score --eval-name eval_judge_pretol1 \
      --fallback-eval-name eval_nojudge_pretol1 --suffix _pretol1
  python analysis/pairsel.py strict --pairsel results/analysis/pairsel_pretol1.json

Outputs:
  <out>/pairsel<suffix>.json     definition, evaluation directories, one entry per run
                                 with the per-suite rows
  <out>/pairsel_strict.json      one entry per run with the strict variants
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import EVAL_JUDGE, EVAL_NOJUDGE, add_common_args, resolve_common  # noqa: E402

DELTA = 5.0
LEAD = 1.0
SUITES = "suites/nearmiss_verified.jsonl"


def load_suites(path: Path) -> list[dict]:
    suites = [json.loads(l) for l in path.open() if l.strip()]
    for s in suites:
        s["_pos_times"] = sorted(w["earliest_response"] for w in s["positive_windows"])
        s["_nm"] = [d["t1"] for d in s["near_miss"]]
    return suites


def load_preds(fp: Path) -> dict[str, list[float]]:
    """item_id -> sorted emission times with non-empty content."""
    out: dict[str, list[float]] = {}
    with fp.open() as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            iid = f"{p['video_id']}#{p['item_index']}"
            ts = [e["t"] for e in p.get("emissions") or []
                  if (e.get("content") or "").strip() and e.get("t") is not None]
            out[iid] = sorted(ts)
    return out


def load_records(fp: Path) -> dict[str, dict]:
    out = {}
    with fp.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["item_id"]] = r
    return out


def silent(ts: list[float], windows: list[tuple[float, float]],
           exclude: list[tuple[float, float]] | None = None) -> bool:
    for t in ts:
        if exclude and any(a <= t <= b for a, b in exclude):
            continue
        if any(a <= t <= b for a, b in windows):
            return False
    return True


def score_run(run_dir: Path, suites: list[dict], eval_judge: str,
              eval_nojudge: str) -> dict | None:
    preds_fp = run_dir / "preds.jsonl"
    if not preds_fp.exists():
        return None
    rec_fp = run_dir / eval_judge
    eval_used = eval_judge
    if not rec_fp.exists():
        if run_dir.name.startswith("scripted_") and (run_dir / eval_nojudge).exists():
            rec_fp, eval_used = run_dir / eval_nojudge, eval_nojudge
        else:
            return None
    preds = load_preds(preds_fp)
    recs = load_records(rec_fp)

    rows = []
    for s in suites:
        iid = s["item_id"]
        if iid not in preds or iid not in recs:
            continue
        ts = preds[iid]
        ta = recs[iid].get("timing_accuracy")
        hit = 1 if (ta is not None and ta > 0) else 0
        sym = [(t1 - DELTA, t1 + DELTA) for t1 in s["_nm"]]
        asym = [(t1 - LEAD, t1 + DELTA) for t1 in s["_nm"]]
        pos = [(r, r + DELTA) for r in s["_pos_times"]]
        rows.append({
            "suite_id": s["suite_id"], "hit": hit, "timing_accuracy": ta,
            "n_emissions": len(ts),
            "silent_sym": int(silent(ts, sym)),
            "silent_asym": int(silent(ts, asym)),
            "silent_excl_pos": int(silent(ts, sym, exclude=pos)),
        })
    G = len(rows)
    if G == 0:
        return None

    def mean(key):
        return 100.0 * sum(r[key] for r in rows) / G

    def pairsel(key):
        return 100.0 * sum(r["hit"] * r[key] for r in rows) / G

    return {
        "run": run_dir.name, "run_root": run_dir.parent.name,
        "eval_used": eval_used, "n_items_in_run": len(preds),
        "coverage_G": G, "full_coverage": G == len(suites),
        "pairsel": round(pairsel("silent_sym"), 1),
        "hit_rate": round(mean("hit"), 1),
        "silence_rate": round(mean("silent_sym"), 1),
        "variants": {
            "asym_window": {"pairsel": round(pairsel("silent_asym"), 1),
                            "silence_rate": round(mean("silent_asym"), 1)},
            "sym_excl_pos": {"pairsel": round(pairsel("silent_excl_pos"), 1),
                             "silence_rate": round(mean("silent_excl_pos"), 1)},
        },
        "per_suite": rows,
    }


def cmd_score(args) -> None:
    eval_judge = f"{args.eval_name}/records.jsonl"
    eval_nojudge = f"{args.fallback_eval_name}/records.jsonl"
    out_fp = args.out_file or f"{args.out}/pairsel{args.suffix}.json"
    suites = load_suites(Path(args.suites))
    n_nm = sum(len(s["_nm"]) for s in suites)
    overlap = sum(1 for s in suites for t1 in s["_nm"]
                  if any(r - DELTA <= t1 <= r + DELTA for r in s["_pos_times"]))
    print(f"{len(suites)} suites, {n_nm} near-miss events; "
          f"{overlap} near-miss windows (+/-{DELTA}s) overlap a positive r* (+/-{DELTA}s)")

    skip = re.compile(args.skip_re)
    results = []
    for root in args.runs_root:
        root = Path(root)
        if not root.is_dir():
            print(f"note: {root} is not a directory, skipped", file=sys.stderr)
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if skip.search(d.name):
                continue
            r = score_run(d, suites, eval_judge, eval_nojudge)
            if r:
                results.append(r)
    results.sort(key=lambda r: (-r["full_coverage"], -r["pairsel"], -r["hit_rate"]))

    hdr = f"{'run':58s} {'PairSel':>7s} {'hit':>6s} {'silent':>7s} {'G':>3s}  {'asym':>6s} {'exclP':>6s} eval"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        flag = "" if r["full_coverage"] else " *"
        print(f"{r['run']:58s} {r['pairsel']:7.1f} {r['hit_rate']:6.1f} {r['silence_rate']:7.1f} "
              f"{r['coverage_G']:3d}{flag:2s} {r['variants']['asym_window']['pairsel']:6.1f} "
              f"{r['variants']['sym_excl_pos']['pairsel']:6.1f} {r['eval_used'].split('/')[0]}")
    print("* = positive item of some suites missing from the run (partial coverage)")

    out = Path(out_fp)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "definition": {
            "suite_file": str(args.suites), "n_suites": len(suites),
            "n_near_miss_events": n_nm,
            "near_miss_window": f"[t1 - {DELTA}, t1 + {DELTA}] (symmetric, DELTA = MetricConfig.delta_s)",
            "variants": {"asym_window": f"[t1 - {LEAD}, t1 + {DELTA}]",
                         "sym_excl_pos": f"symmetric window, ignoring emissions inside any positive [r*, r* + {DELTA}]"},
            "hit": "positive item timing_accuracy > 0 in the eval records",
            "silence": "no non-empty emission inside any near-miss window of the suite",
            "pairsel": "100/G * sum_j hit_j * silent_j",
            "n_nearmiss_windows_overlapping_positive": overlap,
        },
        "eval_dirs": {"judge": eval_judge, "nojudge_fallback": eval_nojudge},
        "runs": results,
    }
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"wrote {out}")


# ------------------------------------------------------------------ strict variants

WORDS = {w: i for i, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve '
                                    'thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
EPS = 1.0


def last_number(text):
    for t in reversed(re.findall(r'\d+|[A-Za-z]+', text or '')):
        if t.isdigit():
            return int(t)
        if t.lower() in WORDS:
            return WORDS[t.lower()]
    return None


def event_correct(suite, emissions):
    ev = sorted((w['earliest_response'], last_number(str(w['evidence']))) for w in suite['positive_windows'])
    ok = 0
    for n, (r, c) in enumerate(ev):
        lo = r - EPS
        hi = ev[n + 1][0] - EPS if n + 1 < len(ev) else float('inf')
        first = next((e for e in emissions if lo <= e['t'] < hi), None)
        ok += bool(first) and last_number(first['content']) == c
    return ok, len(ev)


def cmd_strict(args) -> None:
    ps = json.loads(Path(args.pairsel or f"{args.out}/pairsel{args.suffix}.json").read_text())
    suites = {}
    for line in Path(args.suites).read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            suites[s['suite_id']] = s
    roots = {Path(r).name: Path(r) for r in args.runs_root}

    out = []
    for r in ps['runs']:
        rd = roots[r['run_root']] / r['run']
        recs = {}
        fp = rd / r['eval_used']
        if fp.exists():
            for line in fp.read_text().splitlines():
                if line.strip():
                    x = json.loads(line)
                    recs[x['item_id']] = x
        preds = {}
        for line in (rd / 'preds.jsonl').read_text().splitlines():
            if line.strip():
                p = json.loads(line)
                preds[f"{p['video_id']}#{p['item_index']}"] = sorted(
                    (e for e in p.get('emissions') or [] if (e.get('content') or '').strip() and e.get('t') is not None),
                    key=lambda e: e['t'])
        rows = r['per_suite']
        tot = {'base': 0, 'recall': 0, 'final': 0, 'event': 0, 'event_all': 0}
        for row in rows:
            s = suites[row['suite_id']]
            x = recs.get(s['item_id'], {})
            sil = row['silent_sym']
            ng = x.get('n_gt') or len(s['positive_windows'])
            ok, n = event_correct(s, preds.get(s['item_id'], []))
            tot['base'] += row['hit'] * sil
            tot['recall'] += ((x.get('n_matched') or 0) / ng >= 0.5) * sil
            tot['final'] += row['hit'] * (x.get('final_count_abs_err') == 0) * sil
            tot['event'] += (ok / n >= 0.5) * sil
            tot['event_all'] += (ok == n) * sil
        G = len(rows)
        out.append({'run': r['run'], 'G': G, 'full': r.get('full_coverage'), 'reported': r['pairsel'],
                    **{k: round(100 * v / G, 1) for k, v in tot.items()}})
    out_fp = Path(args.out_file or f"{args.out}/pairsel_strict.json")
    out_fp.write_text(json.dumps(out, indent=1))
    print(f"{'run':52s} {'G':>3s} {'rep':>5s} {'base':>5s} {'recall':>6s} {'final':>5s} {'event':>5s} {'all':>5s}")
    for o in sorted(out, key=lambda o: (-bool(o['full']), -o['base'])):
        print(f"{o['run'][:52]:52s} {o['G']:3d} {o['reported']:5.1f} {o['base']:5.1f} {o['recall']:6.1f} "
              f"{o['final']:5.1f} {o['event']:5.1f} {o['event_all']:5.1f}")
    print(f"wrote {out_fp}")


def _shared(p: argparse.ArgumentParser) -> None:
    add_common_args(p, multi_root=True)
    p.add_argument("--suites", default=None,
                   help="suite file (default <data>/" + SUITES + ")")
    p.add_argument("--suffix", default="",
                   help="suffix of the default output / input file name, e.g. _pretol1")
    p.add_argument("--out-file", default=None, help="output file; overrides --out")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("score", help="PairSel of every run under --runs-root")
    _shared(p)
    p.add_argument("--eval-name", default=EVAL_JUDGE,
                   help="evaluation directory with the judged records (default: %(default)s)")
    p.add_argument("--fallback-eval-name", default=EVAL_NOJUDGE,
                   help="evaluation directory used for scripted_* runs when --eval-name "
                        "is absent (default: %(default)s)")
    p.add_argument("--skip-re", default=r"_s\dof\d$|shard|_tail|^_smoke_|_lenientparse$",
                   help="run-directory name pattern to skip (shards, smoke tests)")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("strict", help="event-level variants from the output of 'score'")
    _shared(p)
    p.add_argument("--pairsel", default=None,
                   help="output file of 'score' (default <out>/pairsel<suffix>.json)")
    p.set_defaults(func=cmd_strict)

    args = ap.parse_args()
    resolve_common(args)
    if args.suites is None:
        args.suites = f"{args.data}/{SUITES}"
    args.func(args)


if __name__ == "__main__":
    main()
