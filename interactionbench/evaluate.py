"""Score a predictions file against the benchmark annotations.

Predictions: one JSON object per line, one line per item:
  {"video_id": "...", "item_index": 0,
   "emissions": [{"t": 12.3, "content": "...", "latency_s": 0.15}, ...],
   "n_polls": 30, "poll_latencies": [...],           # optional, system metrics
   "model": "...", "run": "..."}                     # optional, carried through

Any system can be scored, whatever produced the file. Benchmark items without a
prediction line are scored as fully silent unless ``skip_missing`` is set.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .data import iter_items, load_benchmark
from .mcq import load_mcq_scorers
from .metrics import MetricConfig, aggregate, score_item


@dataclass
class EvalConfig:
    predictions: str
    data: str = "data/interactionbench"
    mcq_key: str | None = None          # answer key: multiple-choice items scored by option
    items: str | None = None            # file of item_ids to score (frozen subsets)
    skip_missing: bool = False
    judge: str | None = None            # judge spec, see interactionbench.judges
    judge_cache: str | None = "results/judge_cache/judge_cache.jsonl"
    judge_prompt: str = "v2"
    judge_args: dict | None = None
    delta: float | None = None
    pre_tol: float | None = None
    gate_thresh: float | None = None
    use_gate: bool | None = None


def load_predictions(path: str | Path) -> dict[str, dict]:
    preds: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            preds[f"{p['video_id']}#{p.get('item_index', 0)}"] = p
    return preds


def evaluate(cfg: EvalConfig, judge=None) -> tuple[dict, list[dict]]:
    """Returns (summary, per-item records). Pass ``judge`` to reuse a built judge."""
    if judge is None and cfg.judge:
        from .judges import make_judge
        judge = make_judge(cfg.judge, cache_path=cfg.judge_cache,
                           prompt_version=cfg.judge_prompt, **(cfg.judge_args or {}))
        print(f"judge: {judge.name}", file=sys.stderr)

    mcq_scorers = load_mcq_scorers(cfg.mcq_key) if cfg.mcq_key else {}
    if mcq_scorers:
        print(f"multiple-choice scoring for {len(mcq_scorers)} items", file=sys.stderr)

    mc = MetricConfig()
    if cfg.delta is not None:
        mc.delta_s = cfg.delta
    if cfg.gate_thresh is not None:
        mc.gate_thresh = cfg.gate_thresh
    if cfg.use_gate is not None:
        mc.use_gate = cfg.use_gate
    if cfg.pre_tol is not None:
        mc.pre_tol_s = cfg.pre_tol

    items = {it.item_id: it for it in iter_items(load_benchmark(cfg.data))}
    if cfg.items:
        keep = {l.strip() for l in Path(cfg.items).read_text().splitlines() if l.strip()}
        items = {k: v for k, v in items.items() if k in keep}

    preds = load_predictions(cfg.predictions)
    unknown = sorted(set(preds) - set(items))
    if unknown:
        print(f"warning: {len(unknown)} prediction lines match no selected benchmark item "
              f"(e.g. {unknown[:3]})", file=sys.stderr)

    records, n_missing = [], 0
    for iid, item in items.items():
        p = preds.get(iid)
        if p is None:
            n_missing += 1
            if cfg.skip_missing:
                continue
            p = {"emissions": []}
        extra = {k: p[k] for k in ("n_polls", "poll_latencies") if k in p}
        rec = score_item(item, p.get("emissions") or [], mc,
                         judge=mcq_scorers.get(iid, judge), extra=extra or None)
        if iid in mcq_scorers:
            rec["mcq"] = True
        for k in ("model", "run"):
            if k in p:
                rec[k] = p[k]
        records.append(rec)

    if n_missing:
        how = "skipped" if cfg.skip_missing else "scored as silent"
        print(f"note: {n_missing}/{len(items)} items had no prediction line ({how})",
              file=sys.stderr)
    if judge is not None and hasattr(judge, "n_missing"):
        print(f"judge verdicts missing from cache: {judge.n_missing}/{judge.n_calls}",
              file=sys.stderr)

    agg = aggregate(records)
    agg["config"] = {
        "delta_s": mc.delta_s, "gate_thresh": mc.gate_thresh, "use_gate": mc.use_gate,
        "pre_tol_s": mc.pre_tol_s, "mcq_key": cfg.mcq_key, "items": cfg.items,
        "judge": judge.name if judge is not None else None,
        "judge_prompt": cfg.judge_prompt if judge is not None else None,
        "judge_cache_misses": (int(getattr(judge, "n_missing", 0) or 0)
                               if judge is not None else 0),
    }
    return agg, records


# ------------------------------------------------------------------ reporting

_COLS = [
    ("n_items", "n", 4), ("total_score", "total", 6),
    ("accuracy", "acc", 6), ("timing_accuracy", "TA", 6),
    ("silence_compliance", "SC", 6), ("timing_score", "T-hm", 6),
    ("delay_mean_s", "delay", 6), ("miss_rate", "miss", 5),
    ("v_premature", "vPre", 5), ("v_redundant", "vRed", 5),
    ("v_spurious", "vSpu", 5), ("segment_coverage", "segCov", 7),
    ("realtime_factor", "RTx*", 5), ("emissions_per_min", "em/m", 5),
]


def _fmt(v, w):
    if v is None:
        return " " * (w - 1) + "-"
    if isinstance(v, float):
        return f"{v:>{w}.2f}"
    return f"{v:>{w}}"


def print_table(title: str, groups: dict) -> None:
    print(f"\n### {title}")
    hdr = f"{'group':>22} | " + " ".join(f"{h:>{w}}" for _, h, w in _COLS)
    print(hdr)
    print("-" * len(hdr))
    for name, s in groups.items():
        row = " ".join(_fmt(s.get(k), w) for k, _, w in _COLS)
        print(f"{name:>22} | {row}")


def print_report(agg: dict, n_records: int, predictions: str) -> None:
    c = agg["config"]
    print("=" * 110)
    print(f"InteractionBench eval — {n_records} items | preds: {predictions} | "
          f"Delta={c['delta_s']}s pre_tol={c['pre_tol_s']}s "
          f"gate={'on@' + str(c['gate_thresh']) if c['use_gate'] else 'off'}")
    print("cols: total=item score (see docs/METRICS.md) | T-hm=hmean(TA,SC) | "
          "v*=violations per item | RTx*=wall-clock over budget, reference only")
    print("=" * 110)
    print_table("Overall", {"ALL": agg["overall"]})
    print_table("Core real-time tasks (TOA/PTR/CST/BRC/LCG) vs reactive QA",
                {"core": agg["by_core_realtime"].get("True", {}),
                 "reactive": agg["by_core_realtime"].get("False", {})})
    print_table("By capability group", agg["by_capability_group"])
    print_table("By time_type", agg["by_time_type"])
    print_table("By range_length", agg["by_range_length"])
    print_table("By metric family", agg["by_family"])


def write_outputs(out_dir: str | Path, agg: dict, records: list[dict]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
