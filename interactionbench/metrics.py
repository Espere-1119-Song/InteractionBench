"""Evaluation metrics for InteractionBench."""

from __future__ import annotations

import re
import statistics as st
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from .data import BenchItem


REALTIME_BUDGET_S = 0.2

CAP_GROUPS = {"CIR": "IVQA+CIR", "IVQA": "IVQA+CIR",
              "BRC": "CST+BRC", "CST": "CST+BRC"}
CORE_REALTIME = {"TOA", "PTR", "CST", "BRC", "LCG"}

SEGMENT_ACCURACY_CAPS = {"LCG", "BRC"}
GATED_TIMING_CAPS = {"PTR"}


@dataclass
class MetricConfig:
    delta_s: float = 5.0
    gate_thresh: float = 0.3
    use_gate: bool = False
    sc_mmax_min: int = 1
    realtime_budget_s: float = REALTIME_BUDGET_S
    pre_tol_s: float = 1.0


_WORD_RE = re.compile(r"[a-z0-9]+")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_ARTICLES = {"a", "an", "the"}


def normalize_text(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    toks = [t for t in _WORD_RE.findall(s) if t not in _ARTICLES]
    return " ".join(toks)


def token_f1(gt: str, pred: str) -> float:
    g, p = normalize_text(gt).split(), normalize_text(pred).split()
    if not g or not p:
        return 0.0
    common: dict[str, int] = {}
    for t in g:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in p:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def extract_number(s: str | None) -> float | None:
    if not s:
        return None
    m = _NUM_RE.findall(s.replace(",", ""))
    return float(m[-1]) if m else None


def content_score(gt: str | None, pred: str | None,
                  question: str = "",
                  judge: Callable[[str, str, str], float] | None = None) -> float:
    if judge is not None:
        return float(judge(question, gt or "", pred or ""))
    if gt is None:
        return 1.0 if not pred else 0.0
    exact = 1.0 if normalize_text(gt) == normalize_text(pred) and gt else 0.0
    f1 = token_f1(gt, pred or "")
    gn, pn = extract_number(gt), extract_number(pred)
    gt_toks = normalize_text(gt).split()
    alpha = [t for t in gt_toks if not _NUM_RE.fullmatch(t)]
    pred_toks = set(normalize_text(pred).split())
    num = 1.0 if (gn is not None and pn is not None and abs(gn - pn) < 1e-9
                  and len(gt_toks) <= 3
                  and all(t in pred_toks for t in alpha)) else 0.0
    return max(exact, f1, num)


def merge_coincident(gts) -> list[tuple[float, str]]:
    by_t: dict[float, list[str]] = {}
    for a in gts:
        by_t.setdefault(a.time_s, []).append(a.content or "")
    return [(t, "; ".join(c for c in cs if c)) for t, cs in sorted(by_t.items())]

def decision_timing(gt_times: list[float], preds: list[dict],
                    cfg: MetricConfig,
                    valid: Callable[[int, dict], bool] | None = None) -> dict:
    N = len(gt_times)
    assert N >= 1
    matched: dict[int, dict] = {}
    delays: list[float | None] = [None] * N
    premature = redundant = spurious = 0

    tol = float(getattr(cfg, "pre_tol_s", 0.0) or 0.0)
    if tol == float("inf"):
        for e in preds:
            t = e["t"]
            if t < gt_times[0]:
                n, d = 0, 0.0
            else:
                n = max(i for i in range(N) if gt_times[i] <= t)
                d = t - gt_times[n]
                if n in matched and n + 1 < N and (n + 1) not in matched:
                    n, d = n + 1, 0.0
            if n in matched:
                redundant += 1
                continue
            matched[n] = e
            delays[n] = d
        preds_iter = []
    else:
        preds_iter = preds
    for e in preds_iter:
        t = e["t"]
        ta = t + tol
        if ta < gt_times[0]:
            premature += 1
            continue
        n = max(i for i in range(N) if gt_times[i] <= ta)
        if n in matched:
            redundant += 1
        elif valid is not None and not valid(n, e):
            spurious += 1
        else:
            matched[n] = e
            delays[n] = max(0.0, t - gt_times[n])

    scores = [max(0.0, 1.0 - d / cfg.delta_s) if d is not None else 0.0
              for d in delays]
    n_viol = premature + redundant + spurious
    m_max = max(N, cfg.sc_mmax_min)
    valid_resp = len(matched)
    n_resp = len(preds)
    return {
        "timing_accuracy": round(100.0 * st.mean(scores), 1),
        "silence_compliance": round(100.0 * max(0.0, 1.0 - n_viol / m_max), 1),
        "n_gt": N, "n_resp": n_resp, "n_matched": valid_resp,
        "n_violations": n_viol,
        "v_premature": premature, "v_redundant": redundant, "v_spurious": spurious,
        "response_precision": round(valid_resp / n_resp, 3) if n_resp else None,
        "delay_mean_s": round(st.mean([d for d in delays if d is not None]), 2)
                        if any(d is not None for d in delays) else None,
        "miss_rate": round(sum(d is None for d in delays) / N, 3),
        "matched_events": matched,
    }


def _hmean(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return a if b is None else b
    if a + b == 0:
        return 0.0
    return round(2 * a * b / (a + b), 1)


def _total(*axes: float | None) -> float:
    xs = [x for x in axes if x is not None]
    return round(st.mean(xs), 1) if xs else 0.0


def _engaged_sc(dt: dict) -> float | None:
    return dt["silence_compliance"] if dt["n_resp"] > 0 else None


def _latency_reference(emissions: list[dict], item: BenchItem, extra: dict | None) -> dict:
    lats = [e["latency_s"] for e in emissions if e.get("latency_s") is not None]
    if extra and extra.get("poll_latencies"):
        lats = list(extra["poll_latencies"])
    out: dict = {}
    if lats:
        out["poll_latency_mean"] = round(st.mean(lats), 3)
        out["poll_latency_max"] = round(max(lats), 3)
        out["realtime_factor"] = round(st.mean(lats) / REALTIME_BUDGET_S, 2)
    dur = item.duration_s or 0.0
    out["emissions_per_min"] = round(60.0 * len(emissions) / dur, 2) if dur > 0 else None
    if extra and extra.get("n_polls"):
        out["chatter_rate"] = round(len(emissions) / extra["n_polls"], 3)
    return out


def _score_silent_item(item: BenchItem, preds: list[dict], cfg: MetricConfig) -> dict:
    n = len(preds)
    sc = 100.0 * max(0.0, 1.0 - n / cfg.sc_mmax_min)
    return {
        "family": "negative",
        "false_alarm": n > 0,
        "n_violations": n, "v_spurious": n, "v_premature": 0, "v_redundant": 0,
        "silence_compliance": round(sc, 1),
        "timing_score": round(sc, 1),
        "total_score": round(sc, 1),
    }


def _score_retro_qa(item: BenchItem, preds: list[dict], cfg: MetricConfig,
                    judge) -> dict:
    gt = item.timed_answers[0]
    r_star = item.question_time_s
    dt = decision_timing([r_star], preds, cfg)
    match = dt.pop("matched_events")
    dt["silence_compliance"] = _engaged_sc(dt)
    acc = 0.0
    if 0 in match:
        _vdc = getattr(judge, "vdc", None)
        acc = round(100.0 * (
            _vdc(item.question, gt.content, match[0].get("content") or "")
            if _vdc else content_score(gt.content, match[0].get("content"),
                                       item.question, judge)), 1)
    r = {"family": "A_retro_qa", **dt,
         "answered": 0 in match,
         "accuracy": acc,
         "timing_score": _hmean(dt["timing_accuracy"], dt["silence_compliance"])}
    if gt.evidence_time_s is not None:
        r["memory_span_s"] = round(r_star - gt.evidence_time_s, 2)
    r["total_score"] = _total(acc, r["timing_score"])
    return r


def _score_trigger(item: BenchItem, preds: list[dict], cfg: MetricConfig,
                   judge) -> dict:
    events = merge_coincident(item.timed_answers)
    gt_times = [t for t, _ in events]
    gt_contents = [c for _, c in events]
    counting = item.is_counting

    def gate(n: int, e: dict) -> bool:
        if not cfg.use_gate:
            return True
        if counting:
            gn, pn = extract_number(gt_contents[n]), extract_number(e.get("content"))
            return gn is None or (pn is not None and abs(pn - gn) < 1e-9)
        s = content_score(gt_contents[n], e.get("content"), item.question, judge)
        return s >= cfg.gate_thresh

    dt = decision_timing(gt_times, preds, cfg, valid=gate)
    match = dt.pop("matched_events")
    dt["silence_compliance"] = _engaged_sc(dt)
    cscores = [content_score(gt_contents[n], e.get("content"), item.question, judge)
               for n, e in match.items()]
    timing = _hmean(dt["timing_accuracy"], dt["silence_compliance"])
    r = {"family": "C_counting" if counting else "B_trigger", **dt,
         "content_score": round(100.0 * st.mean(cscores), 1) if cscores else None,
         "timing_score": timing}

    if counting:
        gt_final = next((extract_number(a.content) for a in item.timed_answers[::-1]
                         if extract_number(a.content) is not None), None)
        nums = [extract_number(e.get("content")) for e in preds]
        nums = [x for x in nums if x is not None]
        pred_final = nums[-1] if nums else None
        r["gt_final_count"] = gt_final
        r["pred_final_count"] = pred_final
        if gt_final is not None:
            err = abs((pred_final or 0.0) - gt_final)
            r["final_count_abs_err"] = round(err, 2)
            r["accuracy"] = 100.0 if (pred_final is not None and err < 1e-9) else 0.0
        else:
            r["accuracy"] = None
    else:
        r["accuracy"] = r["content_score"] if cscores else 0.0
    r["total_score"] = _total(r["accuracy"], timing)
    return r


def _score_segments(item: BenchItem, preds: list[dict], cfg: MetricConfig,
                    judge) -> dict:
    events = merge_coincident(item.timed_answers)
    bounds = [t for t, _ in events] + [max(item.duration_s, events[-1][0]) + 1e-6]
    seg_scores = []
    covered_scores = []
    for i, (_, ref) in enumerate(events):
        inside = [e.get("content") or "" for e in preds
                  if bounds[i] <= e["t"] < bounds[i + 1]]
        text = " ".join(s for s in inside if s.strip())
        if not text:
            seg_scores.append(0.0)
            continue
        _vdc = getattr(judge, "vdc", None)
        s = _vdc(item.question, ref, text) if _vdc else \
            content_score(ref, text, item.question, judge)
        seg_scores.append(s)
        covered_scores.append(s)
    acc = round(100.0 * st.mean(seg_scores), 1)
    coverage = round(len(covered_scores) / len(events), 3)

    dt = decision_timing([t for t, _ in events], preds, cfg)
    dt.pop("matched_events")
    dt["silence_compliance"] = _engaged_sc(dt)
    r = {"family": "C_segments", **dt,
         "accuracy": acc,
         "covered_accuracy": (round(100.0 * st.mean(covered_scores), 1)
                              if covered_scores else None),
         "segment_coverage": coverage,
         "timing_score": _hmean(dt["timing_accuracy"], dt["silence_compliance"]),
         "total_score": acc}
    return r


def score_item(item: BenchItem, emissions: list[dict],
               cfg: MetricConfig | None = None,
               judge: Callable[[str, str, str], float] | None = None,
               extra: dict | None = None) -> dict:
    cfg = cfg or MetricConfig()
    if not item.valid:
        raise ValueError(f"item {item.item_id} is not scoreable")
    preds = sorted((e for e in emissions if (e.get("content") or "").strip()),
                   key=lambda e: e["t"])

    if item.should_remain_silent:
        r = _score_silent_item(item, preds, cfg)
    elif item.time_type == "A":
        r = _score_retro_qa(item, preds, cfg, judge)
    elif item.capability in SEGMENT_ACCURACY_CAPS:
        r = _score_segments(item, preds, cfg, judge)
    else:
        r = _score_trigger(item, preds, cfg, judge)

    r.update(_latency_reference(emissions, item, extra))
    r.update({
        "item_id": item.item_id,
        "video_id": item.video_id,
        "capability": item.capability,
        "capability_group": CAP_GROUPS.get(item.capability, item.capability),
        "core_realtime": item.capability in CORE_REALTIME,
        "time_type": item.time_type,
        "interaction_type": item.interaction_type,
        "sub_tag": item.sub_tag,
        "range_length": item.range_length,
        "domain": item.domain,
        "duration_s": item.duration_s,
    })
    return r


_MEANABLE = [
    "total_score", "accuracy", "timing_accuracy", "silence_compliance",
    "covered_accuracy",
    "timing_score", "content_score", "response_precision",
    "delay_mean_s", "miss_rate", "memory_span_s",
    "v_premature", "v_redundant", "v_spurious",
    "final_count_abs_err", "segment_coverage",
    "poll_latency_mean", "realtime_factor", "emissions_per_min", "chatter_rate",
]
_RATE = ["answered", "false_alarm"]


def _mean(xs: Iterable) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 3) if xs else None


def summarize(records: list[dict]) -> dict:
    out: dict = {"n_items": len(records)}
    for k in _MEANABLE:
        v = _mean(r.get(k) for r in records)
        if v is not None:
            out[k] = v
    for k in _RATE:
        vals = [r[k] for r in records if k in r]
        if vals:
            out[f"{k}_rate"] = round(sum(bool(v) for v in vals) / len(vals), 3)
    return out


def aggregate(records: list[dict], keys: list[str] | None = None) -> dict:
    keys = keys or ["capability", "capability_group", "time_type",
                    "interaction_type", "family", "range_length", "core_realtime"]
    agg = {"overall": summarize(records)}
    for key in keys:
        groups: dict = defaultdict(list)
        for r in records:
            groups[str(r.get(key))].append(r)
        agg[f"by_{key}"] = {k: summarize(v) for k, v in sorted(groups.items())}
    return agg
