"""Evaluation metrics for InteractionBench.

Two scoring axes:

  Accuracy          *what* was said. Discrete-answer tasks: choice/answer
                    accuracy (multiple choice where an answer key exists,
                    otherwise lexical or LLM-judge content scoring). Free-form streaming tasks
                    (LCG, BRC): temporal-VDCScore-style — the timeline is
                    segmented by the GT steps and the content the model
                    produced *within each segment* is scored against that
                    segment's reference. Coverage (did it speak when it
                    should) is NOT folded in; that lives in Decision Timing.

  Decision Timing   *when* it chose to speak, hardware-independent, wall-clock
                    excluded. Per GT event n a standard response time r*_n
                    (the earliest grounded-response moment = the annotated
                    answer time). Windows W_n = [r*_n, r*_{n+1}) partition the
                    stream; event n is matched by its FIRST in-window valid
                    response. Two sub-metrics on 0-100:

                    Timing Accuracy   = 100/N * sum_n (1 - d_n/Delta)_+ ,
                                        d_n = match(n) - r*_n, 0 if unmatched.
                                        Under-response is charged here (denom N).
                    Silence Compliance= 100 * (1 - |V|/M_max)_+ , V = premature
                                        (before r*_1) + redundant (in-window,
                                        not the match) + spurious (silent item
                                        or gate-failed trigger). Over-response
                                        is charged here.
                    timing_score      = harmonic mean of the two (both axes
                                        must be good; chatter and muteness
                                        both collapse it).

  Latency           wall-clock generation cost (poll latency, realtime factor)
                    is reported for REFERENCE ONLY and never enters any score.

Headline ``total_score`` (0-100) has a two-level structure —
top level Accuracy x Decision Timing averaged arithmetically, TA and SC
combined by HARMONIC mean inside the timing axis (both have trivial
degenerate maximisers — talk-always for TA, stay-mute for SC — so they must
not compensate each other; content vs timing may):

  A / B / C-counting   mean(Accuracy, hmean(Timing Accuracy, Silence Compliance))
  LCG / BRC            Accuracy alone (per-event timing is not a
                       scoring dimension for free-form streams); Accuracy is
                       averaged over ALL segments with uncovered segments
                       scoring 0, so going quiet cannot inflate it
  negatives            Silence Compliance alone

Anti-gaming rule: on positive items SC is DEFINED ONLY IF the model responded
at least once. A fully silent stream has no discipline to measure — its SC is
None, excluded from aggregates, and the item total collapses to the mean of
Accuracy and TA (both 0). Without this rule "never speak" would bank a free
100 on one of three axes. Content-as-precondition for PTR still lives in the
matching gate (a right-time wrong-content fire is a spurious violation).

All times are stream seconds (annotations are second-anchored); with a known
fps this is a linear rescale of the spec's frame indices — pass Delta in the
same unit you anchor emissions with.

A model run supplies, per item, time-ordered *emissions*:
``{"t": <stream seconds>, "content": <str>, "latency_s": <optional>}``.
"""

from __future__ import annotations

import re
import statistics as st
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from .data import BenchItem

# ---------------------------------------------------------------- config

REALTIME_BUDGET_S = 0.2  # reference-only real-time budget for wall-clock cost

# merged reporting groups: IVQA absorbs CIR
# (present-frame recognition + cross-frame causal reasoning), CST absorbs BRC
# (state tracking incl. belief revision)
CAP_GROUPS = {"CIR": "IVQA+CIR", "IVQA": "IVQA+CIR",
              "BRC": "CST+BRC", "CST": "CST+BRC"}
# the bolded "real real-time interaction" tasks: standing/streaming obligations
CORE_REALTIME = {"TOA", "PTR", "CST", "BRC", "LCG"}

# capabilities whose accuracy is scored VDCScore-style over temporal segments
SEGMENT_ACCURACY_CAPS = {"LCG", "BRC"}
# capabilities where content is only a gate on the trigger, not an accuracy axis
GATED_TIMING_CAPS = {"PTR"}


@dataclass
class MetricConfig:
    # Delta: acceptable-delay bound of the linear decay, stream seconds.
    delta_s: float = 5.0
    # content-gate threshold: an in-window response whose content scores below
    # this against the event's reference is a hallucinated trigger (violation)
    gate_thresh: float = 0.3
    # The content gate is disabled in the paper protocol; the code path is kept for ablations.
    use_gate: bool = False
    # Silence Compliance normalizer M_max = max(n_events, sc_mmax_min);
    # silent items use sc_mmax_min (one should-be-silent interval)
    sc_mmax_min: int = 1
    realtime_budget_s: float = REALTIME_BUDGET_S
    # Pre-anchor tolerance (stream seconds). An emission at
    # r*_n - pre_tol_s <= t < r*_n is the response to event n with delay 0
    # (humans click ~1 s before the annotated determinability moment in 27% of
    # cases). For attribution every window shifts forward by pre_tol_s:
    # W_n = [r*_n - pre_tol_s, r*_{n+1} - pre_tol_s), so an emission just before
    # r*_{n+1} goes to n+1 instead of being redundant for n; emissions before
    # r*_1 - pre_tol_s stay premature. 0.0 reproduces the original protocol.
    # 1.0 s is the paper protocol and the default; pass MetricConfig(pre_tol_s=0.0)
    # or --pre-tol 0.0 for a strict protocol without tolerance.
    # float('inf') (human reference only): unbounded tolerance with greedy attribution,
    # see decision_timing -- no early emission is ever a hallucination on a
    # positive item; negative items are unaffected.
    pre_tol_s: float = 1.0


# ---------------------------------------------------------------- content scoring

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
    return float(m[-1]) if m else None  # last number: "count is now 7" -> 7


def content_score(gt: str | None, pred: str | None,
                  question: str = "",
                  judge: Callable[[str, str, str], float] | None = None) -> float:
    """Similarity in [0,1] between a GT answer and a model emission."""
    if judge is not None:
        return float(judge(question, gt or "", pred or ""))
    if gt is None:
        return 1.0 if not pred else 0.0
    exact = 1.0 if normalize_text(gt) == normalize_text(pred) and gt else 0.0
    f1 = token_f1(gt, pred or "")
    # numeric credit only for short, number-centric GTs ("7", "40°C", "9:42"):
    # a long sentence that merely contains a number must match on words instead
    gn, pn = extract_number(gt), extract_number(pred)
    # numeric credit additionally requires the GT's non-numeric words (if any)
    # to appear in the prediction: GT "JUMP 4" must not be satisfied by
    # "DUCK 4" — when the GT names an action/object, the count alone is not
    # the answer. Pure-numeric GTs ("7", "9:42") are unaffected.
    gt_toks = normalize_text(gt).split()
    alpha = [t for t in gt_toks if not _NUM_RE.fullmatch(t)]
    pred_toks = set(normalize_text(pred).split())
    num = 1.0 if (gn is not None and pn is not None and abs(gn - pn) < 1e-9
                  and len(gt_toks) <= 3
                  and all(t in pred_toks for t in alpha)) else 0.0
    return max(exact, f1, num)


# ---------------------------------------------------------------- decision timing

def merge_coincident(gts) -> list[tuple[float, str]]:
    """Merge GT answers sharing one timestamp into a single event (a zero-width
    window can never be matched). Returns [(time_s, joined_content)] sorted.
    Coincident timestamps are an annotation artifact; merging keeps such items
    scoreable without special-casing."""
    by_t: dict[float, list[str]] = {}
    for a in gts:
        by_t.setdefault(a.time_s, []).append(a.content or "")
    return [(t, "; ".join(c for c in cs if c)) for t, cs in sorted(by_t.items())]

def decision_timing(gt_times: list[float], preds: list[dict],
                    cfg: MetricConfig,
                    valid: Callable[[int, dict], bool] | None = None) -> dict:
    """Decision Timing over one item.

    gt_times: sorted standard response times r*_1..r*_N (N >= 1).
    preds: time-ordered emissions. ``valid(event_idx, emission)`` is the
    content gate; an in-window emission failing it is a spurious violation.

    Returns TA / SC (0-100), per-event delays, and the violation breakdown.
    """
    N = len(gt_times)
    assert N >= 1
    matched: dict[int, dict] = {}
    delays: list[float | None] = [None] * N
    premature = redundant = spurious = 0

    tol = float(getattr(cfg, "pre_tol_s", 0.0) or 0.0)
    if tol == float("inf"):
        # Unbounded tolerance, greedy attribution (human reference protocol):
        # walking emissions in time order, an emission at t in [r*_n, r*_{n+1})
        # is the response to event n (delay t - r*_n) if n is still unclaimed,
        # else the response to event n+1 with delay 0 if that exists and is
        # unclaimed, else redundant. Emissions before r*_1 answer event 1 with
        # delay 0 (the first one; later ones are redundant). Nothing is ever
        # premature or spurious (the content gate is not applied here).
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
        ta = t + tol  # attribution time: windows shift forward by pre_tol_s
        if ta < gt_times[0]:
            premature += 1
            continue
        # window index: last n with r*_n - tol <= t  (windows partition [r*_1 - tol, inf))
        n = max(i for i in range(N) if gt_times[i] <= ta)
        if n in matched:
            redundant += 1
        elif valid is not None and not valid(n, e):
            spurious += 1          # hallucinated trigger: right time, wrong content
        else:
            matched[n] = e
            # inside the tolerance band (t < r*_n) the response earns full credit
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
        # precision-form diagnostic (self-normalizing alternative to M_max)
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
    """Mean of the defined axes (item headline score)."""
    xs = [x for x in axes if x is not None]
    return round(st.mean(xs), 1) if xs else 0.0


def _engaged_sc(dt: dict) -> float | None:
    """SC under the anti-gaming rule: undefined when nothing was emitted."""
    return dt["silence_compliance"] if dt["n_resp"] > 0 else None


# ---------------------------------------------------------------- per-family scoring

def _latency_reference(emissions: list[dict], item: BenchItem, extra: dict | None) -> dict:
    """Wall-clock cost — reference only, never enters scores."""
    lats = [e["latency_s"] for e in emissions if e.get("latency_s") is not None]
    if extra and extra.get("poll_latencies"):
        lats = list(extra["poll_latencies"])
    out: dict = {}
    if lats:
        out["poll_latency_mean"] = round(st.mean(lats), 3)
        out["poll_latency_max"] = round(max(lats), 3)
        out["realtime_factor"] = round(st.mean(lats) / REALTIME_BUDGET_S, 2)
    dur = item.duration_s or 0.0
    # None (skipped by summarize) when the annotation lacks a duration.
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
    """A-type (LVM / IVQA / CIR-A): r* = question time; answer when queried."""
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
    """B-type and C-counting: per-event decision timing with a content gate."""
    events = merge_coincident(item.timed_answers)
    gt_times = [t for t, _ in events]
    gt_contents = [c for _, c in events]
    counting = item.is_counting

    def gate(n: int, e: dict) -> bool:
        # Paper protocol: no content gate. The first in-window emission is the
        # response to event n whatever it says; content enters Accuracy only.
        if not cfg.use_gate:
            return True
        if counting:  # the running count must be right to count as the match
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
            # exact match (100/0); the absolute error is kept only as a diagnostic
            r["accuracy"] = 100.0 if (pred_final is not None and err < 1e-9) else 0.0
        else:
            r["accuracy"] = None
    else:
        # matched-fire content quality; no fires -> 0 (failed the axis, not N/A)
        r["accuracy"] = r["content_score"] if cscores else 0.0
    r["total_score"] = _total(r["accuracy"], timing)
    return r


def _score_segments(item: BenchItem, preds: list[dict], cfg: MetricConfig,
                    judge) -> dict:
    """LCG / BRC: temporal-VDCScore-style accuracy over GT segments.

    Segment i = [t_i, t_{i+1}) (last: to video end). The model's content inside
    a segment is concatenated and scored against that segment's reference.
    Accuracy averages over *covered* segments only; coverage itself is
    reported under the timing axis (not folded into Accuracy).
    """
    events = merge_coincident(item.timed_answers)
    bounds = [t for t, _ in events] + [max(item.duration_s, events[-1][0]) + 1e-6]
    seg_scores = []   # one entry per segment; uncovered segments score 0
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
    # Accuracy averages ALL segments (silence on a segment = 0), so going quiet
    # cannot inflate it; covered_accuracy shows content quality where it spoke
    acc = round(100.0 * st.mean(seg_scores), 1)
    coverage = round(len(covered_scores) / len(events), 3)

    dt = decision_timing([t for t, _ in events], preds, cfg)  # diagnostic only
    dt.pop("matched_events")
    dt["silence_compliance"] = _engaged_sc(dt)
    r = {"family": "C_segments", **dt,
         "accuracy": acc,
         "covered_accuracy": (round(100.0 * st.mean(covered_scores), 1)
                              if covered_scores else None),
         "segment_coverage": coverage,
         "timing_score": _hmean(dt["timing_accuracy"], dt["silence_compliance"]),
         # timing is not a scoring dimension for free-form streams
         "total_score": acc}
    return r


# ---------------------------------------------------------------- entry points

def score_item(item: BenchItem, emissions: list[dict],
               cfg: MetricConfig | None = None,
               judge: Callable[[str, str, str], float] | None = None,
               extra: dict | None = None) -> dict:
    """Score one item. ``emissions``: time-ordered [{"t", "content", "latency_s"?}]."""
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
    else:  # B triggers + C counting / repeated triggers
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
    """Summaries overall and per grouping key."""
    keys = keys or ["capability", "capability_group", "time_type",
                    "interaction_type", "family", "range_length", "core_realtime"]
    agg = {"overall": summarize(records)}
    for key in keys:
        groups: dict = defaultdict(list)
        for r in records:
            groups[str(r.get(key))].append(r)
        agg[f"by_{key}"] = {k: summarize(v) for k, v in sorted(groups.items())}
    return agg
