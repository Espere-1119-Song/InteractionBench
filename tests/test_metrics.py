"""Unit tests for interactionbench.metrics.

Run:  python tests/test_metrics.py   (or pytest tests/)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interactionbench.metrics import (MetricConfig, content_score, decision_timing,
                           extract_number, score_item, summarize, token_f1)
from interactionbench.data import BenchItem, GTAnswer

CFG = MetricConfig()                  # paper protocol: Delta 5 s, pre-anchor tolerance 1 s, no gate
GATE = MetricConfig(use_gate=True)    # ablation: content gate on trigger matching


def _item(time_type="B", capability="PTR", answers=None, q_time=0.0,
          is_negative=False, auto_number=False, sub_tag=None, dur=60.0):
    return BenchItem(
        video_id="vid", item_index=0, capability=capability,
        time_type=time_type, interaction_type="QA", sub_tag=sub_tag,
        is_negative=is_negative, auto_number=auto_number,
        question="q?", question_time_s=q_time,
        answers=answers or [], duration_s=dur)


def _em(*pairs):
    return [{"t": t, "content": c} for t, c in pairs]


def approx(a, b, tol=1e-6):
    assert a is not None and abs(a - b) <= tol, f"{a} != {b}"


# ---------------------------------------------------------------- content scoring

def test_content_scoring():
    approx(token_f1("a dark blue cup", "the dark blue cup"), 1.0)  # articles dropped
    assert token_f1("open the door", "close window") == 0.0
    approx(content_score("40", "the high is 40 degrees"), 1.0)     # numeric, short GT
    assert content_score("JUMP 4", "DUCK 4") < 1.0                 # the named action must match too
    assert content_score("press the brake with your right foot then start", "40") < 0.2
    approx(content_score("Task Manager", "task  manager"), 1.0)    # normalisation
    assert extract_number("count is now 7") == 7.0
    assert content_score(None, "") == 1.0 and content_score(None, "hi") == 0.0


# ---------------------------------------------------------------- decision timing core

def test_decision_timing_oracle():
    dt = decision_timing([10.0, 40.0], _em((10.0, "a"), (40.0, "b")), CFG)
    approx(dt["timing_accuracy"], 100.0)
    approx(dt["silence_compliance"], 100.0)
    assert dt["n_violations"] == 0 and dt["miss_rate"] == 0.0


def test_decision_timing_linear_decay():
    # delay 2.5s with Delta=5 -> s = 0.5
    dt = decision_timing([10.0], _em((12.5, "a")), CFG)
    approx(dt["timing_accuracy"], 50.0)
    # delay >= Delta -> matched (window runs to infinity) but scores 0
    dt = decision_timing([10.0], _em((15.0, "a")), CFG)
    approx(dt["timing_accuracy"], 0.0)
    approx(dt["silence_compliance"], 100.0)  # still not a violation
    assert dt["miss_rate"] == 0.0


def test_decision_timing_violations():
    # premature: before r*_1 -> charged to SC, never an early hit
    dt = decision_timing([10.0], _em((8.0, "a")), CFG)
    approx(dt["timing_accuracy"], 0.0)
    assert dt["v_premature"] == 1
    approx(dt["silence_compliance"], 0.0)   # M_max = max(N,1) = 1
    # redundant: second response in an already-matched window
    dt = decision_timing([10.0, 40.0], _em((10.0, "a"), (12.0, "again"), (40.0, "b")), CFG)
    assert dt["v_redundant"] == 1
    approx(dt["silence_compliance"], 50.0)  # 1 violation / M_max=2
    approx(dt["timing_accuracy"], 100.0)
    # first-in-window matching: window of event 1 is [10, 40)
    dt = decision_timing([10.0, 40.0], _em((30.0, "late-but-window1"),), CFG)
    assert dt["n_matched"] == 1 and dt["miss_rate"] == 0.5
    approx(dt["timing_accuracy"], 0.0)      # delay 20 >= Delta -> 0, event2 missed


def test_decision_timing_gate():
    # gate failure = hallucinated trigger (spurious); a later correct fire matches
    gate = lambda n, e: e["content"] == "right"
    dt = decision_timing([10.0], _em((10.0, "wrong"), (11.0, "right")), CFG, valid=gate)
    assert dt["v_spurious"] == 1 and dt["n_matched"] == 1
    approx(dt["timing_accuracy"], 80.0)     # delay 1.0 / 5.0
    approx(dt["silence_compliance"], 0.0)


# ---------------------------------------------------------------- A retrospective QA

def test_A_retro_qa():
    it = _item("A", "LVM", q_time=70.0,
               answers=[GTAnswer(70.0, "40°C", evidence_time_s=22.3)])
    r = score_item(it, _em((70.0, "It was 40°C")), CFG)
    assert r["family"] == "A_retro_qa" and r["answered"]
    approx(r["accuracy"], 100.0)
    approx(r["timing_accuracy"], 100.0)
    approx(r["total_score"], 100.0)         # mean(acc, hmean(TA,SC)) all perfect
    approx(r["memory_span_s"], 47.7, 0.01)
    # slow answer: accuracy holds, timing decays -> two axes are orthogonal
    r = score_item(it, _em((72.5, "40°C")), CFG)
    approx(r["accuracy"], 100.0)
    approx(r["timing_accuracy"], 50.0)
    # pre-question emission = premature violation, not an answer
    r = score_item(it, _em((30.0, "40°C")), CFG)
    assert not r["answered"] and r["v_premature"] == 1 and r["total_score"] == 0.0
    # silence
    r = score_item(it, [], CFG)
    assert not r["answered"] and r["total_score"] == 0.0 and r["timing_accuracy"] == 0.0
    assert r["silence_compliance"] is None   # anti-gaming: silence has no SC


# ---------------------------------------------------------------- B triggers

def test_B_PTR_timing_primary_with_gate():
    it = _item("B", "PTR", answers=[GTAnswer(79.0, "He starts putting up the walls")])
    r = score_item(it, _em((79.5, "he starts putting up the walls now")), CFG)
    assert r["family"] == "B_trigger"
    approx(r["timing_accuracy"], 90.0)      # 0.5s / 5s
    approx(r["silence_compliance"], 100.0)
    assert r["total_score"] > 90.0          # mean(acc, hmean(90,100))
    # paper protocol: the first in-window response is the match whatever it says;
    # wrong content costs Accuracy only
    r = score_item(it, _em((79.5, "the weather is nice")), CFG)
    assert r["v_spurious"] == 0
    approx(r["timing_accuracy"], 90.0)
    approx(r["accuracy"], 0.0)
    # gate ablation: right time, wrong content -> spurious, event unmatched
    r = score_item(it, _em((79.5, "the weather is nice")), GATE)
    assert r["v_spurious"] == 1 and r["timing_accuracy"] == 0.0
    approx(r["silence_compliance"], 0.0)
    approx(r["total_score"], 0.0)


def test_B_TOA_both_axes():
    it = _item("B", "TOA", answers=[GTAnswer(20.0, "task manager")])
    r = score_item(it, _em((20.0, "task manager")), CFG)
    approx(r["total_score"], 100.0)         # mean(acc, hmean(TA,SC))
    r = score_item(it, _em((22.5, "task manager")), CFG)
    approx(r["accuracy"], 100.0)
    approx(r["timing_score"], 2 * 50 * 100 / 150, 0.1)  # hmean(TA=50, SC=100)
    approx(r["total_score"], (100 + 2*50*100/150) / 2, 0.1)  # mean(acc, hmean)


def test_B_negative():
    it = _item("B", "PTR", is_negative=True,
               answers=[GTAnswer(None, "SHOULD_REMAIN_SILENT")])
    r = score_item(it, [], CFG)
    assert r["family"] == "negative" and not r["false_alarm"]
    approx(r["total_score"], 100.0)
    r = score_item(it, _em((12.0, "it's 12:00")), CFG)
    assert r["false_alarm"] and r["total_score"] == 0.0 and r["v_spurious"] == 1


# ---------------------------------------------------------------- C counting

def test_C_counting():
    answers = [GTAnswer(t, str(i + 1)) for i, t in enumerate([5.0, 15.0, 25.0])]
    it = _item("C", "CST", answers=answers, auto_number=True, sub_tag="计数型")
    r = score_item(it, _em((5.2, "1"), (15.1, "2"), (25.3, "3")), CFG)
    assert r["family"] == "C_counting"
    approx(r["accuracy"], 100.0)            # final count exact
    assert r["timing_accuracy"] > 90.0
    assert r["total_score"] > 95.0          # mean(acc, hmean(TA,SC))
    # gate ablation: a wrong running count at the right time is a hallucinated trigger
    r = score_item(it, _em((5.2, "2"),), GATE)
    assert r["v_spurious"] == 1 and r["timing_accuracy"] == 0.0
    # right final count, no timing: acc 100, TA 0 for first two events
    r = score_item(it, _em((59.0, "3"),), CFG)
    approx(r["accuracy"], 100.0)
    assert r["timing_accuracy"] == 0.0      # only event3 matched, delay 34s -> 0
    # silence: accuracy 0 (|0-3|/3), TA 0
    r = score_item(it, [], CFG)
    approx(r["accuracy"], 0.0)
    approx(r["total_score"], 0.0)


# ---------------------------------------------------------------- C segments (LCG/BRC)

def test_C_segments():
    answers = [GTAnswer(2.0, "separate the egg yolks"),
               GTAnswer(10.0, "whip the egg whites"),
               GTAnswer(26.0, "pour into a container and bake")]
    it = _item("C", "LCG", answers=answers, sub_tag="叙述型", dur=30.0)
    r = score_item(it, _em((2.5, "separate the egg yolks"),
                           (10.5, "whip the egg whites"),
                           (26.5, "pour into a container and bake")), CFG)
    assert r["family"] == "C_segments"
    approx(r["accuracy"], 100.0)
    approx(r["segment_coverage"], 1.0)
    approx(r["total_score"], 100.0)
    # good content in 1 of 3 segments: accuracy averages covered segments only,
    # but primary is coverage-weighted so silence can't dodge the axis
    r = score_item(it, _em((2.5, "separate the egg yolks"),), CFG)
    approx(r["accuracy"], 33.3, 0.1)        # uncovered segments count as 0
    approx(r["covered_accuracy"], 100.0)
    approx(r["segment_coverage"], 1 / 3, 0.01)
    approx(r["total_score"], 33.3, 0.1)
    # wrong content everywhere -> covered but accuracy ~0
    r = score_item(it, _em((2.5, "blah"), (10.5, "blah"), (26.5, "blah")), CFG)
    approx(r["segment_coverage"], 1.0)
    assert r["accuracy"] < 10.0


# ---------------------------------------------------------------- judge + system

def test_judge_hook_and_latency_reference():
    from interactionbench.judges import parse_score
    approx(parse_score("SCORE: 1"), 1.0)      # verdicts are binary on any scale
    approx(parse_score("score: 7"), 1.0)
    approx(parse_score("30"), 0.0)
    approx(parse_score("garbage"), 0.0)
    it = _item("A", "LVM", q_time=10.0, answers=[GTAnswer(10.0, "40°C")])
    r = score_item(it, _em((10.0, "forty degrees Celsius")), CFG,
                   judge=lambda q, gt, pred: 1.0)
    approx(r["accuracy"], 100.0)
    # latency is reference-only: never changes scores
    ems = [{"t": 10.0, "content": "40°C", "latency_s": 99.0}]
    r2 = score_item(it, ems, CFG)
    approx(r2["accuracy"], 100.0)
    approx(r2["total_score"], 100.0)
    approx(r2["poll_latency_mean"], 99.0)
    approx(r2["realtime_factor"], 495.0)

    s = summarize([r, r2])
    assert s["n_items"] == 2 and s["total_score"] == 100.0


# ---------------------------------------------------------------- pre-anchor tolerance

def test_pre_anchor_tolerance():
    strict = MetricConfig(pre_tol_s=0.0)
    # 0.5 s before the reference time: the match with delay 0 under the default
    # tolerance of 1 s, premature without tolerance
    dt = decision_timing([10.0], _em((9.5, "a")), CFG)
    approx(dt["timing_accuracy"], 100.0)
    assert dt["v_premature"] == 0
    dt = decision_timing([10.0], _em((9.5, "a")), strict)
    approx(dt["timing_accuracy"], 0.0)
    assert dt["v_premature"] == 1
    # earlier than the tolerance: premature under both
    assert decision_timing([10.0], _em((8.5, "a")), CFG)["v_premature"] == 1
    # just before the next event: belongs to the next event, not a repeat of the first
    dt = decision_timing([10.0, 20.0], _em((10.0, "a"), (19.5, "b")), CFG)
    assert dt["v_redundant"] == 0 and dt["n_matched"] == 2
    dt = decision_timing([10.0, 20.0], _em((10.0, "a"), (19.5, "b")), strict)
    assert dt["v_redundant"] == 1 and dt["n_matched"] == 1


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    main()
