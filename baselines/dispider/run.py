"""Run Dispider (CVPR25) over InteractionBench, offline-grounding track.

IMPORTANT — protocol caveat. Dispider's paper is about *active real-time
interaction* (a decision module choosing when to respond), but its released
inference API exposes only whole-video single-shot QA: `videoStream.Run(video,
prompt)` returns one string, and the decision machinery (`ans_position`,
`silent_position`) are *inputs* to generate(), never outputs. There is no
public way to read back "when did it decide to speak". We therefore evaluate
Dispider on the SAME offline temporal-grounding track as the Qwen3-VL offline
runs: it is asked, in the phrasing it was trained on, at what second(s) it
would respond, and its claimed timestamps become emission times. Its scores
are thus comparable to the other offline systems, NOT to the streaming ones,
and they do not measure the proactive decision loop described in the paper.

Answers arrive as spelled-out numbers ("at seventy-seven seconds"), so both
digit and word forms are parsed.

Upstream:
  code        https://github.com/Mark12Ding/Dispider
              --dispider-repo (or the environment variable DISPIDER_REPO),
              default external/Dispider. The runner imports inference.py from
              this checkout.
  checkpoint  Mar2Ding/Dispider (https://huggingface.co/Mar2Ding/Dispider)
              --model (or the environment variable DISPIDER_MODEL) takes a
              repository id or a local directory. The paper run used a local
              copy whose config.json pointed to a local compressor directory.

Environment:
  A dedicated environment: python 3.10, torch 2.2 (cu118), flash-attn 2.5.9,
  numpy<2, decord. decord cannot read AV1 video, so --video-dir defaults to the
  H.264 proxy directory. See README.md in this directory.

Command used for the paper numbers (218-item subset, earlier multiple-choice
file; see README.md):
  python baselines/dispider/run.py \\
      --items benchmark/splits/frozen218.txt --mcq

Output:
  results/runs/dispider_offline[_mcq]/preds.jsonl   (or under --out)
  results/runs/dispider_offline[_mcq]/raw/<item_id>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from interactionbench.data import iter_items, load_benchmark  # noqa: E402
from interactionbench.prompts import format_question  # noqa: E402

MODEL = "Mar2Ding/Dispider"

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
          "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
          "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}


def words_to_number(text: str) -> list[float]:
    """Spelled-out seconds -> numbers ('eighty - ninety-three' -> [80, 93])."""
    toks = re.split(r"[\s\-]+", text.lower())
    out, cur, pending, has_tens, has_unit = [], 0, False, False, False

    def flush():
        nonlocal cur, pending, has_tens, has_unit
        if pending:
            out.append(float(cur))
        cur, pending, has_tens, has_unit = 0, False, False, False

    for t in toks:
        t = t.strip(".,;:")
        if t in _TENS:
            if has_tens or has_unit:   # 'eighty ninety-three' = two numbers
                flush()
            cur += _TENS[t]
            pending = has_tens = True
        elif t in _UNITS:
            if has_unit:               # 'seven eight' = two numbers
                flush()
            cur += _UNITS[t]
            pending = has_unit = True
        elif t == "hundred" and pending:
            cur *= 100                 # 'one hundred twenty' keeps accumulating
            has_tens = has_unit = False
        elif t == "and" and pending:
            continue                   # 'one hundred and five' is one number
        else:
            flush()
    flush()
    return out


_RANGE_SEP = re.compile(r"(?:\bto\b|[-–—])")


def parse_times(text: str) -> list[float]:
    digits = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:s\b|sec|second)", text.lower())]
    bare = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\b", text)]
    # "from 10 to 20 seconds": only the second number carries the unit, so the
    # unit-anchored pass would miss the range start
    if digits and not (len(bare) > len(digits) and _RANGE_SEP.search(text)):
        times = digits
    else:
        times = bare if bare else words_to_number(text)
    # Dispider answers with intervals ("fifty-seven - sixty seconds"). An
    # interval is ONE event, and the moment it claims to respond is its start —
    # keeping both ends would double-count responses and unfairly cost Silence
    # Compliance. Collapse pairs whenever the text uses a range separator.
    if len(times) >= 2 and len(times) % 2 == 0 and _RANGE_SEP.search(text):
        collapsed = times[::2]
        if all(a <= b for a, b in zip(times[::2], times[1::2])):
            return collapsed
    return times


GROUND_PROMPT = ('{question}\n\nAt what time in seconds does this happen? '
                 'If it happens several times, state every time in seconds.')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/interactionbench")
    ap.add_argument("--model", default=os.environ.get("DISPIDER_MODEL", MODEL),
                    help="checkpoint repository id or local directory "
                         "(environment variable DISPIDER_MODEL)")
    ap.add_argument("--dispider-repo",
                    default=os.environ.get("DISPIDER_REPO", "external/Dispider"),
                    help="checkout of https://github.com/Mark12Ding/Dispider "
                         "(environment variable DISPIDER_REPO)")
    ap.add_argument("--items", default=None)
    ap.add_argument("--mcq", default=None, nargs="?",
                    const="data/interactionbench/mcq/mcq_options_v4.jsonl")
    ap.add_argument("--video-dir", default="data/interactionbench/videos_h264")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    dispider_repo = Path(args.dispider_repo).resolve()
    if not dispider_repo.is_dir():
        sys.exit(f"Dispider checkout not found: {dispider_repo} "
                 "(set --dispider-repo or DISPIDER_REPO)")
    sys.path.insert(0, str(dispider_repo))

    mcq: dict[str, list[str]] = {}
    if args.mcq:
        for line in Path(args.mcq).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                mcq[d["item_id"]] = d

    videos = load_benchmark(args.data)
    items = list(iter_items(videos))
    if args.items:
        keep = {l.strip() for l in Path(args.items).read_text().splitlines() if l.strip()}
        items = [it for it in items if it.item_id in keep]
    vroot = Path(args.video_dir)
    ready = [(it, vroot / f"{it.video_id}.mp4") for it in items]
    ready = [(it, p) for it, p in ready if p.exists() and p.stat().st_size > 0]
    if args.limit:
        ready = ready[: args.limit]
    if not ready:
        sys.exit("no runnable items")

    run_tag = "dispider_offline" + ("_mcq" if mcq else "")
    out_dir = Path(args.out) if args.out else Path("results/runs") / run_tag
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    preds_fp = out_dir / "preds.jsonl"
    done = set()
    if preds_fp.exists() and not args.overwrite:
        done = {f"{json.loads(l)['video_id']}#{json.loads(l)['item_index']}"
                for l in preds_fp.read_text(encoding="utf-8").splitlines() if l.strip()}
    elif preds_fp.exists():
        preds_fp.unlink()

    print(f"loading Dispider ...", flush=True)
    from inference import videoStream
    stream = videoStream(args.model)
    print(f"{len(ready)} items | out: {out_dir}", flush=True)

    for i, (it, mp4) in enumerate(ready, 1):
        if it.item_id in done:
            print(f"[{i}/{len(ready)}] {it.item_id} -> skip", flush=True)
            continue
        print(f"[{i}/{len(ready)}] {it.capability}/{it.time_type} {it.item_id}", flush=True)
        question = format_question(it, mcq.get(it.item_id))
        is_A = it.time_type == "A"

        def ask(prompt: str) -> tuple[str, float]:
            t0 = time.perf_counter()
            try:
                out = stream.Run(str(mp4), prompt)
            except Exception as e:
                print(f"  ERROR {it.item_id}: {str(e)[:150]}", flush=True)
                out = ""
            return (out or "").strip(), time.perf_counter() - t0

        # Dispider's grounding answer states only WHEN ("the event happens in
        # 57-60 seconds"), never WHAT — scoring that text as the response would
        # zero its Accuracy for protocol reasons rather than capability ones.
        # So B/C items get two calls: the question itself for content, the
        # grounding phrasing for timing. A-type needs only the question (it is
        # answered at question_time by definition).
        answer, lat_a = ask(question)
        if is_A:
            text, lat = answer, lat_a
            emissions = ([{"t": it.question_time_s, "content": answer,
                           "latency_s": round(lat_a, 3)}] if answer else [])
        else:
            text, lat_t = ask(GROUND_PROMPT.format(question=question))
            lat = lat_a + lat_t
            times = [t for t in parse_times(text) if 0 <= t <= it.duration_s]
            emissions = [{"t": round(t, 2), "content": answer,
                          "latency_s": round(lat, 3)} for t in sorted(set(times))]
        if args.verbose:
            print(f"    [{lat:.1f}s] when={text[:55]!r} what={answer[:45]!r} "
                  f"-> {len(emissions)} emissions", flush=True)

        pred = {"video_id": it.video_id, "item_index": it.item_index,
                "model": "dispider", "run": run_tag,
                "emissions": emissions, "n_polls": 1,
                "poll_latencies": [round(lat, 3)]}
        with preds_fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        (raw_dir / f"{it.video_id}#{it.item_index}.json").write_text(
            json.dumps({"item_id": it.item_id, "question": question,
                        "capability": it.capability, "time_type": it.time_type,
                        "answer_output": answer, "timing_output": text,
                        "latency_s": lat}, indent=1,
                       ensure_ascii=False), encoding="utf-8")

    print(f"\nDONE -> {preds_fp}", flush=True)


if __name__ == "__main__":
    main()
