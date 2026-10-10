"""Run Dispider (CVPR25) over InteractionBench, offline-grounding track."""

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
            if has_tens or has_unit:
                flush()
            cur += _TENS[t]
            pending = has_tens = True
        elif t in _UNITS:
            if has_unit:
                flush()
            cur += _UNITS[t]
            pending = has_unit = True
        elif t == "hundred" and pending:
            cur *= 100
            has_tens = has_unit = False
        elif t == "and" and pending:
            continue
        else:
            flush()
    flush()
    return out


_RANGE_SEP = re.compile(r"(?:\bto\b|[-–—])")


def parse_times(text: str) -> list[float]:
    digits = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:s\b|sec|second)", text.lower())]
    bare = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\b", text)]
    if digits and not (len(bare) > len(digits) and _RANGE_SEP.search(text)):
        times = digits
    else:
        times = bare if bare else words_to_number(text)
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
