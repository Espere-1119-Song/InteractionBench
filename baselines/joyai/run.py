"""Run the JoyAI-VL-Interaction streaming system over InteractionBench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

WATCH_PLACEHOLDER = "(No request yet. Keep watching silently.)"


def capability_roundrobin(items, cap=None, iid=None):
    cap = cap or (lambda it: it.get("capability") or "?")
    iid = iid or (lambda it: it.get("item_id") or "")
    groups: dict[str, list] = {}
    for it in sorted(items, key=iid):
        groups.setdefault(cap(it), []).append(it)
    queues = [groups[k] for k in sorted(groups)]
    out, i = [], 0
    while any(queues):
        q = queues[i % len(queues)]
        if q:
            out.append(q.pop(0))
        i += 1
    return out


def stream_item(item, frames, *, interval: float, a_window: float,
                max_new_tokens: int, ingest_chunk: int, base: str, model: str,
                options=None, verbose=False) -> list[dict]:
    from client import JoyAISession, parse_marker
    from interactionbench.frames import frames_up_to
    from interactionbench.prompts import format_question

    sess = JoyAISession(base=base, model=model, frame_dt=1.0 / max(interval, 1e-6))
    question = format_question(item, options)
    polls = []

    if item.time_type == "A":
        q_t = item.question_time_s
        history = frames_up_to(frames, q_t)
        held = history[-1:] if history else []
        body = history[:-1]
        for i in range(0, len(body), ingest_chunk):
            sess.step(body[i:i + ingest_chunk], WATCH_PLACEHOLDER, max_new_tokens=8)
        t_end = min(item.duration_s, q_t + a_window)
        t, prev = q_t, q_t
        first = True
        while t <= t_end + 1e-6:
            new = held if first else [f for f in frames if prev < f.time <= t]
            raw, dt = sess.step(new, question, max_new_tokens=max_new_tokens)
            spoke, resp = parse_marker(raw)
            polls.append({"t": round(t, 3), "spoke": spoke, "response": resp,
                          "latency_s": round(dt, 3), "n_images": len(new), "raw": raw})
            if verbose:
                print(f"    t={t:7.2f}s [{len(new):2d} frm {dt:5.2f}s] "
                      f"{('SPEAK: ' + str(resp)) if spoke else 'wait'}", flush=True)
            prev, t, first = t, t + interval, False
        return polls

    t, prev = interval, 0.0
    while t <= item.duration_s + 1e-6:
        new = [f for f in frames if prev < f.time <= t]
        if not new:
            t += interval
            continue
        raw, dt = sess.step(new, question, max_new_tokens=max_new_tokens)
        spoke, resp = parse_marker(raw)
        polls.append({"t": round(t, 3), "spoke": spoke, "response": resp,
                      "latency_s": round(dt, 3), "n_images": len(new), "raw": raw})
        if verbose:
            print(f"    t={t:7.2f}s [{len(new):2d} frm {dt:5.2f}s] "
                  f"{('SPEAK: ' + str(resp)) if spoke else 'wait'}", flush=True)
        prev = t
        t += interval
    return polls


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run JoyAI-VL-Interaction (live adapter) over InteractionBench.")
    ap.add_argument("--data", default="data/interactionbench")
    ap.add_argument("--base", default="http://127.0.0.1:8070/v1",
                    help="root URL of the live adapter")
    ap.add_argument("--served-model", default="JoyAI-VL-Interaction-Preview",
                    help="model name sent in the request body")
    ap.add_argument("--model-key", default="joyai",
                    help="model name recorded in the predictions / run tag")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--a-window", type=float, default=10.0)
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--max-long-side", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--ingest-chunk", type=int, default=32,
                    help="history frames per ingest step for A-type items")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--video-dir", default=None,
                    help="flat video dir (<video_id>.mp4) overriding <data>/videos/<domain>/")
    ap.add_argument("--items", default=None,
                    help="file with one item_id per line")
    ap.add_argument("--capabilities", nargs="*", default=None)
    ap.add_argument("--mcq", default=None, nargs="?",
                    const="data/interactionbench/mcq/mcq_options_v4.jsonl",
                    help="multiple-choice options file")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from interactionbench.data import iter_items, load_benchmark
    from interactionbench.frames import extract_frames
    from interactionbench.mcq import load_mcq_options

    mcq: dict[str, dict] = {}
    if args.mcq:
        mcq = load_mcq_options(args.mcq)

    videos = load_benchmark(args.data)
    items = list(iter_items(videos))
    items = capability_roundrobin(items, cap=lambda it: it.capability,
                                  iid=lambda it: it.item_id)
    if args.videos:
        keep = set(args.videos)
        items = [it for it in items if it.video_id in keep]
    if args.items:
        keep = {l.strip() for l in Path(args.items).read_text().splitlines() if l.strip()}
        items = [it for it in items if it.item_id in keep]
    if args.capabilities:
        keep = set(args.capabilities)
        items = [it for it in items if it.capability in keep]
    if args.video_dir:
        vroot = Path(args.video_dir)
        ready = [(it, vroot / f"{it.video_id}.mp4") for it in items]
    else:
        vroot = Path(args.data) / "videos"
        ready = [(it, vroot / it.domain / f"{it.video_id}.mp4") for it in items]
    ready = [(it, p) for it, p in ready if p.exists() and p.stat().st_size > 0]
    if args.limit:
        ready = ready[: args.limit]
    if not ready:
        sys.exit("no runnable items")

    run_tag = (f"{args.model_key}_streaming_iv{args.interval:g}"
               + ("_mcq" if mcq else ""))
    out_dir = Path(args.out) if args.out else Path("results/runs") / run_tag
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    preds_fp = out_dir / "preds.jsonl"
    done_ids = set()
    if preds_fp.exists() and not args.overwrite:
        done_ids = {f"{json.loads(l)['video_id']}#{json.loads(l)['item_index']}"
                    for l in preds_fp.read_text(encoding="utf-8").splitlines() if l.strip()}
    elif preds_fp.exists():
        preds_fp.unlink()

    print(f"{len(ready)} items | adapter {args.base} | out {out_dir}", flush=True)
    frames_cache: dict[str, list] = {}
    for i, (it, mp4) in enumerate(ready, 1):
        if it.item_id in done_ids:
            print(f"[{i}/{len(ready)}] {it.item_id} -> skip (done)", flush=True)
            continue
        print(f"[{i}/{len(ready)}] {it.capability}/{it.time_type} {it.item_id} "
              f"({it.duration_s:.0f}s)", flush=True)
        if it.video_id not in frames_cache:
            frames_cache.clear()
            frames_cache[it.video_id] = extract_frames(
                mp4, sample_fps=args.sample_fps, max_long_side=args.max_long_side)
        try:
            polls = stream_item(it, frames_cache[it.video_id],
                                interval=args.interval, a_window=args.a_window,
                                max_new_tokens=args.max_new_tokens,
                                ingest_chunk=args.ingest_chunk,
                                base=args.base, model=args.served_model,
                                options=mcq.get(it.item_id), verbose=args.verbose)
        except Exception as e:
            print(f"  ERROR {it.item_id}: {e}", flush=True)
            continue
        pred = {"video_id": it.video_id, "item_index": it.item_index,
                "model": args.model_key, "run": run_tag,
                "emissions": [{"t": p["t"], "content": p["response"],
                               "latency_s": p["latency_s"]}
                              for p in polls if p["spoke"] and p["response"]],
                "n_polls": len(polls),
                "poll_latencies": [p["latency_s"] for p in polls]}
        with preds_fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        (raw_dir / f"{it.video_id}#{it.item_index}.json").write_text(
            json.dumps({"item_id": it.item_id, "question": it.question,
                        "capability": it.capability, "time_type": it.time_type,
                        "polls": polls}, indent=1, ensure_ascii=False),
            encoding="utf-8")

    print(f"\nDONE -> {preds_fp}", flush=True)


if __name__ == "__main__":
    main()
