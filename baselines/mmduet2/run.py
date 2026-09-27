"""Run MMDuet2 over InteractionBench.

MMDuet2 is a *proactive* streaming model: frames stream in (fps/decision_hz frames
per turn, at least 2), and after each turn the model either replies or emits
"NO REPLY". The reply moment is the decision signal; `time` is tracked natively by
the upstream client.

Item protocol (same as the other runners):
  B/C  the standing request is injected as text before the first frame
  A    frames stream silently until question_time_s, then the question text is
       injected; polling continues to question_time + --a-window

Upstream: https://github.com/yellow-binary-tree/MMDuet2 (this script imports
``demo/api_server.py`` and ``proactive_eval/`` from a checkout; pass its location with
--repo or the environment variable MMDUET2_REPO).
Checkpoint: wangyueqian/MMDuet2 (Hugging Face).

Environment (versions of the paper run): Python 3.10, torch 2.7.1 (CUDA 12.8),
torchvision 0.22.1, transformers 4.49.0, qwen-vl-utils[decord] 0.0.8, decord 0.6.0,
accelerate 1.14.0, gradio 5.47.2; see README.md.

Command used for the paper numbers (run directory
mmduet2-3b_streaming_4fps_mcq_MERGED; the run was split into shards with --items and
merged with ``ibench merge``):
  python baselines/mmduet2/run.py --mcq --fps 4 --decision-hz 0.5 --max-frames 880 \
      --out results/runs/mmduet2-3b_streaming_4fps_mcq

Output: <out>/preds.jsonl (one line per item, appended, resumable),
<out>/raw/<video_id>#<item_index>.json and <out>/config.json (the experiment setting).
Default <out>: results/runs/mmduet2-3b_streaming_<fps>fps[_mcq].
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

MODEL = "wangyueqian/MMDuet2"
LETTERS = "ABCDEF"


def format_question(it, opt) -> str:
    if not opt:
        return it.question
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opt["options"]))
    stem = opt.get("stem") or it.question
    return (f"{stem}\n{opts}\n"
            f"(When you speak, answer with just the option letter.)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run MMDuet2 over InteractionBench.")
    ap.add_argument("--data", default="data/interactionbench")
    ap.add_argument("--repo", default=os.environ.get("MMDUET2_REPO", "external/MMDuet2"),
                    help="checkout of https://github.com/yellow-binary-tree/MMDuet2 "
                         "(environment variable MMDUET2_REPO)")
    ap.add_argument("--checkpoint", default=MODEL,
                    help="Hugging Face id or local directory of the MMDuet2 checkpoint")
    ap.add_argument("--mcq", default=None, nargs="?",
                    const="data/interactionbench/mcq/mcq_options_v4.jsonl")
    ap.add_argument("--fps", default="auto",
                    help="input fps: number, or 'auto' = highest of 16/8/4 "
                         "whose dur*fps fits --max-frames (context guard)")
    ap.add_argument("--decision-hz", type=float, default=1.0,
                    help="reply/silence decisions per second: "
                         "num_frames_per_turn = fps/decision_hz (visual input "
                         "stays at full fps; only the decision cadence changes)")
    ap.add_argument("--a-window", type=float, default=10.0)
    ap.add_argument("--video-dir", default="data/interactionbench/videos_h264")
    ap.add_argument("--items", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=1200,
                    help="hard cap on frames per item (KV-cache guard)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    MMD = str(Path(args.repo).resolve())
    sys.path.insert(0, f"{MMD}/proactive_eval")
    sys.path.insert(0, f"{MMD}/demo")

    from interactionbench.data import iter_items, load_benchmark

    mcq = {}
    if args.mcq:
        for line in Path(args.mcq).read_text().splitlines():
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

    auto = str(args.fps) == "auto"
    fps_tag = "auto" if auto else str(args.fps).rstrip("0").rstrip(".").replace(".", "p")
    run_tag = f"mmduet2-3b_streaming_{fps_tag}fps" + ("_mcq" if mcq else "")
    out_dir = Path(args.out) if args.out else Path("results/runs") / run_tag
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    preds_fp = out_dir / "preds.jsonl"
    done = set()
    if preds_fp.exists() and not args.overwrite:
        done = {f"{json.loads(l)['video_id']}#{json.loads(l)['item_index']}"
                for l in preds_fp.read_text().splitlines() if l.strip()}
    elif preds_fp.exists():
        preds_fp.unlink()

    (out_dir / "config.json").write_text(json.dumps({
        "system": "MMDuet2", "model": args.checkpoint, "run": run_tag,
        "protocol": "streaming proactive (NO-REPLY silence), 2 frames/turn",
        "fps": args.fps,
        "fps_policy": ("adaptive: highest of 16/8/4 with dur*fps<=max_frames "
                       "(32k-context guard); per-item fps in raw/" if auto
                       else "fixed"),
        "decision_hz": args.decision_hz,
        "a_window_s": args.a_window,
        "max_frames": args.max_frames, "video_dir": str(vroot),
        "mcq_options": args.mcq, "n_items_targeted": len(ready),
        "item_set_sha1": hashlib.sha1(json.dumps(
            sorted(it.item_id for it, _ in ready)).encode()).hexdigest(),
        "attn": "sdpa", "started_at": datetime.now(timezone.utc).isoformat(),
    }, indent=1))

    print(f"loading {args.checkpoint} ...", flush=True)
    from api_server import (ProactiveInferenceAPIClient,  # noqa: E402
                            ProactiveTestAPIArguments)
    margs = ProactiveTestAPIArguments(
        output_dir="/tmp/mmduet2_run", llm_pretrained=args.checkpoint,
        attn_implementation="sdpa", num_frames_per_turn=2)
    client = ProactiveInferenceAPIClient(margs)
    from decord import VideoReader  # noqa: E402  (must import after CUDA init)
    from PIL import Image  # noqa: E402

    print(f"{len(ready)} items | out: {out_dir}", flush=True)
    for i, (it, mp4) in enumerate(ready, 1):
        if it.item_id in done:
            continue
        print(f"[{i}/{len(ready)}] {it.capability}/{it.time_type} {it.item_id} "
              f"({it.duration_s:.0f}s)", flush=True)
        question = format_question(it, mcq.get(it.item_id))
        is_A = it.time_type == "A"
        q_t = it.question_time_s or 0.0
        emissions, polls = [], []
        try:
            vr = VideoReader(str(mp4))
            vfps = vr.get_avg_fps()
            dur = len(vr) / vfps
            end_s = min(dur, q_t + args.a_window) if is_A else dur
            if auto:
                item_fps = next((f for f in (16.0, 8.0, 4.0)
                                 if end_s * f <= args.max_frames), 4.0)
            else:
                item_fps = float(args.fps)
            client.set_fps(frame_interval=1.0 / item_fps)
            client.num_frames_per_turn = max(
                2, int(round(item_fps / args.decision_hz)))
            client.reset()
            if not is_A:
                client.add_text(question)
            asked = not is_A
            step = 1.0 / item_fps
            t, n_fed = 0.0, 0
            while t <= end_s + 1e-6 and n_fed < args.max_frames:
                if is_A and not asked and t >= q_t - 1e-6:
                    client.add_text(question)
                    asked = True
                idx = min(int(t * vfps), len(vr) - 1)
                img = Image.fromarray(vr[idx].asnumpy())
                img.thumbnail((448, 448))
                t0 = time.perf_counter()
                # api_server prints a debug line per frame; mute it
                with contextlib.redirect_stdout(io.StringIO()):
                    r = client.add_image(img)
                lat = time.perf_counter() - t0
                n_fed += 1
                if "response" in r or "respsone" in r:
                    polls.append({"t": round(t, 2), "latency_s": round(lat, 3),
                                  "replied": bool(r.get("response"))})
                if r.get("response"):
                    emissions.append({"t": round(float(r["time"]), 2),
                                      "content": (r.get("content") or "").strip(),
                                      "latency_s": round(lat, 3)})
                t += step
        except Exception as e:
            print(f"  ERROR {it.item_id}: {str(e)[:200]}", flush=True)

        pred = {"video_id": it.video_id, "item_index": it.item_index,
                "model": "mmduet2-3b", "run": run_tag,
                "emissions": emissions, "n_polls": len(polls),
                "poll_latencies": [p["latency_s"] for p in polls]}
        with preds_fp.open("a") as f:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        (raw_dir / f"{it.video_id}#{it.item_index}.json").write_text(
            json.dumps({"item_id": it.item_id, "question": question,
                        "capability": it.capability, "time_type": it.time_type,
                        "fps": item_fps if "item_fps" in dir() else None,
                        "polls": polls}, indent=1, ensure_ascii=False))

    print(f"\nDONE -> {preds_fp}", flush=True)


if __name__ == "__main__":
    main()
