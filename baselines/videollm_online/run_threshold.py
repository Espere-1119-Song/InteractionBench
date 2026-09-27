"""Run VideoLLM-online-8B-v1plus over InteractionBench with a chosen trigger threshold.

Variant of ``baselines/videollm_online/run.py`` for the trigger-threshold ablation.
Differences from ``run.py``:
  * --threshold (required, in [0, 1]) replaces the upstream value 0.725 of the
    frame-interval-token gate (``LiveInfer.frame_token_interval_threshold``). After
    each frame the model stays silent when the probability of the frame-interval
    token is at least the threshold, so a higher threshold makes the model speak
    more often.
  * an item that failed once is recorded as failed (empty emissions) when the same
    command is started again; ``run.py`` allows three failed attempts.
  * the threshold is written to config.json.

Upstream: https://github.com/showlab/videollm-online (this script imports
``demo/inference.py`` and ``data/utils.py`` from a checkout; pass its location with
--repo or the environment variable VLLMONLINE_REPO).
Checkpoint: LoRA chenjoya/videollm-online-8b-v1plus on the base LLM
NousResearch/Meta-Llama-3-8B-Instruct (a mirror of meta-llama/Meta-Llama-3-8B-Instruct)
with the vision encoder google/siglip-large-patch16-384.

Input fps: 8. Each video is resampled once with ffmpeg into a cache directory
(--cache-dir). The upstream helper calls ``./ffmpeg/ffmpeg`` relative to the working
directory, so that path must exist where the runner is started.

Environment (versions of the paper run): Python 3.10, torch 2.7.1 (CUDA 12.8),
torchvision 0.22.1, torchaudio 2.7.1, transformers 4.55.4, peft 0.20.0,
accelerate 1.14.0, av 12.3.0; see README.md.

Commands used for the paper numbers (threshold ablation on the 103-item subset; run
directories videollm_threshold0.5_0921 and videollm_threshold0.9_0921):
  python baselines/videollm_online/run_threshold.py --threshold 0.5 --mcq --fps 8 \
      --items benchmark/splits/subset103.txt --out results/runs/videollm_threshold0.5
  python baselines/videollm_online/run_threshold.py --threshold 0.9 --mcq --fps 8 \
      --items benchmark/splits/subset103.txt --out results/runs/videollm_threshold0.9

Output: <out>/preds.jsonl (one line per item, appended, resumable),
<out>/raw/<video_id>#<item_index>.json, <out>/config.json and
<out>/fail_counts.json (number of failed attempts per item).
Default <out>: results/runs/videollm-online-8b_streaming_<fps>fps[_mcq]; this is the
default directory of ``run.py`` as well, so pass --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run VideoLLM-online over InteractionBench with a chosen "
                    "trigger threshold.")
    ap.add_argument("--data", default="data/interactionbench")
    ap.add_argument("--repo",
                    default=os.environ.get("VLLMONLINE_REPO", "external/videollm-online"),
                    help="checkout of https://github.com/showlab/videollm-online "
                         "(environment variable VLLMONLINE_REPO)")
    ap.add_argument("--checkpoint", default="chenjoya/videollm-online-8b-v1plus",
                    help="LoRA checkpoint (Hugging Face id or local directory)")
    ap.add_argument("--llm", default="NousResearch/Meta-Llama-3-8B-Instruct",
                    help="base LLM (Hugging Face id or local directory)")
    ap.add_argument("--cache-dir",
                    default=os.environ.get("VLLMONLINE_CACHE", "cache/videollm_online"),
                    help="directory of the resampled videos "
                         "(environment variable VLLMONLINE_CACHE)")
    ap.add_argument("--mcq", default=None, nargs="?",
                    const="data/interactionbench/mcq/mcq_options_v4.jsonl")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--a-window", type=float, default=10.0)
    ap.add_argument("--video-dir", default="data/interactionbench/videos_h264")
    ap.add_argument("--items", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--threshold", type=float, required=True,
                    help="trigger threshold of the frame-interval-token gate, in [0, 1]")
    args, _ = ap.parse_known_args()
    assert 0 <= args.threshold <= 1

    from interactionbench.data import iter_items, load_benchmark
    from interactionbench.prompts import format_question

    VREPO = str(Path(args.repo).resolve())

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

    run_tag = f"videollm-online-8b_streaming_{args.fps}fps" + ("_mcq" if mcq else "")
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
        "system": "VideoLLM-online", "run": run_tag,
        "model": (f"{args.checkpoint} LoRA on "
                  f"{args.llm} + siglip-large-384"),
        "protocol": (f"streaming EOS gate (frame-interval token, thresh {args.threshold}); "
                     "A: query injected at question_time; B/C: query at t=0"),
        "fps": args.fps, "a_window_s": args.a_window, "threshold": args.threshold,
        "attn": "sdpa", "video_dir": str(vroot), "mcq_options": args.mcq,
        "n_items_targeted": len(ready),
        "item_set_sha1": hashlib.sha1(json.dumps(
            sorted(it.item_id for it, _ in ready)).encode()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }, indent=1))

    # LiveInfer parses CLI args itself
    sys.argv = ["run_videollm_online",
                "--resume_from_checkpoint", args.checkpoint,
                "--llm_pretrained", args.llm,
                "--attn_implementation", "sdpa",
                "--frame_fps", str(args.fps),
                "--output_dir", "/tmp/vllmonline_run"]
    cache = Path(args.cache_dir).resolve()
    cwd = os.getcwd()
    os.chdir(VREPO)
    sys.path.insert(0, VREPO)
    from data.utils import ffmpeg_once  # noqa: E402
    from demo.inference import LiveInfer  # noqa: E402
    liveinfer = LiveInfer()
    liveinfer.frame_token_interval_threshold = args.threshold
    print(f"Native gate threshold = {liveinfer.frame_token_interval_threshold}", flush=True)
    os.chdir(cwd)
    cache.mkdir(parents=True, exist_ok=True)

    print(f"{len(ready)} items | out: {out_dir}", flush=True)
    fail_fp = out_dir / "fail_counts.json"
    fail_counts = json.loads(fail_fp.read_text()) if fail_fp.exists() else {}
    for i, (it, mp4) in enumerate(ready, 1):
        if it.item_id in done:
            continue
        if fail_counts.get(it.item_id, 0) >= 1:
            # deterministic failure (a very long LCG item trips the stream-token
            # assertion): record the item as failed so that the run can terminate;
            # it is scored as silence/miss per protocol
            pred = {"video_id": it.video_id, "item_index": it.item_index,
                    "model": "videollm-online-8b", "run": run_tag,
                    "emissions": [], "n_polls": 0, "poll_latencies": [],
                    "error": f"deterministic_failure_x{fail_counts[it.item_id]}"}
            with preds_fp.open("a") as f:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            print(f"  {it.item_id}: {fail_counts[it.item_id]}x failures -> recorded as failed", flush=True)
            continue
        print(f"[{i}/{len(ready)}] {it.capability}/{it.time_type} {it.item_id} "
              f"({it.duration_s:.0f}s)", flush=True)
        question = format_question(it, mcq.get(it.item_id))
        is_A = it.time_type == "A"
        q_t = it.question_time_s or 0.0
        emissions = []
        n_frames = 0
        try:
            proc = cache / (f"{it.video_id}_{liveinfer.frame_fps}fps_"
                            f"{liveinfer.frame_resolution}.mp4")
            if not proc.exists():
                ffmpeg_once(str(mp4), str(proc), fps=liveinfer.frame_fps,
                            resolution=liveinfer.frame_resolution)
            liveinfer.reset()
            liveinfer.load_video(str(proc))
            if not is_A:
                liveinfer.input_query_stream(question, video_time=0.0)
            n_video = liveinfer.num_video_frames
            end_f = n_video
            if is_A:
                end_f = min(n_video,
                            int((q_t + args.a_window) * liveinfer.frame_fps))
            asked = not is_A
            for k in range(end_f):
                t = k / liveinfer.frame_fps
                if is_A and not asked and t >= q_t - 1e-6:
                    liveinfer.input_query_stream(question, video_time=t)
                    asked = True
                t0 = time.perf_counter()
                liveinfer.input_video_stream(t)
                _, response = liveinfer()
                lat = time.perf_counter() - t0
                n_frames += 1
                if response:
                    txt = response.split("Assistant:", 1)[-1].strip()
                    emissions.append({"t": round(liveinfer.video_time, 2),
                                      "content": txt,
                                      "latency_s": round(lat, 3)})
        except Exception as e:
            print(f"  ERROR {it.item_id}: {str(e)[:200]}", flush=True)
            fail_counts[it.item_id] = fail_counts.get(it.item_id, 0) + 1
            fail_fp.write_text(json.dumps(fail_counts))
            continue

        pred = {"video_id": it.video_id, "item_index": it.item_index,
                "model": "videollm-online-8b", "run": run_tag,
                "emissions": emissions, "n_polls": n_frames,
                "poll_latencies": []}
        with preds_fp.open("a") as f:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        (raw_dir / f"{it.video_id}#{it.item_index}.json").write_text(
            json.dumps({"item_id": it.item_id, "question": question,
                        "capability": it.capability,
                        "time_type": it.time_type,
                        "n_frames": n_frames,
                        "emissions": emissions}, indent=1, ensure_ascii=False))
        if emissions:
            print(f"    {len(emissions)} emissions, first t={emissions[0]['t']}",
                  flush=True)

    print(f"\nDONE -> {preds_fp}", flush=True)


if __name__ == "__main__":
    main()
