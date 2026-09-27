"""Run MOSS-Video-Preview (realtime-SFT) natively-streaming over InteractionBench.

MOSS consumes frames from an image_queue at REAL-TIME pace (we sleep 1/fps per
frame, as in the official realtime_streaming_infer.py) and emits tokens
continuously: <|silence|> while observing, text tokens when speaking. Stream
time is therefore wall-clock since feed start; an emission's t is the moment
its first token appears. This measures the system's genuine real-time
behaviour (a 60s video takes ~60s to evaluate).

Item protocol (mirrors the other runners):
  B/C  prompt pushed ~1s after feed start (official demo timing)
  A    prompt pushed when the stream reaches question_time_s; feed stops at
       question_time + --a-window

Upstream:
  code        https://github.com/OpenMOSS/MOSS-Video-Preview
  checkpoint  OpenMOSS-Team/moss-video-preview-realtime-sft
              (https://huggingface.co/OpenMOSS-Team/moss-video-preview-realtime-sft)
  The model code is loaded from the checkpoint with trust_remote_code=True, so
  no checkout of the code repository is needed. --model (or the environment
  variable MOSS_MODEL) takes a repository id or a local directory.

Environment:
  A dedicated environment with torch 2.4 and transformers 4.46, plus the `av`
  package (required by the remote code of the checkpoint) and an ffmpeg binary
  on PATH. See README.md in this directory.

Command used for the paper numbers (218-item subset, earlier multiple-choice
file; see README.md):
  python baselines/moss/run.py \\
      --items benchmark/splits/frozen218.txt --mcq

Output:
  results/runs/moss-video-preview_streaming_rt[_mcq]/preds.jsonl   (or under --out)
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from interactionbench.frames import extract_frames  # noqa: E402
from interactionbench.data import iter_items, load_benchmark  # noqa: E402
from interactionbench.prompts import format_question  # noqa: E402

CKPT = "OpenMOSS-Team/moss-video-preview-realtime-sft"
END_MARKERS = {"[DONE]", "[ERROR]", "<|round_end|>"}


def stream_item(model, processor, item, frames, *, fps: float, a_window: float,
                max_new_tokens: int, verbose: bool) -> dict:
    image_q: queue.Queue = queue.Queue()
    prompt_q: queue.Queue = queue.Queue()
    token_q: queue.Queue = queue.Queue()

    is_A = item.time_type == "A"
    q_t = item.question_time_s if is_A else 1.0
    end_s = min(item.duration_s, item.question_time_s + a_window) if is_A \
        else item.duration_s
    feed = [f for f in frames if f.time <= end_s]
    question = format_question(item, stream_item.mcq.get(item.item_id))

    t0 = time.perf_counter()
    stop_feed = threading.Event()

    def feeder():
        sent_prompt = False
        for f in feed:
            if stop_feed.is_set():
                return
            # pace the stream: wait until wall clock reaches this frame's time
            delay = f.time - (time.perf_counter() - t0)
            if delay > 0:
                time.sleep(delay)
            if not sent_prompt and (time.perf_counter() - t0) >= q_t - 1e-3:
                prompt_q.put(question)
                sent_prompt = True
            image_q.put(f.image)
        if not sent_prompt:  # question lands at/after the last frame (A-type)
            prompt_q.put(question)

    gen_err = []

    def generator():
        try:
            model.real_time_generate(image_q, prompt_q, token_q, processor,
                                     max_new_tokens=max_new_tokens,
                                     do_sample=False)
        except Exception as e:  # surfaced in the result
            gen_err.append(str(e)[:300])

    threading.Thread(target=feeder, daemon=True).start()
    threading.Thread(target=generator, daemon=True).start()

    # Consume tokens; group into timed utterances.
    emissions = []
    cur_tokens: list[str] = []
    cur_t = None
    feed_wall = end_s + 8.0  # listen until stream end + grace
    while (time.perf_counter() - t0) < feed_wall:
        try:
            tok = token_q.get(timeout=0.25)
        except queue.Empty:
            continue
        now = time.perf_counter() - t0
        if tok == "<|silence|>" or tok in END_MARKERS or tok == "<|round_start|>":
            if cur_tokens:
                emissions.append({"t": round(cur_t, 2),
                                  "content": "".join(cur_tokens).strip()})
                if verbose:
                    print(f"    t={cur_t:6.1f}s SPEAK: "
                          f"{''.join(cur_tokens).strip()[:80]}", flush=True)
                cur_tokens, cur_t = [], None
            if tok == "[ERROR]":
                break
            continue
        if cur_t is None:
            cur_t = now
        cur_tokens.append(tok)
    if cur_tokens:
        emissions.append({"t": round(cur_t, 2),
                          "content": "".join(cur_tokens).strip()})
    # strip special markers; drop pre-prompt chatter (the official listener
    # discards tokens generated before the prompt round starts)
    import re as _re
    cleaned = []
    for e in emissions:
        c = _re.sub(r"<\|[^|>]{1,30}\|>", "", e["content"]).strip()
        if c and e["t"] >= q_t - 0.2:
            cleaned.append({"t": e["t"], "content": c})
    emissions = cleaned
    stop_feed.set()
    if hasattr(model, "stop_real_time_generate"):
        try:
            model.stop_real_time_generate()
        except Exception:
            pass
    time.sleep(0.5)
    emissions = [e for e in emissions if e["content"]]
    return {"emissions": emissions, "n_polls": len(feed),
            "gen_error": gen_err[0] if gen_err else None}


stream_item.mcq = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/interactionbench")
    ap.add_argument("--model", default=os.environ.get("MOSS_MODEL", CKPT),
                    help="checkpoint repository id or local directory "
                         "(environment variable MOSS_MODEL)")
    ap.add_argument("--items", default=None)
    ap.add_argument("--mcq", default=None, nargs="?",
                    const="data/interactionbench/mcq/mcq_options_v4.jsonl")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--a-window", type=float, default=10.0)
    ap.add_argument("--max-new-tokens", type=int, default=86400)
    ap.add_argument("--max-long-side", type=int, default=448)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.mcq:
        for line in Path(args.mcq).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                stream_item.mcq[d["item_id"]] = d

    videos = load_benchmark(args.data)
    items = list(iter_items(videos))
    if args.items:
        keep = {l.strip() for l in Path(args.items).read_text().splitlines() if l.strip()}
        items = [it for it in items if it.item_id in keep]
    vroot = Path(args.data) / "videos"
    ready = [(it, vroot / it.domain / f"{it.video_id}.mp4") for it in items]
    ready = [(it, p) for it, p in ready if p.exists() and p.stat().st_size > 0]
    if args.limit:
        ready = ready[: args.limit]
    if not ready:
        sys.exit("no runnable items")

    run_tag = "moss-video-preview_streaming_rt" + ("_mcq" if stream_item.mcq else "")
    out_dir = Path(args.out) if args.out else Path("results/runs") / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_fp = out_dir / "preds.jsonl"
    done = set()
    if preds_fp.exists() and not args.overwrite:
        done = {f"{json.loads(l)['video_id']}#{json.loads(l)['item_index']}"
                for l in preds_fp.read_text(encoding="utf-8").splitlines() if l.strip()}
    elif preds_fp.exists():
        preds_fp.unlink()

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    print(f"loading {args.model} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True,
                                              frame_extract_num_threads=1)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, device_map="cuda:0",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()

    frames_cache: dict[str, list] = {}
    print(f"{len(ready)} items | out: {out_dir} | REAL-TIME pacing "
          f"(~{sum(min(i.duration_s, i.question_time_s + args.a_window) if i.time_type == 'A' else i.duration_s for i, _ in ready) / 3600:.1f}h of stream)",
          flush=True)
    for i, (it, mp4) in enumerate(ready, 1):
        if it.item_id in done:
            print(f"[{i}/{len(ready)}] {it.item_id} -> skip", flush=True)
            continue
        print(f"[{i}/{len(ready)}] {it.capability}/{it.time_type} {it.item_id} "
              f"({it.duration_s:.0f}s)", flush=True)
        if it.video_id not in frames_cache:
            frames_cache.clear()
            frames_cache[it.video_id] = extract_frames(
                mp4, sample_fps=args.fps, max_long_side=args.max_long_side)
        res = stream_item(model, processor, it, frames_cache[it.video_id],
                          fps=args.fps, a_window=args.a_window,
                          max_new_tokens=args.max_new_tokens,
                          verbose=args.verbose)
        pred = {"video_id": it.video_id, "item_index": it.item_index,
                "model": "moss-video-preview", "run": run_tag,
                "emissions": res["emissions"], "n_polls": res["n_polls"]}
        if res["gen_error"]:
            pred["gen_error"] = res["gen_error"]
        with preds_fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"\nDONE -> {preds_fp}", flush=True)


if __name__ == "__main__":
    main()
