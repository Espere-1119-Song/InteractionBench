"""Run VideoChat3-4B (MCG-NJU) natively-streaming over InteractionBench.

VideoChat3 is a proactive streaming VLM: per ~1s round it emits </Silence>,
</Standby> (event in progress; next frame gets 2x resolution, per the official
demo), or </Response> <text>. We drive its official StreamingSession:
  - B/C items: standing question from round 0 (global_question=True)
  - A items:   question_time = reveal round; earlier rounds ingest frames and
               auto-emit </Silence> without generation (native late-question)

Greedy decoding (eval determinism). Emissions/latency recorded in the same
predictions schema as `ibench run`.

Upstream:
  code        https://github.com/MCG-NJU/VideoChat3
  checkpoint  MCG-NJU/VideoChat3-4B (https://huggingface.co/MCG-NJU/VideoChat3-4B)
  The streaming classes (inference_fast_vc3.py, demo_vc3_proactive.py) are
  imported from the downloaded checkpoint snapshot, so no checkout of the code
  repository is needed. --model takes a Hugging Face repository id (the runner
  calls snapshot_download on it).

Environment:
  torch, transformers, accelerate, qwen-vl-utils, huggingface_hub, and an ffmpeg
  binary on PATH. The original runner records no version pins. The remote code
  of the checkpoint imports BASE_VIDEO_PROCESSOR_DOCSTRING from
  transformers.video_processing_utils, which transformers 5.16.0 does not
  provide. See README.md in this directory.

Command used for the paper numbers (218-item subset, earlier multiple-choice
file; see README.md):
  python baselines/videochat3/run.py \\
      --items benchmark/splits/frozen218.txt --mcq

Output:
  results/runs/videochat3-4b_streaming_iv1[_mcq]/preds.jsonl   (or under --out)
  results/runs/videochat3-4b_streaming_iv1[_mcq]/raw/<item_id>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from interactionbench.frames import extract_frames  # noqa: E402
from interactionbench.data import iter_items, load_benchmark  # noqa: E402
from interactionbench.prompts import format_question  # noqa: E402

TAG_RE = re.compile(r"</(Silence|Standby|Response)>", re.IGNORECASE)


def load_vc3(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(model_id))
    sys.path.insert(0, str(snap))
    from inference_fast_vc3 import SYSTEM, StreamingSession, VideoChat3StreamEngine  # noqa

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="cuda:0", trust_remote_code=True)
    model.eval()
    return model, snap, SYSTEM, StreamingSession, VideoChat3StreamEngine


def parse_answer(text: str) -> tuple[bool, str | None, bool]:
    """-> (spoke, content, standby)"""
    m = TAG_RE.search(text or "")
    tag = m.group(1).lower() if m else None
    if tag == "response":
        content = TAG_RE.sub("", text).strip()
        return True, (content or None), False
    if tag == "standby":
        return False, None, True
    if tag == "silence" or tag is None:
        return False, None, False
    return False, None, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/interactionbench")
    ap.add_argument("--model", default="MCG-NJU/VideoChat3-4B")
    ap.add_argument("--items", default=None,
                    help="file with one item_id (video_id#idx) per line; "
                         "default: all valid items with local videos")
    ap.add_argument("--mcq", default=None, nargs="?",
                    const="data/interactionbench/mcq/mcq_options_v4.jsonl")
    ap.add_argument("--target-fps", type=float, default=4.0)
    ap.add_argument("--max-pixels", type=int, default=224 * 224)
    ap.add_argument("--max-rounds", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--a-window", type=float, default=10.0,
                    help="seconds of active polling after an A-type reveal")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import torch  # noqa: F401  (fail fast if env broken)

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
    vroot = Path(args.data) / "videos"
    ready = [(it, vroot / it.domain / f"{it.video_id}.mp4") for it in items]
    ready = [(it, p) for it, p in ready if p.exists() and p.stat().st_size > 0]
    if args.limit:
        ready = ready[: args.limit]
    if not ready:
        sys.exit("no runnable items")

    run_tag = "videochat3-4b_streaming_iv1" + ("_mcq" if mcq else "")
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

    print(f"loading {args.model} ...", flush=True)
    model, snap, SYSTEM, StreamingSession, Engine = load_vc3(args.model)
    from transformers import AutoProcessor
    standby_px = args.max_pixels * 4
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, min_pixels=28 * 28,
        max_pixels=standby_px)
    sys.path.insert(0, str(snap))
    from demo_vc3_proactive import _resize_frame  # reuse official resizing

    import torch

    class _Engine:
        def infer(self, messages, max_tokens=args.max_new_tokens, **_):
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=max_tokens,
                                     do_sample=False)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
            text = processor.batch_decode(
                trimmed, skip_special_tokens=False,
                clean_up_tokenization_spaces=False)[0]
            return Engine._strip_end_tokens(text)

    engine = _Engine()
    fpr = max(1, round(args.target_fps))  # frames per 1s round

    frames_cache: dict[str, list] = {}
    print(f"{len(ready)} items | out: {out_dir}", flush=True)
    for i, (it, mp4) in enumerate(ready, 1):
        if it.item_id in done:
            print(f"[{i}/{len(ready)}] {it.item_id} -> skip", flush=True)
            continue
        print(f"[{i}/{len(ready)}] {it.capability}/{it.time_type} {it.item_id} "
              f"({it.duration_s:.0f}s)", flush=True)
        if it.video_id not in frames_cache:
            frames_cache.clear()
            frames_cache[it.video_id] = extract_frames(
                mp4, sample_fps=args.target_fps, max_long_side=None)
        frames = frames_cache[it.video_id]

        is_A = it.time_type == "A"
        q_round = int(it.question_time_s) if is_A else 0
        end_s = min(it.duration_s, it.question_time_s + args.a_window) \
            if is_A else it.duration_s
        n_rounds = max(1, int(end_s))
        # A-type: if the question lands at the very end, still give one round
        q_round = min(q_round, n_rounds - 1)

        session = StreamingSession(
            engine, system=SYSTEM,
            question=format_question(it, mcq.get(it.item_id)),
            question_time=q_round, global_question=not is_A,
            max_rounds=args.max_rounds, max_tokens=args.max_new_tokens)

        polls = []
        standby_remaining = 0
        for r in range(n_rounds):
            fs = [f for f in frames if r <= f.time < r + 1] or frames[-1:]
            high = standby_remaining > 0
            if standby_remaining > 0:
                standby_remaining -= 1
            px = standby_px if high else args.max_pixels
            fimgs = [_resize_frame(f.image, px) for f in fs[:fpr]]
            t0 = time.perf_counter()
            try:
                ans = session.step(fimgs, round_idx=r, frame_max_pixels=px,
                                   time_start=float(r), time_end=float(r + 1))
            except Exception as e:
                polls.append({"t": r + 1.0, "spoke": False, "response": None,
                              "latency_s": 0.0, "n_images": len(fimgs),
                              "raw": f"ERROR: {e}"})
                continue
            lat = time.perf_counter() - t0
            spoke, content, standby = parse_answer(ans)
            if standby:
                standby_remaining = 1
            polls.append({"t": r + 1.0, "spoke": spoke, "response": content,
                          "latency_s": round(lat, 3), "n_images": len(fimgs),
                          "raw": ans})
            if args.verbose and (spoke or r % 30 == 0):
                tag = f"SPEAK: {content}" if spoke else ("standby" if standby else "wait")
                print(f"    t={r + 1:6.1f}s [{lat:5.2f}s] {tag}", flush=True)

        pred = {"video_id": it.video_id, "item_index": it.item_index,
                "model": "videochat3-4b", "run": run_tag,
                "emissions": [{"t": p["t"], "content": p["response"],
                               "latency_s": p["latency_s"]}
                              for p in polls if p["spoke"] and p["response"]],
                "n_polls": len(polls),
                "poll_latencies": [p["latency_s"] for p in polls if p["latency_s"] > 0]}
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
