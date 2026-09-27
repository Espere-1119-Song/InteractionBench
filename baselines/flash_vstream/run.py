"""Flash-VStream-Qwen-7b on InteractionBench via the POLLING protocol.

Flash-VStream is reactive (it has no speak/silence head), so it runs the standard
polling protocol: its flash memory ingests the stream incrementally (8 fps), and
every --interval seconds the model is asked the SPEAK-or-WAIT question; silence is
the instructed default.

Ingest is O(1) per tick (fixed-size flash memory); the query reads the memory.

Upstream: https://github.com/IVGSZ/Flash-VStream (this script imports the ``models``
package of its ``Flash-VStream-Qwen`` sub-directory; pass that directory with --repo
or the environment variable FVSTREAM_REPO).
Checkpoint: zhang9302002/Flash-VStream-Qwen-7b (Hugging Face). By default the newest
snapshot in the Hugging Face cache is used; --checkpoint takes a local directory.

Environment (versions of the paper run): Python 3.10, torch 2.7.1 (CUDA 12.8),
torchvision 0.22.1, transformers 4.45.0, flash-attn 2.8.3, decord 0.6.0,
accelerate 1.14.0, peft 0.20.0; see README.md. flash-attn is required.

Command used for the paper numbers (run directory fvstream-7b_polling_iv1_8fps_mcq,
then ``baselines/flash_vstream/reparse.py`` for
fvstream-7b_polling_iv1_8fps_mcq_lenientparse; the run was split into shards with
--items and merged with ``ibench merge``):
  python baselines/flash_vstream/run.py --mcq --fps 8 --interval 1.0

Exit code 17 means a GPU out-of-memory error: start the same command again (a new
process gets a clean CUDA context). An item that ran out of memory in two processes
is recorded as failed (empty emissions) on the next start.

Output: <out>/preds.jsonl (one line per item, appended, resumable),
<out>/raw/<video_id>#<item_index>.json (the first 120 characters of every reply),
<out>/config.json, <out>/oom_restarts.json and <out>/current_item.txt (the item in
progress). Default <out>: results/runs/fvstream-7b_polling_iv<interval>_<fps>fps[_mcq].
"""
from __future__ import annotations

import argparse
import contextlib
import glob as _glob
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import os

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

SYSTEM = ("You are a real-time visual assistant watching a LIVE video stream "
          "through a streaming memory. You must decide, right now, whether to "
          "speak or keep watching silently. Only speak when this exact moment "
          "calls for it; never repeat what you already reported.")
FORMAT = ("Respond in EXACTLY this format, nothing else:\n"
          "DECISION: SPEAK or WAIT\n"
          "RESPONSE: <if SPEAK, one short line; if WAIT, leave blank>")
DEC_RE = re.compile(r"DECISION:\s*(SPEAK|WAIT)", re.I)
RESP_RE = re.compile(r"RESPONSE:\s*(.*)", re.I | re.S)


def format_question(item, opt):
    if not opt:
        return item.question
    L = "ABCDEF"
    opts = "\n".join(f"{L[i]}. {o}" for i, o in enumerate(opt["options"]))
    stem = opt.get("stem") or item.question
    return (f"{stem}\n{opts}\n"
            f"(When you speak, answer with just the option letter.)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Flash-VStream-Qwen-7b over InteractionBench (polling protocol).")
    ap.add_argument("--data", default="data/interactionbench")
    ap.add_argument("--repo",
                    default=os.environ.get("FVSTREAM_REPO",
                                           "external/Flash-VStream/Flash-VStream-Qwen"),
                    help="the Flash-VStream-Qwen directory of a checkout of "
                         "https://github.com/IVGSZ/Flash-VStream "
                         "(environment variable FVSTREAM_REPO)")
    ap.add_argument("--checkpoint", default=os.environ.get("FVSTREAM_CKPT"),
                    help="local directory of the zhang9302002/Flash-VStream-Qwen-7b "
                         "checkpoint (environment variable FVSTREAM_CKPT); default: the "
                         "newest snapshot under $HF_HOME/hub, with HF_HOME defaulting "
                         "to ~/.cache/huggingface")
    ap.add_argument("--mcq", default=None, nargs="?",
                    const="data/interactionbench/mcq/mcq_options_v4.jsonl")
    ap.add_argument("--fps", type=int, default=8, help="ingest fps")
    ap.add_argument("--interval", type=float, default=1.0, help="poll every N s")
    ap.add_argument("--a-window", type=float, default=10.0)
    ap.add_argument("--video-dir", default="data/interactionbench/videos_h264")
    ap.add_argument("--items", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    FV = str(Path(args.repo).resolve())
    sys.path.insert(0, FV)

    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from PIL import Image  # noqa: E402

    from interactionbench.data import iter_items, load_benchmark  # noqa: E402

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

    run_tag = f"fvstream-7b_polling_iv{args.interval:g}_{args.fps}fps" + \
              ("_mcq" if mcq else "")
    out_dir = Path(args.out) if args.out else Path("results/runs") / run_tag
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    preds_fp = out_dir / "preds.jsonl"
    done = set()
    if preds_fp.exists():
        done = {f"{json.loads(l)['video_id']}#{json.loads(l)['item_index']}"
                for l in preds_fp.read_text().splitlines() if l.strip()}

    (out_dir / "config.json").write_text(json.dumps({
        "system": "Flash-VStream-Qwen-7b", "run": run_tag,
        "protocol": ("POLLING (non-native system rule): flash memory ingests "
                     f"{args.fps} fps continuously; SPEAK/WAIT prompt every "
                     f"{args.interval}s; silence instructed by default"),
        "fps": args.fps, "interval_s": args.interval,
        "a_window_s": args.a_window, "attn": "flash_attention_2",
        "video_dir": str(vroot), "mcq_options": args.mcq,
        "n_items_targeted": len(ready),
        "item_set_sha1": hashlib.sha1(json.dumps(
            sorted(it.item_id for it, _ in ready)).encode()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }, indent=1))

    print("loading Flash-VStream ...", flush=True)
    from models import (FlashVStreamQwen2VLConfig,  # noqa: E402
                        FlashVStreamQwen2VLProcessor)
    from models.vstream_qwen2vl_realtime import FlashVStreamQwen2VLModel  # noqa: E402
    from models.flash_memory_constants import DEFAULT_FLASH_MEMORY_CONFIG  # noqa: E402
    if args.checkpoint:
        SNAP = args.checkpoint
    else:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        SNAP = sorted(_glob.glob(
            f"{hf_home}/hub/models--zhang9302002--Flash-VStream-Qwen-7b/snapshots/*"))[-1]
    cfg = FlashVStreamQwen2VLConfig.from_pretrained(SNAP, trust_remote_code=True)
    if getattr(cfg.vision_config, "flash_memory_config", None) is None:
        cfg.vision_config.flash_memory_config = DEFAULT_FLASH_MEMORY_CONFIG
    fmc = dict(cfg.vision_config.flash_memory_config)
    for k, v in DEFAULT_FLASH_MEMORY_CONFIG.items():
        fmc.setdefault(k, v)
    model = FlashVStreamQwen2VLModel.from_pretrained(
        SNAP, config=cfg, device_map="cuda", trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").eval()
    processor = FlashVStreamQwen2VLProcessor.from_pretrained(SNAP)
    from decord import VideoReader  # noqa: E402 (after CUDA init)

    def ingest(clip_frames, start_idx):
        vi = processor.image_processor(
            images=None, videos=clip_frames, return_tensors="pt",
            additional_pool_size=fmc["flash_memory_temporal_poolsize"])
        with torch.inference_mode():
            model.embed_new_video_clip(**vi.to("cuda"), start_idx=start_idx)
        return start_idx + len(clip_frames)

    def memory_tokens():
        # exact formula from the model's own position-id assert:
        # spa_size + tem_size where size = thw.prod()//4
        # (memory slots: 1 = tem_thw, 5 = spa_thw)
        mem = model.video_embedding_memory
        if not mem:
            return 0
        tem = int(mem[1].prod().item()) // 4
        spa = int(mem[5].prod().item()) // 4
        return spa + tem

    def ask(question, t):
        n_tok = memory_tokens()
        if n_tok <= 0:
            return ""
        text = processor.apply_chat_template([{"role": "user", "content": [
            {"type": "text", "text": "<|vision_start|><|video_pad|><|vision_end|>"
             + f"{SYSTEM}\n\nCurrent stream time: {t:.1f}s.\n\n"
               f"User's request: \"{question}\"\n\n{FORMAT}"}]}],
            tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=None, videos=None, padding=True,
                           return_tensors="pt", flash_memory_config=fmc,
                           dummy_video_tokens=n_tok * 4).to("cuda")
        with torch.inference_mode():
            out = model.generate(**inputs,
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=False, use_cache=True)
        text = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0]
        del inputs, out
        torch.cuda.empty_cache()  # per-tick allocations fragment the memory of a 46 GB GPU
        return text

    print(f"{len(ready)} items | out: {out_dir}", flush=True)
    oom_fp = out_dir / "oom_restarts.json"
    oom_counts = json.loads(oom_fp.read_text()) if oom_fp.exists() else {}
    marker = out_dir / "current_item.txt"
    if marker.exists():
        # the previous process died in the middle of an item (GPU stall inside a
        # call, killed by an external watchdog, etc.)
        dead = marker.read_text().strip()
        if dead:
            oom_counts[dead] = oom_counts.get(dead, 0) + 1
            oom_fp.write_text(json.dumps(oom_counts))
            print(f"  {dead}: uncleanly died last run (count {oom_counts[dead]})", flush=True)
        marker.unlink()
    for i, (it, mp4) in enumerate(ready, 1):
        if it.item_id in done:
            continue
        n_oom = oom_counts.get(it.item_id, 0)
        if n_oom >= 2:
            # Out of memory from a clean context in >=2 processes: the item exceeds
            # a full 180 GB GPU for this system. Record it as a failed item (empty
            # emissions, error noted) so that the run can terminate; it is scored as
            # silence/miss.
            pred = {"video_id": it.video_id, "item_index": it.item_index,
                    "model": "fvstream-7b", "run": run_tag, "emissions": [],
                    "n_polls": 0, "poll_latencies": [],
                    "error": f"oom_x{n_oom}_on_180GB"}
            with preds_fp.open("a") as f:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            print(f"  {it.item_id}: {n_oom}x clean-context OOM -> recorded as failed", flush=True)
            continue
        print(f"[{i}/{len(ready)}] {it.capability}/{it.time_type} {it.item_id} "
              f"({it.duration_s:.0f}s)", flush=True)
        marker.write_text(it.item_id)
        question = format_question(it, mcq.get(it.item_id))
        is_A = it.time_type == "A"
        q_t = it.question_time_s or 0.0
        emissions, polls = [], []
        try:
            vr = VideoReader(str(mp4))
            vfps = vr.get_avg_fps()
            dur = len(vr) / vfps
            end_s = min(dur, q_t + args.a_window) if is_A else dur
            # fresh memory per item
            model.use_video_streaming_mode = True
            model.video_embedding_memory = []
            frame_cnt = 0
            t = 0.0
            t_item0 = time.time()
            while t < end_s - 1e-6:
                if time.time() - t_item0 > 5400:
                    # GPU memory nearly full: allocations barely succeed, GPU
                    # utilisation is about 8%, and the log keeps printing, so a
                    # watchdog that looks at the log does not see the stall
                    raise RuntimeError("out of memory (item wall-clock cap 90min — VRAM thrash)")
                t_next = min(t + args.interval, end_s)
                idxs = [min(int(x * vfps), len(vr) - 1)
                        for x in np.arange(t, t_next, 1.0 / args.fps)]
                if len(idxs) % 2:  # odd frame count breaks temporal patching
                    idxs.append(idxs[-1])  # (memory grid then desyncs -> reshape errors downstream)
                if idxs:
                    clip = [Image.fromarray(vr[k].asnumpy()) for k in idxs]
                    frame_cnt = ingest(clip, frame_cnt)
                t = t_next
                if is_A and t < q_t - 1e-6:
                    # B/C items stay memory-bounded because per-tick queries drive
                    # the flash-memory consolidation; pure pre-reveal ingest grows
                    # to ~177G and thrashes. Run a neutral consolidation query
                    # every 30s of stream time: the question text never appears
                    # before the reveal, output discarded, not recorded as a poll.
                    if int(t) % 30 == 0:
                        with contextlib.suppress(Exception):
                            ask("Keep watching. Reply WAIT.", t)
                    continue  # A: no polling before the reveal
                t0 = time.perf_counter()
                raw = ask(question, t)
                lat = time.perf_counter() - t0
                m = DEC_RE.search(raw or "")
                spoke = bool(m and m.group(1).upper() == "SPEAK")
                resp = ""
                if spoke:
                    rm = RESP_RE.search(raw or "")
                    resp = (rm.group(1).strip().splitlines()[0].strip()
                            if rm and rm.group(1).strip() else "")
                polls.append({"t": round(t, 2), "raw": (raw or "")[:120],
                              "latency_s": round(lat, 3), "spoke": spoke})
                if spoke and resp:
                    emissions.append({"t": round(t, 2), "content": resp,
                                      "latency_s": round(lat, 3)})
        except Exception as e:
            marker.unlink(missing_ok=True)
            msg = str(e)
            print(f"  ERROR {it.item_id}: {msg[:200]}", flush=True)
            model.video_embedding_memory = []
            torch.cuda.empty_cache()
            if "out of memory" in msg:
                # Out of memory leaves ~100G of unreclaimable resident tensors
                # (the same failure then repeats on every later item), so exit and
                # let the caller restart with a clean CUDA context. The side file
                # caps the retries per item so that oversized items cannot loop.
                fp = out_dir / "oom_restarts.json"
                d = json.loads(fp.read_text()) if fp.exists() else {}
                d[it.item_id] = d.get(it.item_id, 0) + 1
                fp.write_text(json.dumps(d))
                print(f"  OOM -> clean-restart (count {d[it.item_id]})", flush=True)
                sys.exit(17)
            continue
        model.video_embedding_memory = []
        torch.cuda.empty_cache()

        pred = {"video_id": it.video_id, "item_index": it.item_index,
                "model": "fvstream-7b", "run": run_tag,
                "emissions": emissions, "n_polls": len(polls),
                "poll_latencies": [p["latency_s"] for p in polls]}
        marker.unlink(missing_ok=True)
        with preds_fp.open("a") as f:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        (raw_dir / f"{it.video_id}#{it.item_index}.json").write_text(
            json.dumps({"item_id": it.item_id, "question": question,
                        "capability": it.capability, "time_type": it.time_type,
                        "polls": polls}, indent=1, ensure_ascii=False))
        if emissions:
            print(f"    {len(emissions)} emissions, first t={emissions[0]['t']}",
                  flush=True)

    print(f"\nDONE -> {preds_fp}", flush=True)


if __name__ == "__main__":
    main()
