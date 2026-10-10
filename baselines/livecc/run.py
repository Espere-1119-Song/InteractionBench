"""Run LiveCC-7B-Instruct (showlab, CVPR25) over InteractionBench."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from interactionbench.data import iter_items, load_benchmark  # noqa: E402
from interactionbench.prompts import format_question  # noqa: E402

MODEL = "chenjoya/LiveCC-7B-Instruct"
NEUTRAL = "Please describe what is happening."


_SENT_END = (".", "!", "?", "。", "!", "?")


def group_sentences(fragments: list[dict]) -> list[dict]:
    out, buf, t0, lat = [], [], None, 0.0
    for f in fragments:
        txt = (f["content"] or "").strip()
        while txt.endswith("..."):
            txt = txt[:-3].strip()
        if not txt:
            continue
        if t0 is None:
            t0 = f["t"]
        buf.append(txt)
        lat += f.get("latency_s", 0.0)
        if txt.endswith(_SENT_END):
            out.append({"t": t0, "content": " ".join(buf), "latency_s": round(lat, 3)})
            buf, t0, lat = [], None, 0.0
    if buf:
        out.append({"t": t0, "content": " ".join(buf), "latency_s": round(lat, 3)})
    return out


def _build_infer(cls, model_path: str):
    import functools

    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from livecc_utils import prepare_multiturn_multimodal_inputs_for_generation

    self = cls.__new__(cls)
    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    self.streaming_eos_token_id = self.processor.tokenizer(" ...").input_ids[-1]
    self.model.prepare_inputs_for_generation = functools.partial(
        prepare_multiturn_multimodal_inputs_for_generation, self.model)
    texts = self.processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "livecc"}]}],
        tokenize=False)
    self.system_prompt_offset = texts.index("<|im_start|>user")
    self._cached_video_readers_with_hw = {}
    return self


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/interactionbench")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--livecc-repo",
                    default=os.environ.get("LIVECC_REPO", "external/livecc"),
                    help="checkout of https://github.com/showlab/livecc "
                         "(environment variable LIVECC_REPO)")
    ap.add_argument("--items", default=None)
    ap.add_argument("--mcq", default=None, nargs="?",
                    const="data/interactionbench/mcq/mcq_options_v4.jsonl")
    ap.add_argument("--a-window", type=float, default=10.0)
    ap.add_argument("--video-dir", default=None,
                    help="override video root (LiveCC's decord cannot read AV1; "
                         "use data/interactionbench/videos_h264 proxies)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    livecc_repo = Path(args.livecc_repo).resolve()
    if not livecc_repo.is_dir():
        sys.exit(f"LiveCC checkout not found: {livecc_repo} "
                 "(set --livecc-repo or LIVECC_REPO)")
    sys.path.insert(0, str(livecc_repo))
    sys.path.insert(0, str(livecc_repo / "demo"))
    sys.path.insert(0, str(livecc_repo / "livecc-utils" / "src"))

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
    vroot = Path(args.video_dir) if args.video_dir else Path(args.data) / "videos"
    flat = args.video_dir is not None
    ready = [(it, (vroot / f"{it.video_id}.mp4") if flat
              else (vroot / it.domain / f"{it.video_id}.mp4")) for it in items]
    ready = [(it, p) for it, p in ready if p.exists() and p.stat().st_size > 0]
    if args.limit:
        ready = ready[: args.limit]
    if not ready:
        sys.exit("no runnable items")

    run_tag = "livecc-7b_streaming_iv1" + ("_mcq" if mcq else "")
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
    from infer import LiveCCDemoInfer
    infer = _build_infer(LiveCCDemoInfer, args.model)
    print(f"{len(ready)} items | out: {out_dir}", flush=True)

    for i, (it, mp4) in enumerate(ready, 1):
        if it.item_id in done:
            print(f"[{i}/{len(ready)}] {it.item_id} -> skip", flush=True)
            continue
        print(f"[{i}/{len(ready)}] {it.capability}/{it.time_type} {it.item_id} "
              f"({it.duration_s:.0f}s)", flush=True)
        question = format_question(it, mcq.get(it.item_id))
        is_A = it.time_type == "A"
        end_s = min(it.duration_s, it.question_time_s + args.a_window) if is_A \
            else it.duration_s

        state = {"video_path": str(mp4)}
        emissions, polls = [], []
        t = 1.0
        try:
            while t <= end_s + 1e-6:
                msg = question if (not is_A or t >= it.question_time_s - 1e-6) else NEUTRAL
                state["video_timestamp"] = t
                t0 = time.perf_counter()
                for (t_start, t_stop), resp, state in infer.live_cc(
                        message=msg, state=state, do_sample=False,
                        repetition_penalty=1.05):
                    lat = time.perf_counter() - t0
                    txt = (resp or "").strip()
                    polls.append({"t": round(t_stop, 2), "response": txt,
                                  "latency_s": round(lat, 3),
                                  "revealed": msg is question})
                    if txt and (not is_A or t_stop >= it.question_time_s - 1e-6):
                        emissions.append({"t": round(t_stop, 2), "content": txt,
                                          "latency_s": round(lat, 3)})
                    t0 = time.perf_counter()
                if state.get("video_end"):
                    break
                t += 1.0
        except Exception as e:
            print(f"  ERROR {it.item_id}: {str(e)[:200]}", flush=True)
        infer._cached_video_readers_with_hw.pop(str(mp4), None)

        emissions = group_sentences(emissions)
        if args.verbose:
            for e in emissions[:5]:
                print(f"    t={e['t']:6.1f}s {e['content'][:80]}", flush=True)
        pred = {"video_id": it.video_id, "item_index": it.item_index,
                "model": "livecc-7b", "run": run_tag,
                "emissions": emissions, "n_polls": len(polls),
                "poll_latencies": [p["latency_s"] for p in polls]}
        with preds_fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        (raw_dir / f"{it.video_id}#{it.item_index}.json").write_text(
            json.dumps({"item_id": it.item_id, "question": question,
                        "capability": it.capability, "time_type": it.time_type,
                        "polls": polls}, indent=1, ensure_ascii=False),
            encoding="utf-8")

    print(f"\nDONE -> {preds_fp}", flush=True)


if __name__ == "__main__":
    main()
