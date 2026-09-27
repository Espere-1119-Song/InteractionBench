"""Run ViSpeak-s3 (fushh7/ViSpeak-s3, VITA-1.5 base) over InteractionBench.

ViSpeak is a proactive streaming model: an informative_head scores each video
segment and the model starts speaking when a 3-segment sliding window of
sigmoid scores exceeds 0.35. `streaming_generate` returns ONE response and the
video-time of the triggering segment, so multi-event items are handled by
re-invoking from the segment after each trigger (responses are not fed back —
each continuation is independent; recorded in config).

System-fixed sampling: `_get_rawvideo_dec` caps frames at
MAX_IMAGE_LENGTH * pooling^2 (s3: 16*2*2 = 64) uniformly over the video, so
input "fps" is duration-dependent and NOT freely settable (recorded in
config.json of the run).

Item protocol:
  B/C  question appended to the system prompt (standing request from t=0),
       inference scans from the first segment
  A    question in system prompt, but inference may only start at the first
       segment >= question_time_s (answer-when-asked); scan to q_t + a-window

Upstream:
  code        https://github.com/HumanMLLM/ViSpeak
              --vispeak-repo (or the environment variable VISPEAK_REPO), default
              external/ViSpeak. The runner imports the `vispeak` package from
              this checkout.
  checkpoint  fushh7/ViSpeak-s3 (https://huggingface.co/fushh7/ViSpeak-s3)
              --model (or the environment variable VISPEAK_MODEL). The upstream
              README requires a local copy whose config.json points to the
              audio encoder (VITA-MLLM/VITA-1.5) and the visual encoder
              (OpenGVLab/InternViT-300M-448px); pass that directory.

Environment:
  A dedicated environment: python 3.10, torch 2.4.0 (cu121), torchaudio 2.4,
  transformers 4.44.2, xformers 0.0.27.post2, six, decord. decord cannot read
  AV1 video, so --video-dir defaults to the H.264 proxy directory. See
  README.md in this directory.

Command used for the paper numbers (1,060-item set, v4 multiple-choice file):
  python baselines/vispeak/run.py --model /path/to/ViSpeak-s3 --mcq

Output:
  results/runs/vispeak-s3_streaming_native[_mcq]/preds.jsonl   (or under --out)
  results/runs/vispeak-s3_streaming_native[_mcq]/raw/<item_id>.json
  results/runs/vispeak-s3_streaming_native[_mcq]/config.json
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from interactionbench.data import iter_items, load_benchmark  # noqa: E402
from interactionbench.prompts import format_question  # noqa: E402

MODEL_PATH = "fushh7/ViSpeak-s3"
MAX_TRIGGERS = 24


def zero_audio_dict():
    import torch

    audio = torch.zeros(400, 80).unsqueeze(0)
    return {"audios": audio.half().cuda(),
            "lengths": torch.unsqueeze(torch.tensor(400), dim=0).half().cuda(),
            "lengths_for_llm": torch.unsqueeze(torch.tensor(60), dim=0).cuda()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/interactionbench")
    ap.add_argument("--model", default=os.environ.get("VISPEAK_MODEL", MODEL_PATH),
                    help="checkpoint directory or repository id "
                         "(environment variable VISPEAK_MODEL)")
    ap.add_argument("--vispeak-repo",
                    default=os.environ.get("VISPEAK_REPO", "external/ViSpeak"),
                    help="checkout of https://github.com/HumanMLLM/ViSpeak "
                         "(environment variable VISPEAK_REPO)")
    ap.add_argument("--mcq", default=None, nargs="?",
                    const="data/interactionbench/mcq/mcq_options_v4.jsonl")
    ap.add_argument("--a-window", type=float, default=10.0)
    ap.add_argument("--video-dir", default="data/interactionbench/videos_h264")
    ap.add_argument("--items", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    vispeak_repo = Path(args.vispeak_repo).resolve()
    if not vispeak_repo.is_dir():
        sys.exit(f"ViSpeak checkout not found: {vispeak_repo} "
                 "(set --vispeak-repo or VISPEAK_REPO)")
    sys.path.insert(0, str(vispeak_repo))

    import torch
    from vispeak.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN_NUMBER,
                                   DEFAULT_SEG_TOKEN, MAX_IMAGE_LENGTH)
    from vispeak.model.builder import load_pretrained_model
    from vispeak.util.data_utils import SYSTEM_PROMTP, _get_rawvideo_dec
    from vispeak.util.mm_utils import (KeywordsStoppingCriteria,
                                       get_model_name_from_path,
                                       tokenizer_image_audio_token)

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

    run_tag = "vispeak-s3_streaming_native" + ("_mcq" if mcq else "")
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
        "system": "ViSpeak-s3", "model": "fushh7/ViSpeak-s3 (VITA-1.5 base)",
        "run": run_tag,
        "protocol": ("proactive informative_head (3-seg window > 0.35); "
                     "multi-event = re-invoke from segment after each trigger, "
                     "responses NOT fed back"),
        "sampling": ("system-fixed: <=64 segments uniform over video "
                     "(MAX_IMAGE_LENGTH*pooling^2); input fps not settable"),
        "a_window_s": args.a_window, "max_triggers": MAX_TRIGGERS,
        "video_dir": str(vroot), "mcq_options": args.mcq,
        "n_items_targeted": len(ready),
        "item_set_sha1": hashlib.sha1(json.dumps(
            sorted(it.item_id for it, _ in ready)).encode()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }, indent=1))

    print(f"loading {args.model} ...", flush=True)
    model_name = get_model_name_from_path(args.model)
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model, None, model_name, "qwen2p5_instruct")
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor
    audio_encoder = model.get_audio_encoder()
    audio_encoder.to(dtype=torch.float16)
    model.eval()
    pooling_size = getattr(model.config, "pooling_size", 1)
    img_token_num = DEFAULT_IMAGE_TOKEN_NUMBER // pooling_size // pooling_size
    seg_id = tokenizer.convert_tokens_to_ids(DEFAULT_SEG_TOKEN)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    audios = zero_audio_dict()
    video_audios = zero_audio_dict()

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
            patch_images, _, _, _, sample_time = _get_rawvideo_dec(
                str(mp4), image_processor,
                max_frames=MAX_IMAGE_LENGTH * pooling_size * pooling_size,
                video_framerate=1,
                image_aspect_ratio=model.config.image_aspect_ratio)
            n_patch = len(patch_images)
            patch_images = torch.stack(patch_images).half().cuda()
            # wide/tall videos slice each frame into k patches; the model wants
            # one IMAGE_TOKEN per PATCH, timestamps repeated per slice
            if n_patch != len(sample_time):
                if n_patch % len(sample_time) == 0:
                    k = n_patch // len(sample_time)
                    sample_time = [t for t in sample_time for _ in range(k)]
                else:
                    print(f"  SKIP {it.item_id}: patches {n_patch} !% "
                          f"frames {len(sample_time)}", flush=True)
                    continue
            n_seg = len(sample_time)

            sys_text = (SYSTEM_PROMTP["vispeak"]
                        + "\nUser request (answer at the right moment): "
                        + question)
            ids_sys = tokenizer_image_audio_token(
                sys_text + DEFAULT_SEG_TOKEN, tokenizer, return_tensors="pt")
            parts, num_token = [ids_sys], []
            acc = len(ids_sys)
            for _ in sample_time:
                ids = tokenizer_image_audio_token(
                    DEFAULT_IMAGE_TOKEN + DEFAULT_SEG_TOKEN, tokenizer,
                    return_tensors="pt", image_token_number=img_token_num)
                parts.append(ids)
                acc += len(ids)
                num_token.append(acc)
            user_ids = torch.cat(parts).unsqueeze(0).cuda()
            agent_ids = torch.full_like(user_ids, tokenizer.pad_token_id)
            stop = KeywordsStoppingCriteria(["<|im_end|>"], tokenizer, user_ids)

            # first segment inference may start from
            start_seg = 0
            if is_A:
                start_seg = next((k for k, t in enumerate(sample_time)
                                  if t >= q_t - 1e-6), n_seg - 1)
            end_seg = n_seg - 1
            if is_A:
                t_end = q_t + args.a_window
                end_seg = next((k for k in range(n_seg - 1, -1, -1)
                                if sample_time[k] <= t_end + 1e-6), n_seg - 1)

            seg = start_seg
            while seg <= end_seg and len(emissions) < MAX_TRIGGERS:
                t0 = time.perf_counter()
                with torch.inference_mode():
                    # streaming_generate MUTATES user/agent ids in place
                    # (image placeholders get overwritten) — always pass clones
                    cont, resp_t = model.streaming_generate(
                        user_ids.clone(), agent_input_ids=agent_ids.clone(),
                        start_inference_seg=(seg, num_token[seg]),
                        timestamps=list(sample_time),
                        seg_token_id=seg_id,
                        images=patch_images, video_audios=video_audios,
                        audios=audios, pad_token_id=tokenizer.pad_token_id,
                        temperature=0.01, max_new_tokens=256, padding_size=128,
                        stopping_criteria=stop,
                        # ViSpeak's own bench: proactive head for standing
                        # tasks, forced answer (proactive=False) for reactive
                        # QA — mirrors their Visual_Reference protocol
                        proactive=not is_A,
                        sentence_end_token_id=end_id)
                lat = time.perf_counter() - t0
                text = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
                if text and text[0] in "☞☜☟":
                    text = text[1:]
                text = (text or "").strip()
                polls.append({"t": float(resp_t) if resp_t is not None else None,
                              "latency_s": round(lat, 3),
                              "text": text[:120]})
                if resp_t is None or not text:
                    break
                if float(resp_t) > (sample_time[end_seg] + 1e-6):
                    break  # trigger past the allowed window
                emissions.append({"t": round(float(resp_t), 2),
                                  "content": text, "latency_s": round(lat, 3)})
                nxt = next((k for k, t in enumerate(sample_time)
                            if t > float(resp_t) + 1e-6), None)
                if nxt is None or nxt <= seg:
                    break
                seg = nxt
        except Exception as e:
            import traceback
            print(f"  ERROR {it.item_id}: {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
            traceback.print_exc()
            continue

        pred = {"video_id": it.video_id, "item_index": it.item_index,
                "model": "vispeak-s3", "run": run_tag,
                "emissions": emissions, "n_polls": len(polls),
                "poll_latencies": [p["latency_s"] for p in polls]}
        with preds_fp.open("a") as f:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        (raw_dir / f"{it.video_id}#{it.item_index}.json").write_text(
            json.dumps({"item_id": it.item_id, "question": question,
                        "capability": it.capability, "time_type": it.time_type,
                        "n_segments": n_seg, "polls": polls},
                       indent=1, ensure_ascii=False))
        if emissions:
            print(f"    {len(emissions)} emissions, first t={emissions[0]['t']}",
                  flush=True)

    print(f"\nDONE -> {preds_fp}", flush=True)


if __name__ == "__main__":
    main()
