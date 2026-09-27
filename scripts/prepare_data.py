#!/usr/bin/env python3
"""Prepare the benchmark data directory.

  download   fetch annotations, videos and multiple-choice files from the Hugging Face Hub
  h264       build H.264 proxies for videos in other codecs (AV1, HEVC)
  check      report what is present and what is missing

Expected layout afterwards:

  <data>/annotations/<domain>/<video_id>.json
  <data>/videos/<domain>/<video_id>.mp4
  <data>/videos_h264/<video_id>.mp4          (optional, from `h264`)
  <data>/mcq/mcq_options_v4.jsonl
  <data>/mcq/mcq_key_v4.jsonl
  <data>/items.jsonl                         (all items in one table, not read by the code)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATASET_REPO = "InteractionBench/InteractionBench"
_YT_ID = re.compile(r"\[([A-Za-z0-9_-]{11})\]$")


def cmd_download(args) -> None:
    from huggingface_hub import snapshot_download
    patterns = ["README.md", "items.jsonl", "annotations/**", "mcq/**"]
    if not args.no_videos:
        patterns.append("videos/**")
    snapshot_download(args.repo, repo_type="dataset", local_dir=args.data,
                      allow_patterns=patterns)
    print(f"downloaded {args.repo} -> {args.data}")


def _video_id(path: Path) -> str:
    m = _YT_ID.search(path.stem)
    return m.group(1) if m else path.stem


def cmd_h264(args) -> None:
    from interactionbench import load_benchmark
    data = Path(args.data)
    out = data / "videos_h264"
    out.mkdir(parents=True, exist_ok=True)
    need = {v.video_id for v in load_benchmark(data) if v.items}
    have = {p.stem for p in out.glob("*.mp4")}
    local = {_video_id(p): p for p in (data / "videos").glob("*/*.mp4")}
    todo = sorted(need - have)
    print(f"need {len(need)}, have {len(have)}, to do {len(todo)}", flush=True)
    ok = fail = 0
    for i, vid in enumerate(todo, 1):
        src = local.get(vid)
        if src is None:
            print("missing source video:", vid, flush=True)
            fail += 1
            continue
        codec = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name", "-of", "csv=p=0", str(src)],
            capture_output=True, text=True).stdout.strip()
        dst = out / f"{vid}.mp4"
        if codec == "h264":
            r = subprocess.run(["cp", str(src), str(dst)])
        else:
            r = subprocess.run(["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-preset",
                                "fast", "-crf", "23", "-c:a", "copy", "--", str(dst)],
                               capture_output=True)
        if r.returncode:
            print("failed:", vid, flush=True)
            fail += 1
        else:
            ok += 1
        if i % 20 == 0:
            print(f"{i}/{len(todo)} ok={ok} fail={fail}", flush=True)
    print(f"done ok={ok} fail={fail}")


def cmd_check(args) -> None:
    from interactionbench import iter_items, load_benchmark
    from interactionbench.run import find_video
    data = Path(args.data)
    items = list(iter_items(load_benchmark(data)))
    videos = {it.video_id for it in items}
    found = sum(1 for it in items if find_video(data / "videos", it) is not None)
    h264 = len(list((data / "videos_h264").glob("*.mp4"))) if (data / "videos_h264").is_dir() else 0
    print(f"scoreable items: {len(items)} over {len(videos)} videos")
    print(f"items whose video is present under {data}/videos: {found}")
    print(f"H.264 proxies under {data}/videos_h264: {h264}")
    for name in ("mcq_options_v4.jsonl", "mcq_key_v4.jsonl"):
        fp = data / "mcq" / name
        n = sum(1 for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()) if fp.exists() else None
        print(f"{fp}: {'missing' if n is None else str(n) + ' items'}")
    for split in sorted((REPO / "benchmark/splits").glob("*.txt")):
        ids = {l.strip() for l in split.read_text().splitlines() if l.strip()}
        known = {it.item_id for it in items}
        print(f"split {split.name}: {len(ids)} items, {len(ids - known)} not in the annotations")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/interactionbench")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download")
    p.add_argument("--repo", default=DATASET_REPO, help="dataset repository on the Hub")
    p.add_argument("--no-videos", action="store_true", help="annotations only (enough for scoring)")
    p.set_defaults(func=cmd_download)
    p = sub.add_parser("h264")
    p.set_defaults(func=cmd_h264)
    p = sub.add_parser("check")
    p.set_defaults(func=cmd_check)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
