"""Generate predictions: drive one system over the benchmark with one protocol.

Output directory layout:
  <out>/preds.jsonl      one line per item, directly consumable by the evaluator
  <out>/raw/<item>.json  every decision step including the silent ones
  <out>/config.json      the full configuration of the run

preds.jsonl schema:
  {"video_id", "item_index", "model", "run",
   "emissions": [{"t", "content", "latency_s"}],
   "n_polls", "poll_latencies", "failed"?}

A run is resumable: items already present in preds.jsonl are skipped.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .data import BenchItem, iter_items, load_benchmark
from .frames import extract_frames
from .mcq import load_mcq_options
from .prompts import format_question
from .protocols import ProtocolConfig, get_protocol


@dataclass
class RunConfig:
    data: str = "data/interactionbench"
    model: str = "qwen3vl-8b"
    model_path: str | None = None
    model_args: dict = field(default_factory=dict)
    protocol: str = "sliding"
    protocol_config: ProtocolConfig = field(default_factory=ProtocolConfig)
    sample_fps: float = 2.0
    max_long_side: int = 512
    mcq: str | None = None              # mcq_options.jsonl; eligible items become multiple choice
    video_dir: str | None = None        # default <data>/videos
    items: str | None = None            # file with one item_id per line
    videos: list[str] | None = None
    capabilities: list[str] | None = None
    limit: int = 0
    num_shards: int = 1
    shard_index: int = 0
    out: str | None = None
    overwrite: bool = False
    skip_oom: bool = True               # record a GPU out-of-memory item as silent and continue


def is_gpu_oom(exc: BaseException) -> bool:
    """CUDA out-of-memory, including its cuDNN-attention symptom. Anything else is a
    real error and must propagate."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    msg = str(exc)
    return isinstance(exc, RuntimeError) and ("out of memory" in msg or "mha_graph" in msg)


def find_video(video_root: Path, item: BenchItem) -> Path | None:
    for cand in (video_root / item.domain / f"{item.video_id}.mp4",
                 video_root / f"{item.video_id}.mp4"):
        if cand.exists() and cand.stat().st_size > 0:
            return cand
    return None


def select_items(cfg: RunConfig) -> list[BenchItem]:
    items = list(iter_items(load_benchmark(cfg.data)))
    if cfg.videos:
        keep = set(cfg.videos)
        items = [it for it in items if it.video_id in keep]
    if cfg.items:
        keep = {l.strip() for l in Path(cfg.items).read_text().splitlines() if l.strip()}
        items = [it for it in items if it.item_id in keep]
    if cfg.capabilities:
        keep = set(cfg.capabilities)
        items = [it for it in items if it.capability in keep]
    return items


def done_item_ids(preds_fp: Path) -> set[str]:
    done = set()
    if preds_fp.exists():
        for line in preds_fp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = json.loads(line)
                done.add(f"{p['video_id']}#{p['item_index']}")
    return done


def to_prediction(item: BenchItem, polls: list[dict], model: str, run_tag: str,
                  failed: str | None = None) -> dict:
    pred = {
        "video_id": item.video_id, "item_index": item.item_index,
        "model": model, "run": run_tag,
        "emissions": [{"t": p["t"], "content": p["response"], "latency_s": p["latency_s"]}
                      for p in polls if p["spoke"] and p["response"]],
        "n_polls": len(polls),
        "poll_latencies": [p["latency_s"] for p in polls],
    }
    if failed:
        pred["failed"] = failed
    return pred


def run_benchmark(cfg: RunConfig, model=None) -> Path:
    """Run and return the path of preds.jsonl. Pass ``model`` to reuse a loaded one."""
    from .models import build_model

    protocol = get_protocol(cfg.protocol)
    pcfg = cfg.protocol_config

    mcq: dict[str, dict] = {}
    if cfg.mcq:
        mcq = load_mcq_options(cfg.mcq)
        print(f"multiple choice: {len(mcq)} items have options", file=sys.stderr)

    items = select_items(cfg)
    video_root = Path(cfg.video_dir) if cfg.video_dir else Path(cfg.data) / "videos"
    ready = []
    for it in items:
        mp4 = find_video(video_root, it)
        if mp4 is not None:
            ready.append((it, mp4))
    missing = len(items) - len(ready)
    if missing:
        print(f"note: {missing} items skipped (video not found under {video_root})",
              file=sys.stderr)
    if cfg.num_shards > 1:
        ready = [x for i, x in enumerate(ready) if i % cfg.num_shards == cfg.shard_index]
    if cfg.limit:
        ready = ready[: cfg.limit]
    if not ready:
        sys.exit("no runnable items")

    model_tag = cfg.model.replace("/", "_").replace(":", "_")
    run_tag = (f"{model_tag}_{protocol.run_tag(pcfg)}"
               + ("_mcq" if mcq else "") + ("_blind" if pcfg.blind else ""))
    out_dir = Path(cfg.out) if cfg.out else Path("results/runs") / run_tag
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    preds_fp = out_dir / "preds.jsonl"
    if preds_fp.exists() and cfg.overwrite:
        preds_fp.unlink()
    done = done_item_ids(preds_fp)
    (out_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    if model is None:
        print(f"building model '{cfg.model}' ...", flush=True)
        model = build_model(cfg.model, model_path=cfg.model_path, **cfg.model_args)
    print(f"model ready: {model.name} | protocol: {protocol.name} | "
          f"{len(ready)} items | out: {out_dir}", flush=True)

    needs_frames = protocol.needs_frames and not pcfg.blind
    frames_cache: dict[str, list] = {}
    for i, (it, mp4) in enumerate(ready, 1):
        if it.item_id in done:
            print(f"[{i}/{len(ready)}] {it.item_id} -> skip (done)", flush=True)
            continue
        print(f"[{i}/{len(ready)}] {it.capability}/{it.time_type} {it.item_id} "
              f"({it.duration_s:.0f}s)", flush=True)
        if it.video_id not in frames_cache:
            frames_cache.clear()  # one video at a time; items are grouped by video
            frames_cache[it.video_id] = extract_frames(
                mp4, sample_fps=cfg.sample_fps,
                max_long_side=cfg.max_long_side) if needs_frames else []
        if pcfg.verbose:
            print(f"  Q: {it.question[:90]}", flush=True)
        question = format_question(it, mcq.get(it.item_id))
        failed = None
        try:
            polls = protocol.run_item(model, frames_cache[it.video_id], it, question, pcfg)
        except Exception as exc:
            if not (cfg.skip_oom and is_gpu_oom(exc)):
                raise
            # An item that does not fit the GPU is recorded as silent (no emissions)
            # with a "failed" note, so one item cannot block the whole run.
            import torch
            torch.cuda.empty_cache()
            failed = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
            print(f"  !! skipped (GPU memory): {failed}", flush=True)
            polls = []
        pred = to_prediction(it, polls, cfg.model, run_tag, failed)
        with preds_fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        (raw_dir / f"{it.video_id}#{it.item_index}.json").write_text(
            json.dumps({"item_id": it.item_id, "question": it.question,
                        "capability": it.capability, "time_type": it.time_type,
                        "polls": polls, "failed": failed}, indent=1, ensure_ascii=False),
            encoding="utf-8")

    print(f"\nDONE -> {preds_fp}\nscore with: ibench eval {preds_fp} --data {cfg.data}",
          flush=True)
    return preds_fp


def merge_predictions(inputs: list[str], output: str) -> int:
    """Concatenate prediction files, keeping the first line seen for each item."""
    seen, rows = set(), []
    for fp in inputs:
        for line in Path(fp).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            k = f"{p['video_id']}#{p.get('item_index', 0)}"
            if k in seen:
                continue
            seen.add(k)
            rows.append(line)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)
