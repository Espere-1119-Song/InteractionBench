"""Run OneStreamer-4B natively over InteractionBench; inference and scoring are separate.

Copyright 2026 OneStreamer contributors. Licensed under Apache-2.0; see LICENSE.
"""

from __future__ import annotations

import argparse
import bisect
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from baselines.onestreamer.inference import (  # noqa: E402
    MODEL_ID, MODEL_REVISION, NativeSession, OneStreamerEngine, decision_ticks, parse_action,
)
from interactionbench.data import iter_items, load_benchmark  # noqa: E402
from interactionbench.prompts import format_question  # noqa: E402
from interactionbench.run import find_video  # noqa: E402

SCHEME = {
    "schema": 1, "fps": 4, "first_time_s": 0.25,
    "filter": "setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=0.25,"
              "fps=4:start_time=0.25:round=up:eof_action=pass",
    "eof_policy": "pad the last observed source frame; stop at floor(source_duration*4)/4",
    "timestamp_rule": "tick/4; tick starts at 1; select only frames available by tick/4",
    "filename": "SSSSS_CC.jpeg", "jpeg_quality": 2, "native_resolution": True,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    write_text(path, canonical(value) + "\n")


@contextmanager
def lock_file(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_ids(path):
    rows = [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
    if len(rows) != len(set(rows)):
        raise ValueError("Duplicate item IDs in selection")
    return set(rows)


def select_samples(args):
    """The model receives only the question and causal observations, never references."""
    items = list(iter_items(load_benchmark(args.data)))
    by_id = {it.item_id: it for it in items}
    if len(by_id) != len(items):
        raise ValueError("Duplicate annotation item IDs")
    expected = read_ids(args.items or REPO / "benchmark/splits/all1060.txt")
    if expected - by_id.keys():
        raise ValueError(f"Missing annotations for {len(expected - by_id.keys())} requested items")
    items = [it for it in items if it.item_id in expected]
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise ValueError("No selected items")
    options, options_hash = {}, None
    if args.mcq:
        path = Path(args.data) / "mcq/mcq_options_v4.jsonl" if args.mcq == "auto" else Path(args.mcq)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["item_id"] in options:
                raise ValueError("Duplicate MCQ options")
            opts = row.get("options")
            if not isinstance(opts, list) or not 2 <= len(opts) <= 6 or not all(
                    isinstance(s, str) and s.strip() for s in opts):
                raise ValueError("Invalid MCQ options")
            options[row["item_id"]] = row
        required = read_ids(REPO / "benchmark/splits/mcq688.txt") & {it.item_id for it in items}
        if required - options.keys():
            raise ValueError("Missing v4 MCQ options for selected items")
        options_hash = file_hash(path)
    samples, sources = [], {}
    for it in items:
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in (it.domain, it.video_id)):
            raise ValueError("Unsafe video or domain identifier")
        question = format_question(it, options.get(it.item_id))
        if "<image>" in question:
            raise ValueError("Question contains a reserved image placeholder")
        sample = dict(item_id=it.item_id, video_id=it.video_id, item_index=it.item_index,
                      domain=it.domain, question=question, time_type=it.time_type,
                      question_time_s=it.question_time_s, duration_s=it.duration_s)
        decision_ticks(sample, args.a_window)
        samples.append(sample)
        key = f"{it.domain}/{it.video_id}"
        source = find_video(Path(args.data) / "videos", it)
        if source is None:
            raise ValueError(f"Missing source video for {it.item_id}")
        sources[key] = source
    return samples, sources, options_hash


def frame_name(tick):
    return f"{tick // 4:05d}_{tick % 4 * 25:02d}.jpeg"


class FrameStore:
    """Read only the new frames in (start, end], validating their bytes on access."""

    def __init__(self, path, source_hash=None, manifest_hash=None):
        self.path = Path(path)
        if not (self.path / "COMPLETE").is_file():
            raise ValueError("Incomplete frame cache")
        self.manifest_hash = file_hash(self.path / "manifest.json")
        if manifest_hash and self.manifest_hash != manifest_hash:
            raise ValueError("Frame manifest changed since run preparation")
        self.manifest = m = read_json(self.path / "manifest.json")
        if (m["scheme"] != SCHEME or
                (self.path / "COMPLETE").read_text().strip() != m["cache_fingerprint"]):
            raise ValueError("Frame cache sampling scheme or completion marker mismatch")
        if source_hash and m["source_sha256"] != source_hash:
            raise ValueError("Frame cache belongs to a different source video")
        expected = math.floor(float(m["source_duration_s"]) * 4 + 1e-7)
        self.frames = m["frames"]
        if expected < 0 or m["expected_frames"] != expected or len(self.frames) != expected:
            raise ValueError("Incomplete frame manifest")
        for tick, row in enumerate(self.frames, 1):
            if row["file"] != frame_name(tick) or row["time_s"] != tick / 4:
                raise ValueError("Frame cache has a gap or an incorrect timestamp")
        self.times = [row["time_s"] for row in self.frames]

    def load_interval(self, start, end):
        from PIL import Image

        rows = self.frames[bisect.bisect_right(self.times, start + 1e-8):
                           bisect.bisect_right(self.times, end + 1e-8)]
        images = []
        for row in rows:
            data = (self.path / row["file"]).read_bytes()
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("Corrupt cached frame")
            with Image.open(io.BytesIO(data)) as image:
                images.append(image.convert("RGB"))
        return images, [row["time_s"] for row in rows]


def build_frames(source, destination, source_hash):
    """Publish a complete cache atomically; also accept the original cache format."""
    destination = Path(destination)
    with lock_file(destination.parent / ("." + destination.name + ".lock")):
        if destination.exists():
            return FrameStore(destination, source_hash)
        info = json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=duration:format=duration", "-of", "json", str(source)], text=True))
        duration = info["streams"][0].get("duration")
        if duration in (None, "N/A"):
            duration = info.get("format", {}).get("duration")
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid source video duration")
        expected = math.floor(duration * 4 + 1e-7)
        ffmpeg = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
        cache_fingerprint = digest(dict(scheme=SCHEME, ffmpeg=ffmpeg,
                                        decoder_threads=2, encoder_threads=1))
        temporary = Path(tempfile.mkdtemp(prefix="." + destination.name + ".", dir=destination.parent))
        try:
            if expected:
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2",
                     "-i", str(source), "-map", "0:v:0", "-an", "-vf", SCHEME["filter"],
                     "-fps_mode", "passthrough", "-filter_threads", "1", "-threads", "1",
                     "-q:v", "2", "-frames:v", str(expected), "-start_number", "1",
                     str(temporary / "extract_%08d.jpeg")], check=True, capture_output=True)
            files = sorted(temporary.glob("extract_*.jpeg"))
            if len(files) != expected:
                raise ValueError("Decoded frame count differs from source duration")
            frames = []
            for tick, file in enumerate(files, 1):
                name = frame_name(tick)
                file.rename(temporary / name)
                frames.append(dict(file=name, time_s=tick / 4,
                                   bytes=(temporary / name).stat().st_size,
                                   sha256=file_hash(temporary / name)))
            write_json(temporary / "manifest.json", dict(
                schema=1, scheme=SCHEME, source_sha256=source_hash,
                source_duration_s=duration, expected_frames=expected, frames=frames,
                ffmpeg=ffmpeg, cache_fingerprint=cache_fingerprint))
            write_text(temporary / "COMPLETE", cache_fingerprint + "\n")
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return FrameStore(destination, source_hash)


def prepare_caches(sources, args):
    def prepare(pair):
        key, source = pair
        source_hash = file_hash(source)
        destination = Path(args.frames_root) / key
        if destination.exists() or args.cache_policy == "require":
            store = FrameStore(destination, source_hash)
        else:
            store = build_frames(source, destination, source_hash)
        return key, dict(manifest_sha256=store.manifest_hash,
                         n_frames=len(store.frames), source_sha256=source_hash)

    with ThreadPoolExecutor(max_workers=args.frame_workers) as pool:
        return dict(pool.map(prepare, sorted(sources.items())))


def resolve_model(args):
    """Resolve once before launching workers; store hashes, never a local model path."""
    model = Path(args.model)
    repo_id, revision = None, None
    if not model.is_dir():
        if model.is_absolute() or args.model.startswith("."):
            raise ValueError("Local model directory does not exist")
        from huggingface_hub import snapshot_download

        repo_id = args.model
        revision = args.revision or (MODEL_REVISION if repo_id == MODEL_ID else "main")
        model = Path(snapshot_download(repo_id, revision=revision))
        revision = model.name
    elif args.revision:
        raise ValueError("--revision applies to Hub repositories, not local directories")
    files = sorted(p for p in model.iterdir() if p.is_file() and
                   p.suffix in (".json", ".jinja", ".safetensors", ".bin", ".model"))
    if not (model / "config.json").is_file() or not any(
            p.suffix in (".safetensors", ".bin") for p in files):
        raise ValueError("Model directory lacks configuration or weights")
    identity = dict(name=args.model_name, repo_id=repo_id, revision=revision,
                    files={p.name: file_hash(p) for p in files})
    return model.resolve(), identity


def assign_shards(samples, count, a_window):
    groups = defaultdict(list)
    for sample in samples:
        groups[f"{sample['domain']}/{sample['video_id']}"].append(sample)
    def cost(group):
        return sum(sum(t >= (s["question_time_s"] if s["time_type"] == "A" else 0)
                       for t in decision_ticks(s, a_window)) for s in group)
    shards, loads = [[] for _ in range(count)], [0] * count
    for _, group in sorted(groups.items(), key=lambda kv: (-cost(kv[1]), kv[0])):
        rank = min(range(count), key=lambda i: (loads[i], i))
        shards[rank].extend(s["item_id"] for s in sorted(group, key=lambda s: s["item_id"]))
        loads[rank] += cost(group)
    return shards


def prepare_run(args, samples, caches, model_identity, options_hash):
    system = Path(args.system_prompt).read_text(encoding="utf-8")
    if not system.strip() or "<image>" in system:
        raise ValueError("Empty system prompt or reserved image placeholder")
    environment = {}
    for package in ("torch", "torchvision", "transformers", "accelerate", "Pillow", "flash-attn"):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = None
    protocol = dict(sample_fps=4, decision_interval_s=1, max_rounds=args.max_rounds,
                    max_new_tokens=args.max_new_tokens, min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels, a_window=args.a_window, seed=args.seed,
                    attn_implementation=args.attn_implementation, dtype="bfloat16",
                    do_sample=False, standby_resize=False, include_final_partial_round=True)
    config = dict(schema=1, model=model_identity, protocol=protocol, samples=samples,
                  frames=caches, system_sha256=hashlib.sha256(system.encode()).hexdigest(),
                  options_sha256=options_hash, full_set=not (args.items or args.limit),
                  environment=environment,
                  code_sha256={p: file_hash(HERE / p) for p in ("run.py", "inference.py")},
                  shards=assign_shards(samples, args.num_shards, args.a_window))
    selected = {s["item_id"] for s in samples}
    config["annotations_sha256"] = digest([
        asdict(it) for it in iter_items(load_benchmark(args.data)) if it.item_id in selected])
    manifest = dict(fingerprint=digest(config), config=config)
    out = Path(args.out)
    with lock_file(out / ".config.lock"):
        if (out / "run.json").exists():
            if read_json(out / "run.json") != manifest:
                raise ValueError("Run configuration changed; use a new --out directory")
            if file_hash(out / "system_prompt.txt") != config["system_sha256"]:
                raise ValueError("Saved system prompt changed")
        else:
            if any(p.name != ".config.lock" for p in out.iterdir()):
                raise ValueError("Output directory is not empty and has no run manifest")
            write_text(out / "system_prompt.txt", system)
            write_json(out / "run.json", manifest)
    return manifest


def load_run(out):
    manifest = read_json(Path(out) / "run.json")
    if digest(manifest["config"]) != manifest["fingerprint"]:
        raise ValueError("Run manifest checksum mismatch")
    if file_hash(Path(out) / "system_prompt.txt") != manifest["config"]["system_sha256"]:
        raise ValueError("Saved system prompt changed")
    return manifest


def prediction(sample, config, emissions, latencies):
    return dict(video_id=sample["video_id"], item_index=sample["item_index"],
                model=config["model"]["name"], run="onestreamer_native",
                emissions=emissions, n_polls=len(latencies), poll_latencies=latencies)


def run_item(sample, engine, config, system, frames_root, shard, fingerprint):
    key = f"{sample['domain']}/{sample['video_id']}"
    store = FrameStore(Path(frames_root) / key,
                       manifest_hash=config["frames"][key]["manifest_sha256"])
    protocol = config["protocol"]
    session = NativeSession(engine, system, sample["question"], sample["time_type"],
                            sample["question_time_s"], protocol["max_rounds"],
                            protocol["max_new_tokens"])
    raw_path = shard / "raw" / (sample["item_id"] + ".jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=raw_path.parent)
    emissions, latencies, start = [], [], 0.0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for end in decision_ticks(sample, protocol["a_window"]):
                images, times = store.load_interval(start, end)
                images = [engine.prepare_frame(image) for image in images]
                begin = time.perf_counter()
                result = session.step(images, times, start, end)
                latency = time.perf_counter() - begin
                latencies.append(latency)
                if result["kind"] == "response":
                    emissions.append(dict(t=end, content=result["content"], latency_s=latency))
                f.write(canonical(dict(start=start, t=end, frame_times=times,
                                       latency_s=latency, **result)) + "\n")
                f.flush()
                start = end
            os.fsync(f.fileno())
        os.replace(temporary, raw_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    record = dict(status="ok", item_id=sample["item_id"], run_fingerprint=fingerprint,
                  raw_sha256=file_hash(raw_path),
                  prediction=prediction(sample, config, emissions, latencies))
    write_json(shard / "items" / (sample["item_id"] + ".json"), record)
    return record["prediction"]


def validate_record(sample, shard, manifest):
    """Reconstruct the scored emissions from a complete, causal raw trace."""
    config, item_id = manifest["config"], sample["item_id"]
    record = read_json(shard / "items" / (item_id + ".json"))
    if (record["status"] != "ok" or record["item_id"] != item_id
            or record["run_fingerprint"] != manifest["fingerprint"]):
        raise ValueError("Unsuccessful item or mismatched run fingerprint")
    raw = shard / "raw" / (item_id + ".jsonl")
    if file_hash(raw) != record["raw_sha256"]:
        raise ValueError("Missing or corrupt raw trace")
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    protocol = config["protocol"]
    ticks = decision_ticks(sample, protocol["a_window"])
    if len(rows) != len(ticks):
        raise ValueError("Incomplete decision trace")
    frame_count = config["frames"][f"{sample['domain']}/{sample['video_id']}"]["n_frames"]
    history = deque(maxlen=protocol["max_rounds"])
    emissions, latencies, start = [], [], 0.0
    reveal = sample["question_time_s"] if sample["time_type"] == "A" else 0.0
    for row, end in zip(rows, ticks):
        expected_times = [k / 4 for k in range(math.floor(start * 4 + 1e-8) + 1,
                                              min(frame_count, math.floor(end * 4 + 1e-8)) + 1)]
        history.append(len(expected_times))
        generated = end + 1e-8 >= reveal
        if (row["start"] != start or row["t"] != end or row["frame_times"] != expected_times
                or row["generated"] != generated or row["question_visible"] != generated
                or row["history_rounds"] != len(history)
                or row["history_frames"] != sum(history)):
            raise ValueError("Incorrect timing, frame history or question reveal in trace")
        action = parse_action(row["raw"])
        if any(row[k] != v for k, v in action.items()):
            raise ValueError("Raw text and parsed action disagree")
        if not generated and row["raw"] != "</Silence>":
            raise ValueError("Model output before the question reveal")
        latency = row["latency_s"]
        if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise ValueError("Invalid latency")
        latencies.append(latency)
        if action["kind"] == "response":
            emissions.append(dict(t=end, content=action["content"], latency_s=latency))
        start = end
    pred = prediction(sample, config, emissions, latencies)
    if pred != record["prediction"]:
        raise ValueError("Prediction differs from raw trace")
    return pred


def execute_worker(args, manifest, rank, engine_factory=None):
    out, config = Path(args.out), manifest["config"]
    if any(file_hash(HERE / p) != h for p, h in config["code_sha256"].items()):
        raise ValueError("Runner code changed since run preparation")
    if not 0 <= rank < len(config["shards"]):
        raise ValueError("Invalid shard index")
    shard = out / "shards" / f"{rank:03d}"
    by_id = {s["item_id"]: s for s in config["samples"]}
    samples = [by_id[item_id] for item_id in config["shards"][rank]]
    with lock_file(shard / ".worker.lock", blocking=False):
        (shard / "items").mkdir(exist_ok=True)
        if {p.stem for p in (shard / "items").glob("*.json")} - set(config["shards"][rank]):
            raise ValueError("Unexpected items in shard")
        pending = []
        for sample in samples:
            if (shard / "items" / (sample["item_id"] + ".json")).exists():
                validate_record(sample, shard, manifest)
            else:
                pending.append(sample)
        if pending:
            if engine_factory is None:
                import torch
                import numpy as np

                torch.set_num_threads(2)
                random.seed(config["protocol"]["seed"])
                np.random.seed(config["protocol"]["seed"])
                torch.manual_seed(config["protocol"]["seed"])
                engine_factory = OneStreamerEngine
            p = config["protocol"]
            engine = engine_factory(args.model, min_pixels=p["min_pixels"],
                                    max_pixels=p["max_pixels"],
                                    attn_implementation=p["attn_implementation"])
            system = (out / "system_prompt.txt").read_text(encoding="utf-8")
            for sample in pending:
                try:
                    pred = run_item(sample, engine, config, system, args.frames_root,
                                    shard, manifest["fingerprint"])
                except Exception as exc:
                    write_json(shard / "error.json", dict(
                        item_id=sample["item_id"], exception_type=type(exc).__name__))
                    raise
                print(f"shard={rank} item={sample['item_id']} "
                      f"responses={len(pred['emissions'])}", flush=True)
        (shard / "error.json").unlink(missing_ok=True)
        write_json(shard / "complete.json", dict(fingerprint=manifest["fingerprint"],
                                                item_ids=config["shards"][rank]))


def merge_run(out):
    out = Path(out)
    manifest = load_run(out)
    config = manifest["config"]
    samples = {s["item_id"]: s for s in config["samples"]}
    assigned = [item for shard in config["shards"] for item in shard]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(samples):
        raise ValueError("Duplicate or missing items in shard plan")
    predictions = {}
    for rank, expected in enumerate(config["shards"]):
        shard = out / "shards" / f"{rank:03d}"
        if ((shard / "error.json").exists() or
                read_json(shard / "complete.json") !=
                dict(fingerprint=manifest["fingerprint"], item_ids=expected)):
            raise ValueError("Shard has not completed successfully")
        actual = {p.stem for p in (shard / "items").glob("*.json")}
        if actual != set(expected):
            raise ValueError("Missing, duplicate or unexpected shard items")
        for item_id in expected:
            predictions[item_id] = validate_record(samples[item_id], shard, manifest)
    write_text(out / "preds.jsonl", "".join(
        canonical(predictions[s["item_id"]]) + "\n" for s in config["samples"]))
    write_json(out / "validation.json", dict(status="passed", fingerprint=manifest["fingerprint"],
                                             n_items=len(predictions), full_set=config["full_set"]))
    print(f"Complete: {len(predictions)} items; score preds.jsonl with ibench eval", flush=True)


def launch_workers(args, manifest, devices):
    """Workers share prepared inputs, but never share a model process."""
    processes = []
    try:
        for rank in range(len(manifest["config"]["shards"])):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[rank // args.workers_per_gpu])
            command = [sys.executable, str(HERE / "run.py"), "--worker-rank", str(rank),
                       "--out", args.out, "--model", str(args.model),
                       "--frames-root", args.frames_root]
            processes.append(subprocess.Popen(command, env=env))
        active = set(processes)
        while active:
            for process in list(active):
                code = process.poll()
                if code is not None:
                    active.remove(process)
                    if code:
                        raise RuntimeError("An inference worker failed; rerun to resume")
            if active:
                time.sleep(0.2)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    merge_run(args.out)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/interactionbench")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--model-name", default="OneStreamer-4B", help="public display name")
    ap.add_argument("--revision", help="Hub revision; pinned for the default public model")
    ap.add_argument("--items", help="item list; default: benchmark/splits/all1060.txt")
    ap.add_argument("--mcq", nargs="?", const="auto", help="options file; no value selects MCQ v4")
    ap.add_argument("--system-prompt", default=str(HERE / "system_prompt.txt"))
    ap.add_argument("--frames-root", default="results/frame_cache/onestreamer_4fps")
    ap.add_argument("--cache-policy", choices=("build", "require"), default="build")
    ap.add_argument("--frame-workers", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=32)
    ap.add_argument("--min-pixels", type=int, default=3136)
    ap.add_argument("--max-pixels", type=int, default=100352)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--a-window", type=float, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn-implementation", default="flash_attention_2",
                    choices=("flash_attention_2", "sdpa", "eager"))
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--devices", help="comma-separated GPU IDs for CUDA_VISIBLE_DEVICES")
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/runs/onestreamer_native")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--prepare-frames", action="store_true")
    modes.add_argument("--merge-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true", help="validate inputs without loading a model")
    ap.add_argument("--worker-rank", type=int, help=argparse.SUPPRESS)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker_rank is not None:
        execute_worker(args, load_run(args.out), args.worker_rank)
        return
    if args.merge_only:
        merge_run(args.out)
        return
    if (min(args.num_shards, args.workers_per_gpu, args.frame_workers, args.max_rounds,
            args.max_new_tokens, args.min_pixels) < 1 or args.max_pixels < args.min_pixels
            or args.limit < 0 or not 0 <= args.shard_index < args.num_shards):
        raise ValueError("Invalid limits or shard selection")
    devices = args.devices.split(",") if args.devices else []
    if devices:
        if args.num_shards != 1 or args.shard_index != 0:
            raise ValueError("Use --devices or explicit sharding, not both")
        if len(set(devices)) != len(devices) or any(not s.strip() for s in devices):
            raise ValueError("Invalid device list")
        args.num_shards = len(devices) * args.workers_per_gpu
    samples, sources, options_hash = select_samples(args)
    caches = prepare_caches(sources, args)
    print(f"Validated {len(samples)} items and {len(caches)} video caches", flush=True)
    if args.prepare_frames or args.dry_run:
        return
    model, identity = resolve_model(args)
    manifest = prepare_run(args, samples, caches, identity, options_hash)
    args.model = str(model)
    if devices:
        with lock_file(Path(args.out) / ".launch.lock", blocking=False):
            launch_workers(args, manifest, devices)
    else:
        execute_worker(args, manifest, args.shard_index)
        if args.num_shards == 1:
            merge_run(args.out)


if __name__ == "__main__":
    main()
