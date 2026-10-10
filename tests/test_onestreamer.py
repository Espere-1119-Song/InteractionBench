"""Causal protocol, cache and recovery checks; no model weights or GPU required."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image
import pytest

from baselines.onestreamer import run
from baselines.onestreamer.inference import NativeSession, OneStreamerEngine, decision_ticks, parse_action


class FakeEngine:
    def __init__(self, *args, **kwargs):
        self.calls = []

    @staticmethod
    def prepare_frame(image):
        return image

    def infer(self, data, **kwargs):
        self.calls.append(data)
        return ("</Silence>", "</Standby>", "</Response> 1", "invalid")[len(self.calls) % 4]


def test_fractional_reveal_and_partial_last_round():
    sample = dict(duration_s=12.6, question_time_s=12.4, time_type="A")
    ticks = decision_ticks(sample)
    assert ticks == [float(i) for i in range(1, 13)] + [12.6]
    engine = FakeEngine()
    session = NativeSession(engine, "system", "secret question", "A", 12.4)
    start = 0
    for end in ticks:
        session.step([], [], start, end)
        if end < 12.4:
            assert not engine.calls
            assert all("secret question" not in m["content"] for m in session.last_input["messages"])
        start = end
    assert len(engine.calls) == 1
    assert "secret question" in engine.calls[0]["messages"][-1]["content"]
    assert decision_ticks(dict(sample, duration_s=0.15, question_time_s=0.15)) == [0.15]


def test_rolling_history_preserves_question_and_actual_replies():
    engine = FakeEngine()
    session = NativeSession(engine, "system", "standing question", "C", 0)
    for index in range(36):
        times = [index + k / 4 for k in range(1, 5)]
        result = session.step(times, times, index, index + 1)
    assert result["history_rounds"] == 32 and result["history_frames"] == 128
    data = engine.calls[-1]
    assert data["images"] == [k / 4 for k in range(17, 145)]
    assert data["messages"][1]["content"].startswith("standing question\n<4s-5s>")
    replies = [m["content"] for m in data["messages"] if m["role"] == "assistant"]
    assert len(replies) == 31 and "invalid" in replies and "</Response> 1" in replies


@pytest.mark.parametrize("times", ([1.25], [0], [0.5, 0.25], [0.25, 0.25]))
def test_rejects_future_or_repeated_frames(times):
    with pytest.raises(ValueError):
        NativeSession(FakeEngine(), "system", "question", "B", 0).step(times, times, 0, 1)


@pytest.mark.parametrize("text,kind,content", [
    ("</Silence>", "silence", None), (" </standby> ", "standby", None),
    ("</Response> B", "response", "B"), ("</Response>", "invalid", None),
    ("prefix </Response> B", "invalid", None),
    ("</Silence> explanation", "invalid", None),
    ("</Response> B </Response> C", "invalid", None),
    ("</Standby></Response> B", "invalid", None), ("B", "invalid", None),
])
def test_action_parser(text, kind, content):
    parsed = parse_action(text)
    assert (parsed["kind"], parsed["content"]) == (kind, content)


def test_chat_conversion_matches_image_first_reference_order():
    image = object()
    converted = OneStreamerEngine.to_chat_messages(
        [{"role": "system", "content": " system\n"},
         {"role": "user", "content": "question\n<0s-1s>\n<image>\n"}], [image])
    assert converted[0]["content"] == [{"type": "text", "text": " system\n"}]
    assert converted[1]["content"] == [
        {"type": "image", "image": image}, {"type": "text", "text": "question\n<0s-1s>"}]


@pytest.fixture
def prepared(bench, tmp_path):
    ids = [it.item_id for it in run.iter_items(run.load_benchmark(bench))]
    selection = tmp_path / "items.txt"
    selection.write_text("\n".join(ids))
    args = run.parser().parse_args([
        "--data", str(bench), "--items", str(selection), "--mcq",
        "--frames-root", str(tmp_path / "frames"), "--out", str(tmp_path / "output")])
    samples, sources, options_hash = run.select_samples(args)
    caches = {}
    for key, source in sources.items():
        sample = next(s for s in samples if f"{s['domain']}/{s['video_id']}" == key)
        cache = Path(args.frames_root) / key
        cache.mkdir(parents=True)
        frames = []
        for tick in range(1, int(sample["duration_s"] * 4) + 1):
            file = cache / run.frame_name(tick)
            Image.new("RGB", (28, 28), (tick, 10, 20)).save(file)
            frames.append(dict(file=file.name, time_s=tick / 4, bytes=file.stat().st_size,
                               sha256=run.file_hash(file)))
        source_hash = run.file_hash(source)
        run.write_json(cache / "manifest.json", dict(
            scheme=run.SCHEME, cache_fingerprint="test-cache", source_sha256=source_hash,
            source_duration_s=sample["duration_s"], expected_frames=len(frames), frames=frames))
        (cache / "COMPLETE").write_text("test-cache\n")
        store = run.FrameStore(cache, source_hash)
        caches[key] = dict(manifest_sha256=store.manifest_hash, n_frames=len(frames),
                           source_sha256=source_hash)
    identity = dict(name="OneStreamer-4B", repo_id="example/test", revision="test", files={})
    manifest = run.prepare_run(args, samples, caches, identity, options_hash)
    return args, manifest


def test_inference_inputs_exclude_references_and_missing_annotations_fail(prepared):
    args, manifest = prepared
    samples = manifest["config"]["samples"]
    assert all(not ({"answers", "is_negative", "capability", "evidence_time_s"} & s.keys())
               for s in samples)
    assert "A. red\nB. blue" in samples[0]["question"]
    Path(args.items).write_text("unknown#0\n")
    with pytest.raises(ValueError, match="Missing annotations"):
        run.select_samples(args)


def test_missing_mcq_options_do_not_fall_back_to_free_form(prepared, monkeypatch):
    args, manifest = prepared
    original = run.read_ids
    monkeypatch.setattr(run, "read_ids", lambda p: {"vidBBBBBBBB#0"}
                        if Path(p).name == "mcq688.txt" else original(p))
    with pytest.raises(ValueError, match="Missing v4 MCQ"):
        run.select_samples(args)


def test_cache_detects_corrupt_pixels_and_changed_source(prepared):
    args, manifest = prepared
    key = next(iter(manifest["config"]["frames"]))
    cache = Path(args.frames_root) / key
    store = run.FrameStore(cache)
    _, times = store.load_interval(0.5, 1.0)
    assert times == [0.75, 1.0]
    (cache / run.frame_name(1)).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt cached frame"):
        store.load_interval(0, 0.25)
    with pytest.raises(ValueError, match="different source"):
        run.FrameStore(cache, "wrong-source")


def test_complete_inference_resume_and_official_score_format(prepared):
    args, manifest = prepared
    run.execute_worker(args, manifest, 0, FakeEngine)
    run.merge_run(args.out)
    before = (Path(args.out) / "preds.jsonl").read_bytes()
    def must_not_load(*args, **kwargs):
        pytest.fail("Completed items should not reload the model")
    run.execute_worker(args, manifest, 0, must_not_load)
    run.merge_run(args.out)
    assert (Path(args.out) / "preds.jsonl").read_bytes() == before
    predictions = [json.loads(line) for line in before.splitlines()]
    assert len(predictions) == 4
    assert all(p["n_polls"] == len(p["poll_latencies"]) for p in predictions)
    from interactionbench.evaluate import EvalConfig, evaluate
    _, records = evaluate(EvalConfig(
        predictions=str(Path(args.out) / "preds.jsonl"), data=args.data, items=args.items,
        mcq_key=str(Path(args.data) / "mcq/mcq_key_v4.jsonl")))
    assert len(records) == 4


def test_exception_is_not_a_successful_silent_item(prepared):
    args, manifest = prepared
    class FailingEngine(FakeEngine):
        def infer(self, data, **kwargs):
            raise RuntimeError("simulated inference failure")
    with pytest.raises(RuntimeError):
        run.execute_worker(args, manifest, 0, FailingEngine)
    shard = Path(args.out) / "shards/000"
    assert (shard / "error.json").exists()
    assert not list((shard / "items").glob("*.json"))
    assert not (Path(args.out) / "preds.jsonl").exists()
    with pytest.raises((ValueError, FileNotFoundError)):
        run.merge_run(args.out)
    run.execute_worker(args, manifest, 0, FakeEngine)
    run.merge_run(args.out)
    assert not (shard / "error.json").exists()


def test_resume_rejects_changed_configuration(prepared):
    args, manifest = prepared
    c = manifest["config"]
    args.max_rounds += 1
    with pytest.raises(ValueError, match="configuration changed"):
        run.prepare_run(args, c["samples"], c["frames"], c["model"], c["options_sha256"])


def test_missing_and_extra_predictions_block_merge(prepared):
    args, manifest = prepared
    run.execute_worker(args, manifest, 0, FakeEngine)
    folder = Path(args.out) / "shards/000/items"
    file = next(folder.glob("*.json"))
    content = file.read_bytes()
    file.unlink()
    with pytest.raises(ValueError, match="shard items"):
        run.merge_run(args.out)
    file.write_bytes(content)
    (folder / "extra#0.json").write_bytes(content)
    with pytest.raises(ValueError, match="shard items"):
        run.merge_run(args.out)


def test_semantically_noncausal_trace_rejected_even_with_updated_hash(prepared):
    args, manifest = prepared
    run.execute_worker(args, manifest, 0, FakeEngine)
    sample = manifest["config"]["samples"][0]
    shard = Path(args.out) / "shards/000"
    raw = shard / "raw" / (sample["item_id"] + ".jsonl")
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    rows[0]["frame_times"][-1] = 1.25
    raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    file = shard / "items" / (sample["item_id"] + ".json")
    record = run.read_json(file)
    record["raw_sha256"] = run.file_hash(raw)
    run.write_json(file, record)
    with pytest.raises(ValueError, match="Incorrect timing"):
        run.validate_record(sample, shard, manifest)


def test_shards_are_disjoint_video_groups(prepared):
    _, manifest = prepared
    samples = manifest["config"]["samples"]
    shards = run.assign_shards(samples, 8, 10)
    assert sorted(i for shard in shards for i in shard) == sorted(s["item_id"] for s in samples)
    location = {item: rank for rank, shard in enumerate(shards) for item in shard}
    assert location["vidAAAAAAAA#0"] == location["vidAAAAAAAA#1"]
    assert location["vidBBBBBBBB#0"] == location["vidBBBBBBBB#1"]
    assert shards == run.assign_shards(list(reversed(samples)), 8, 10)


def test_launcher_binds_workers_and_propagates_failure(prepared, monkeypatch):
    args, manifest = prepared
    manifest["config"]["shards"] = [[], [], [], []]
    args.workers_per_gpu = 2
    processes, environments = [], []
    class Process:
        def __init__(self, command, env):
            environments.append(env["CUDA_VISIBLE_DEVICES"])
            self.code = 7 if not processes else None
            self.terminated = False
            processes.append(self)
        def poll(self):
            return self.code
        def terminate(self):
            self.terminated, self.code = True, -15
        def wait(self, **kwargs):
            return self.code
    monkeypatch.setattr(run.subprocess, "Popen", Process)
    with pytest.raises(RuntimeError, match="worker failed"):
        run.launch_workers(args, manifest, ["2", "5"])
    assert environments == ["2", "2", "5", "5"]
    assert all(p.terminated for p in processes[1:])


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="ffmpeg and ffprobe are required for sampling integration")
def test_ffmpeg_quarter_grid_uses_only_available_source_frames(tmp_path):
    for index in range(6):
        Image.new("RGB", (32, 32), (40 + index * 30,) * 3).save(tmp_path / f"frame{index:02d}.png")
    source = tmp_path / "source.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "3", "-i",
                    str(tmp_path / "frame%02d.png"), "-c:v", "ffv1", str(source)], check=True)
    store = run.build_frames(source, tmp_path / "cache", run.file_hash(source))
    images, times = store.load_interval(0, 2)
    assert times == [i / 4 for i in range(1, 9)]
    assert [round(image.getpixel((16, 16))[0]) for image in images] == [
        40, 70, 100, 130, 130, 160, 190, 190]
    reused = run.build_frames(source, tmp_path / "cache", run.file_hash(source))
    assert reused.manifest_hash == store.manifest_hash


def test_help_does_not_import_gpu_libraries():
    script = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'torchvision'}:
        raise AssertionError('Unexpected GPU dependency import')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['run.py', '--help']
runpy.run_path('baselines/onestreamer/run.py', run_name='__main__')
"""
    subprocess.run([sys.executable, "-c", script], cwd=run.REPO, check=True, capture_output=True)
