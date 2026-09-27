"""Shared fixtures: a tiny synthetic benchmark on disk, no videos decoded, no GPU."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _annotation(video_id, duration, items):
    return {"video_id": video_id, "domain": "demo", "video": f"videos/demo/{video_id}.mp4",
            "duration_s": duration, "items": items}


@pytest.fixture()
def bench(tmp_path):
    """Four items over two videos: an A-type question, a B-type trigger with options,
    a counting stream and a negative item."""
    root = tmp_path / "data"
    videos = {
        "vidAAAAAAAA": (20.0, [
            {"capability": "LVM", "time_type": "A", "interaction_type": "QA",
             "question": "What colour was the cup?", "question_time_s": 15.0,
             "answers": [{"time_s": 15.0, "content": "blue", "evidence_time_s": 4.0}]},
            {"capability": "PTR", "time_type": "B", "interaction_type": "INS",
             "question": "Tell me when the door opens.", "question_time_s": 0.0,
             "answers": [{"time_s": 8.0, "content": "The door opens."}]},
        ]),
        "vidBBBBBBBB": (12.0, [
            {"capability": "CST", "time_type": "C", "interaction_type": "INS", "auto_number": True,
             "question": "Count the jumps.", "question_time_s": 0.0,
             "answers": [{"time_s": 3.0, "content": "1"}, {"time_s": 6.0, "content": "2"},
                         {"time_s": 9.0, "content": "3"}]},
            {"capability": "PTR", "time_type": "B", "interaction_type": "INS", "is_negative": True,
             "question": "Tell me when a dog appears.", "question_time_s": 0.0,
             "answers": [{"time_s": None, "content": "SHOULD_REMAIN_SILENT"}]},
        ]),
    }
    for vid, (dur, items) in videos.items():
        d = root / "annotations" / "demo"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{vid}.json").write_text(json.dumps(_annotation(vid, dur, items)))
        v = root / "videos" / "demo"
        v.mkdir(parents=True, exist_ok=True)
        (v / f"{vid}.mp4").write_bytes(b"placeholder: tests never decode it")
    mcq = root / "mcq"
    mcq.mkdir()
    (mcq / "mcq_options_v4.jsonl").write_text(json.dumps(
        {"item_id": "vidAAAAAAAA#0", "stem": "Which colour was the cup?",
         "options": ["red", "blue", "green"]}) + "\n")
    (mcq / "mcq_key_v4.jsonl").write_text(json.dumps(
        {"item_id": "vidAAAAAAAA#0", "correct_index": 1, "correct_letter": "B",
         "answer_text": "blue", "distractors": ["red", "green"]}) + "\n")
    return root


class Img:
    """Stand-in for a PIL image."""

    def __init__(self, t):
        self.t = t


@pytest.fixture()
def make_frames():
    from interactionbench.frames import Frame

    def _make(duration, fps=2.0):
        return [Frame(time=(i + 0.5) / fps, image=Img((i + 0.5) / fps))
                for i in range(int(duration * fps))]
    return _make
