"""Loader for the InteractionBench annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SILENT_TOKEN = "SHOULD_REMAIN_SILENT"
COUNTING_SUB_TAGS = ("counting", "计数型")

CAPABILITIES = ("LVM", "IVQA", "PTR", "TOA", "CIR", "CST", "LCG", "BRC")


@dataclass
class GTAnswer:
    time_s: float | None
    content: str | None
    evidence_time_s: float | None = None

    @property
    def is_silent(self) -> bool:
        return bool(self.content) and SILENT_TOKEN in self.content.upper()


@dataclass
class BenchItem:
    video_id: str
    item_index: int
    capability: str
    time_type: str | None
    interaction_type: str | None
    sub_tag: str | None
    is_negative: bool
    auto_number: bool
    question: str
    question_time_s: float
    answers: list[GTAnswer]
    duration_s: float
    domain: str = "author"
    range_length: str | None = None
    notes: str = ""

    @property
    def item_id(self) -> str:
        return f"{self.video_id}#{self.item_index}"

    @property
    def should_remain_silent(self) -> bool:
        return self.is_negative or (
            len(self.answers) > 0 and all(a.is_silent for a in self.answers)
        )

    @property
    def is_counting(self) -> bool:
        return self.auto_number or self.sub_tag in COUNTING_SUB_TAGS

    @property
    def timed_answers(self) -> list[GTAnswer]:
        return [a for a in self.answers if a.time_s is not None and not a.is_silent]

    @property
    def valid(self) -> bool:
        if self.time_type not in ("A", "B", "C"):
            return False
        return self.should_remain_silent or len(self.timed_answers) > 0


@dataclass
class BenchVideo:
    video_id: str
    domain: str
    duration_s: float
    not_annotatable: bool
    items: list[BenchItem] = field(default_factory=list)


def load_video(annotation_path: Path) -> BenchVideo:
    d = json.loads(annotation_path.read_text(encoding="utf-8"))
    legacy = annotation_path.name == "annotation.json"
    vid = d.get("video_id") or (annotation_path.parent.name if legacy else annotation_path.stem)
    domain = (d.get("domain") or d.get("category")
              or (annotation_path.parent.parent.name if legacy else annotation_path.parent.name))
    dur = float(d.get("duration_s") or 0.0)
    video = BenchVideo(
        video_id=vid,
        domain=domain,
        duration_s=dur,
        not_annotatable=bool(d.get("not_annotatable")),
    )
    for i, it in enumerate(d.get("items") or []):
        answers = [
            GTAnswer(
                time_s=a.get("time_s"),
                content=a.get("content"),
                evidence_time_s=a.get("evidence_time_s"),
            )
            for a in (it.get("answers") or [])
        ]
        video.items.append(
            BenchItem(
                video_id=vid,
                item_index=i,
                capability=it.get("capability") or "UNKNOWN",
                time_type=it.get("time_type"),
                interaction_type=it.get("interaction_type"),
                sub_tag=it.get("sub_tag"),
                is_negative=bool(it.get("is_negative")),
                auto_number=bool(it.get("auto_number")),
                question=it.get("question") or "",
                question_time_s=float(it.get("question_time_s") or 0.0),
                answers=answers,
                duration_s=dur,
                domain=domain,
                range_length=it.get("range_length"),
                notes=it.get("notes") or "",
            )
        )
    return video


def load_benchmark(root: str | Path) -> list[BenchVideo]:
    root = Path(root)
    if (root / "annotations").is_dir():
        root = root / "annotations"
    elif (root / "results").is_dir():
        root = root / "results"
    legacy = sorted(root.rglob("annotation.json"))
    if legacy:
        return [load_video(p) for p in legacy]
    paths = sorted(root.rglob("*.json"), key=lambda p: (*p.parent.parts, p.stem))
    return [load_video(p) for p in paths]


def iter_items(videos: list[BenchVideo], only_valid: bool = True):
    for v in videos:
        for it in v.items:
            if only_valid and not it.valid:
                continue
            yield it


if __name__ == "__main__":
    import sys
    from collections import Counter

    root = sys.argv[1] if len(sys.argv) > 1 else "data/interactionbench"
    vids = load_benchmark(root)
    items = list(iter_items(vids))
    skipped = [it for v in vids for it in v.items if not it.valid]
    print(f"{len(vids)} videos, {len(items)} valid items, {len(skipped)} skipped")
    print("capability:", dict(Counter(i.capability for i in items)))
    print("time_type:", dict(Counter(i.time_type for i in items)))
    print("negatives:", sum(i.should_remain_silent for i in items))
    print("counting:", sum(i.is_counting for i in items))
