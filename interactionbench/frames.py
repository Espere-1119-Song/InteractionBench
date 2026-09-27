"""Frame sampling for the polling protocols.

The whole video is decoded once at a fixed ``sample_fps`` into an in-memory list of
``(timestamp_seconds, PIL.Image)``. A protocol then slices this list to build the
visual context available "up to time t" at each decision step, so the video is never
re-decoded per step.

Decoding shells out to an ``ffmpeg`` binary on PATH (no decord / PyAV dependency).
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class Frame:
    time: float          # approximate timestamp in seconds
    image: "Image.Image"


def extract_frames(
    video_path: str | Path,
    sample_fps: float = 2.0,
    max_long_side: int | None = 512,
) -> list[Frame]:
    """Decode ``video_path`` into frames sampled at ``sample_fps``.

    Frame i is assigned timestamp ``(i + 0.5) / sample_fps`` (the centre of its
    sampling interval). ``max_long_side`` optionally downscales frames to cap the
    longer side (keeps token cost predictable); pass ``None`` to keep native size.
    """
    from PIL import Image

    video_path = Path(video_path)
    frames: list[Frame] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        vf = f"fps={sample_fps}"
        if max_long_side:
            # scale so the longer side <= max_long_side, preserve aspect, even dims
            vf += (
                f",scale='if(gt(iw,ih),min({max_long_side},iw),-2)':"
                f"'if(gt(iw,ih),-2,min({max_long_side},ih))'"
            )
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-vf", vf,
            "-vsync", "vfr",
            str(tmp_dir / "f_%06d.jpg"),
        ]
        subprocess.run(cmd, check=True)
        files = sorted(tmp_dir.glob("f_*.jpg"))
        for i, fp in enumerate(files):
            t = (i + 0.5) / sample_fps
            frames.append(Frame(time=t, image=Image.open(fp).convert("RGB").copy()))
    return frames


def frames_up_to(frames: list[Frame], t: float) -> list[Frame]:
    """All frames whose timestamp is <= t (the streaming history visible at poll t)."""
    return [f for f in frames if f.time <= t]


def subsample(frames: list[Frame], max_frames: int) -> list[Frame]:
    """Evenly subsample to at most ``max_frames`` frames, always keeping the last one."""
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    n = len(frames)
    # evenly spaced indices across the window, inclusive of the final frame
    idxs = sorted({round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)})
    return [frames[i] for i in idxs]

