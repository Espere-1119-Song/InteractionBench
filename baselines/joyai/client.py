"""Client for the JoyAI-VL-Interaction live adapter (the streaming service, default
port 8070).

Upstream: https://github.com/jd-opensource/JoyAI-VL-Interaction
(``services/webinfer/live_adapter.py``); checkpoint
``jdopensource/JoyAI-VL-Interaction-Preview``.

JoyAI is a stateful streaming interaction model: a session accumulates video history
and long-term memory on the server, the standing question persists, and each call
returns either ``</silence>`` (stay quiet) or ``</response> <text>`` (speak now). This
is the native equivalent of the benchmark's SPEAK/WAIT decision. The client feeds only
the new frames at each step and maps the output onto the poll schema used by the
polling protocols, so the predictions are scored by the same evaluator.

Environment: standard library only, plus Pillow for JPEG encoding of the frames.
This module is imported by ``baselines/joyai/run.py``; it has no command line.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.request
import uuid

from interactionbench.frames import Frame


def _data_url(img, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _post(url: str, body: dict, headers: dict, timeout: float = 900.0) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class JoyAISession:
    """One streaming session against the JoyAI live adapter."""

    def __init__(
        self,
        base: str = "http://127.0.0.1:8070/v1",
        model: str = "JoyAI-VL-Interaction-Preview",
        session_id: str | None = None,
        frame_dt: float = 0.5,
    ):
        self.base = base.rstrip("/")
        self.model = model
        self.sid = session_id or uuid.uuid4().hex
        self.frame_dt = frame_dt
        self.reset()

    def reset(self) -> None:
        try:
            _post(f"{self.base}/streaming/reset", {"user": self.sid}, self._headers(), timeout=30)
        except Exception:
            pass  # a fresh session id needs no reset; ignore an error from the endpoint

    def _headers(self) -> dict:
        return {"x-streaming-session": self.sid}

    def step(self, frames: list[Frame], question: str, max_new_tokens: int = 64) -> tuple[str, float]:
        """Feed new frames + standing question; return (raw_marker_text, latency_s)."""
        content = [{"type": "image_url", "image_url": {"url": _data_url(f.image)}} for f in frames]
        content.append({"type": "text", "text": question})
        trs = [f"{f.time:.2f}-{f.time + self.frame_dt:.2f} seconds" for f in frames]
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "frame_time_ranges": trs,
            "max_tokens": max_new_tokens,
        }
        t0 = time.perf_counter()
        resp = _post(f"{self.base}/chat/completions", body, self._headers())
        dt = time.perf_counter() - t0
        raw = (resp.get("streamingharness") or {}).get("raw_content")
        if not raw:
            raw = resp["choices"][0]["message"]["content"]
        return (raw or "").strip(), dt


def parse_marker(raw: str) -> tuple[bool, str | None]:
    """Map JoyAI ``</silence>`` / ``</response> text`` to (spoke, response)."""
    s = (raw or "").strip()
    if s.startswith("</response>"):
        text = s[len("</response>"):].strip()
        # drop any trailing delegation marker content
        text = text.split("</delegation>")[0].split("<delegation>")[0].strip()
        return True, (text or None)
    if "</response>" in s and "</silence>" not in s:
        text = s.split("</response>", 1)[1].strip()
        return True, (text or None)
    return False, None
