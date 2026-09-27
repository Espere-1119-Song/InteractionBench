"""Parsers that turn raw model text into speak/wait decisions and timed emissions."""

from __future__ import annotations

import re

from .data import BenchItem

_DECISION_RE = re.compile(r"decision\s*:\s*(speak|wait)", re.IGNORECASE)
_RESPONSE_RE = re.compile(r"response\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
_TIMED_LINE_RE = re.compile(r"\s*\[?\s*t\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*\]?\s*(.*)")


def parse_decision(text: str) -> tuple[bool, str | None]:
    """Parse the two-line ``DECISION: / RESPONSE:`` format. Returns (spoke, response)."""
    m = _DECISION_RE.search(text)
    spoke = bool(m and m.group(1).lower() == "speak")
    response = None
    rm = _RESPONSE_RE.search(text)
    if rm and rm.group(1).strip():
        response = rm.group(1).strip().splitlines()[0].strip()
    if m is None and response:  # the model ignored the format but produced content
        spoke = True
    return spoke, (response or None)


def parse_offline(text: str, item: BenchItem) -> list[dict]:
    """Parse ``[t=12.3] content`` lines into emissions.

    For A-type items the answer is re-anchored to ``question_time_s``: the timestamp
    the model reports is evidence grounding (kept in the raw log only), while
    Timing Accuracy measures answering when asked."""
    out = []
    for line in (text or "").splitlines():
        m = _TIMED_LINE_RE.match(line.strip())
        if not m:
            continue
        t, content = float(m.group(1)), m.group(2).strip()
        if not content:
            continue
        out.append({"t": t, "content": content})
    if item.time_type == "A" and out:
        out = [{"t": item.question_time_s, "content": out[0]["content"]}]
    return sorted(out, key=lambda e: e["t"])
