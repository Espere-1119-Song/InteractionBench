"""Judge interface and the shared grading prompt."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PROMPTS = {
    "v1": """You are grading answers from a real-time video assistant.

Question asked about the video:
{question}

Ground-truth answer:
{gt}

Model's answer:
{pred}

Does the model's answer convey the same information as the ground truth?
Ignore wording, fillers, and phrasing differences; judge meaning only.
A paraphrase or description that clearly refers to the same thing counts as
a match. Numbers, directions and yes/no polarity must match to count.
This is a strict binary judgment: an incomplete or partially correct answer
counts as NO.

Reply with exactly one line: SCORE: 1 (correct) or SCORE: 0 (incorrect)""",
    "v2": """You are grading answers from a real-time video assistant.

Question asked about the video:
{question}

Ground-truth answer:
{gt}

Model's answer:
{pred}

Does the model's answer contain the information stated in the ground truth?
Ignore wording, fillers, and phrasing differences; judge meaning only.
A paraphrase or description that clearly refers to the same thing counts as
a match. If the model's answer states the ground-truth information and also
adds further details or steps, it still counts as a match. Numbers, directions
and yes/no polarity must match to count. An answer that omits or contradicts
the ground-truth information counts as NO.

Reply with exactly one line: SCORE: 1 (correct) or SCORE: 0 (incorrect)""",
}
DEFAULT_PROMPT_VERSION = "v2"

_SCORE_RE = re.compile(r"(?:score\s*:\s*)?(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_score(text: str) -> float:
    m = _SCORE_RE.search(text or "")
    if not m:
        return 0.0
    v = float(m.group(1))
    if v > 10:
        v /= 100.0
    elif v > 1:
        v /= 10.0
    return 1.0 if v >= 0.5 else 0.0


def cache_key(question: str, gt: str, pred: str) -> str:
    h = hashlib.sha256()
    for part in (question, "\x00", gt, "\x00", pred):
        h.update(part.encode("utf-8"))
    return h.hexdigest()[:32]


def read_cache(path: str | Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["k"]] = d["s"]
    return out


def trivial_verdict(gt: str, pred: str) -> float | None:
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    return None


class CachedJudge:
    name = "judge"

    def __init__(self, cache_path: str | Path | None = None,
                 prompt_version: str = DEFAULT_PROMPT_VERSION):
        if prompt_version not in PROMPTS:
            raise KeyError(f"unknown judge prompt '{prompt_version}'. known: {sorted(PROMPTS)}")
        self.prompt_version = prompt_version
        self.prompt = PROMPTS[prompt_version]
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, float] = {}
        if self.cache_path and self.cache_path.exists():
            self._cache = read_cache(self.cache_path)
        self.n_calls = 0

    _key = staticmethod(cache_key)

    def __call__(self, question: str, gt: str, pred: str) -> float:
        gt, pred = (gt or "").strip(), (pred or "").strip()
        v = trivial_verdict(gt, pred)
        if v is not None:
            return v
        k = cache_key(question or "", gt, pred)
        if k in self._cache:
            return self._cache[k]
        raw = self._generate(self.prompt.format(question=question or "(none)", gt=gt, pred=pred))
        self.n_calls += 1
        s = parse_score(raw)
        self._cache[k] = s
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"k": k, "s": s}) + "\n")
        return s

    def _generate(self, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError
