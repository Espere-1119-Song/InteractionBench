"""Judges that only read stored verdicts. No model is loaded.

``CacheOnlyJudge`` replays the verdicts of one judge, which reproduces published
numbers from published verdict files without a GPU. ``EnsembleJudge`` takes a majority
vote over the stored verdicts of several judges.
"""

from __future__ import annotations

import glob
from pathlib import Path

from .base import cache_key, read_cache, trivial_verdict


def _load(patterns: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    files = [fp for pat in patterns for fp in sorted(glob.glob(pat))]
    if not files:
        raise FileNotFoundError(f"no verdict files match {patterns}")
    for fp in files:
        out.update(read_cache(fp))
    return out


class CacheOnlyJudge:
    """spec: ``cache:<file-or-glob>[,<file-or-glob>...]``

    A pair that is absent from the files scores 0 and is counted in ``n_missing``;
    with ``strict=True`` it raises instead."""

    def __init__(self, patterns: list[str], strict: bool = False, **_ignored):
        self.name = "cache:" + ",".join(patterns)
        self.verdicts = _load(patterns)
        self.strict = strict
        self.n_calls = 0
        self.n_missing = 0

    def __call__(self, question: str, gt: str, pred: str) -> float:
        gt, pred = (gt or "").strip(), (pred or "").strip()
        v = trivial_verdict(gt, pred)
        if v is not None:
            return v
        self.n_calls += 1
        k = cache_key(question or "", gt, pred)
        if k not in self.verdicts:
            self.n_missing += 1
            if self.strict:
                raise KeyError(f"no stored verdict for question={question!r} pred={pred!r}")
            return 0.0
        return self.verdicts[k]


class EnsembleJudge:
    """spec: ``ensemble:<slug1>,<slug2>,<slug3>``

    Each slug names verdict files ``<cache_dir>/<slug>_<prompt_version>*.jsonl``.
    Missing verdicts are skipped; a tie falls back to the first judge's verdict."""

    def __init__(self, slugs: list[str], prompt_version: str = "v2",
                 cache_dir: str | Path = "results/judge_cache", **_ignored):
        self.name = "ensemble:" + ",".join(slugs)
        self.caches = [_load([f"{cache_dir}/{slug}_{prompt_version}*.jsonl"]) for slug in slugs]
        self.n_calls = 0
        self.n_missing = 0

    def __call__(self, question: str, gt: str, pred: str) -> float:
        gt, pred = (gt or "").strip(), (pred or "").strip()
        v = trivial_verdict(gt, pred)
        if v is not None:
            return v
        k = cache_key(question or "", gt, pred)
        votes = [c[k] for c in self.caches if k in c]
        self.n_calls += 1
        if len(votes) < len(self.caches):
            self.n_missing += 1
        if not votes:
            return 0.0
        total = sum(votes) * 2
        if total > len(votes):
            return 1.0
        if total < len(votes):
            return 0.0
        return votes[0]
