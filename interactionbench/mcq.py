"""Multiple-choice rendering and scoring."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .metrics import normalize_text, token_f1

LETTERS = "ABCDEF"


def load_mcq_options(path: str | Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["item_id"]] = d
    return out


def make_mcq_scorer(options: list[str], correct_index: int):
    letters = LETTERS[: len(options)]
    letter_re = re.compile(rf"(?:^|[^a-z0-9])([{letters}{letters.lower()}])(?:[^a-z0-9]|$)")

    def scorer(question: str, gt: str, pred: str) -> float:
        p = (pred or "").strip()
        if not p:
            return 0.0
        m = letter_re.search(p if len(p) <= 40 else p[:40])
        if m:
            return 1.0 if m.group(1).upper() == letters[correct_index] else 0.0
        sims = [token_f1(o, p) for o in options]
        best = max(range(len(options)), key=lambda i: sims[i])
        if sims[best] >= 0.5 and sims[best] > max(
                (s for i, s in enumerate(sims) if i != best), default=0.0):
            return 1.0 if best == correct_index else 0.0
        pn = normalize_text(p)
        hits = [i for i, o in enumerate(options) if normalize_text(o) and normalize_text(o) in pn]
        if len(hits) == 1:
            return 1.0 if hits[0] == correct_index else 0.0
        return 0.0

    return scorer


def load_mcq_scorers(path: str | Path) -> dict[str, object]:
    scorers = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            k = json.loads(line)
            options = list(k["distractors"])
            options.insert(k["correct_index"], k["answer_text"])
            scorers[k["item_id"]] = make_mcq_scorer(options, k["correct_index"])
    return scorers
