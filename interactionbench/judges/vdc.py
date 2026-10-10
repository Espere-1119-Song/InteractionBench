"""Decomposition scoring for free-form text (VDCScore style, AuroraCap, arXiv 2410.03051)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_GEN_PROMPT = """You are building an evaluation for text produced about a video.
Reference text (ground truth for one video segment or one answer):
"{gt}"
Context question (may be empty): "{question}"

Write short question-answer pairs that together cover EVERY distinct fact in
the reference (1 pair if it is a single short fact, at most 5). Ask about the
facts themselves (what happened, what is shown, which object/action), not
about exact wording or proper names, so a faithful paraphrase can answer
them. Answers must be words from, or directly implied by, the reference.
Reply with ONLY a JSON array like [["q1","a1"],["q2","a2"]] and nothing else."""

_ANS_PROMPT = """Candidate text (what a model said about a video segment):
"{cand}"

Question: {q}
Answer using only information stated or clearly implied by the candidate
text (paraphrasing is fine). If the candidate text contains no relevant
information at all, reply exactly: UNANSWERABLE
Reply with the short answer only."""

_MATCH_PROMPT = """Question: {q}
Reference answer: {ref}
Candidate answer: {cand}
Does the candidate answer convey the same meaning as the reference answer?
Paraphrases, synonyms and descriptions of the same thing count as the same
meaning; numbers and yes/no polarity must match.
Reply with exactly one word: yes or no."""


def _h(*parts: str) -> str:
    m = hashlib.sha256()
    for p in parts:
        m.update((p or "").encode("utf-8"))
        m.update(b"\x00")
    return m.hexdigest()[:32]


class _JsonlCache:
    def __init__(self, path: Path):
        self.path = path
        self.d: dict[str, object] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.d[row["k"]] = row["v"]

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        self.d[k] = v
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


class VDCScorer:
    def __init__(self, engine, pairs_cache: str | None = None,
                 step_cache: str | None = None,
                 cache_dir: str = "results/judge_cache", **_ignored):
        self.engine = engine
        self.name = f"vdc:{engine.name}"
        slug = re.sub(r"[^A-Za-z0-9.-]+", "_", getattr(engine, "name", "judge"))
        self.pairs = _JsonlCache(Path(pairs_cache or f"{cache_dir}/vdc_pairs_{slug}.jsonl"))
        self.steps = _JsonlCache(Path(step_cache or f"{cache_dir}/vdc_steps_{slug}.jsonl"))
        self.n_calls = 0

    def __call__(self, question: str, gt: str, pred: str) -> float:
        return self.engine(question, gt, pred)

    def _gen(self, prompt: str) -> str:
        self.n_calls += 1
        return self.engine._generate(prompt)

    def _qa_pairs(self, question: str, gt: str) -> list[list[str]]:
        k = _h("pairs", question, gt)
        hit = self.pairs.get(k)
        if hit is not None:
            return hit
        raw = self._gen(_GEN_PROMPT.format(gt=gt, question=question or "(none)"))
        m = re.search(r"\[.*\]", raw or "", re.DOTALL)
        pairs = []
        if m:
            try:
                pairs = [[str(q), str(a)] for q, a in json.loads(m.group(0))
                         if str(q).strip() and str(a).strip()][:5]
            except (json.JSONDecodeError, TypeError, ValueError):
                pairs = []
        if not pairs:
            pairs = [[question or "What does the reference state?", gt]]
        self.pairs.put(k, pairs)
        return pairs

    def vdc(self, question: str, gt: str, pred: str) -> float:
        gt, pred = (gt or "").strip(), (pred or "").strip()
        if not gt and not pred:
            return 1.0
        if not gt or not pred:
            return 0.0
        hits = 0
        pairs = self._qa_pairs(question, gt)
        for q, ref_a in pairs:
            ak = _h("ans", q, pred)
            ans = self.steps.get(ak)
            if ans is None:
                ans = (self._gen(_ANS_PROMPT.format(cand=pred, q=q)) or "").strip()
                self.steps.put(ak, ans)
            if not ans or "UNANSWERABLE" in ans.upper():
                continue
            mk = _h("match", q, ref_a, ans)
            ok = self.steps.get(mk)
            if ok is None:
                verdict = (self._gen(_MATCH_PROMPT.format(q=q, ref=ref_a, cand=ans)) or "")
                ok = 1 if re.search(r"\byes\b", verdict.strip().lower()) else 0
                self.steps.put(mk, ok)
            hits += int(ok)
        return hits / len(pairs)
