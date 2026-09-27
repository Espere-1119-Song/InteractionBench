"""Example plugin: add a judge.

Load it with ``--plugin examples/custom_judge.py`` and select it with
``--judge exact`` or ``--judge my-llm:<model name>``.

A judge is any callable ``(question, gt, pred) -> float in [0, 1]``. Subclassing
``CachedJudge`` adds the disk cache, the grading prompt and verdict parsing; only
``_generate`` (prompt in, raw text out) is left to implement.
"""

from interactionbench import register_judge
from interactionbench.judges import CachedJudge
from interactionbench.metrics import normalize_text


@register_judge("exact")
def make_exact_judge(arg, **_unused):
    """Model-free judge: normalized exact match."""

    def judge(question, gt, pred):
        return 1.0 if normalize_text(gt) and normalize_text(gt) == normalize_text(pred) else 0.0

    judge.name = "exact"
    return judge


class MyLLMJudge(CachedJudge):
    """Template for a judge backed by your own model or service."""

    def __init__(self, model: str, cache_path=None, prompt_version="v2", **kwargs):
        super().__init__(cache_path, prompt_version)
        self.name = f"my-llm:{model}"
        # load a model or open a client here

    def _generate(self, prompt: str) -> str:
        # send `prompt`, return the raw reply; the reply must contain "SCORE: 1" or "SCORE: 0"
        raise NotImplementedError


@register_judge("my-llm")
def make_my_llm_judge(arg, **kwargs):
    return MyLLMJudge(arg, **kwargs)
