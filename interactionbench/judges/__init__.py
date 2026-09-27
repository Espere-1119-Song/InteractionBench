"""Content judges and ``make_judge``.

Judge specs:

  hf:<model_id>                local Hugging Face model        e.g. hf:Qwen/Qwen3-14B
  api:<model>                  OpenAI-compatible endpoint      e.g. api:gpt-4o-mini
  cache:<file-or-glob>         replay stored verdicts, no model
  ensemble:<slug>,<slug>,...   majority vote over stored verdicts
  vdc:<judge spec>             decomposition scoring on top of any judge
  <registered name>[:<arg>]    a judge added with ``register_judge``
  <module>:<Class>             any class with ``__call__(question, gt, pred) -> float``
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..registry import Registry, import_object
from .base import (DEFAULT_PROMPT_VERSION, PROMPTS, CachedJudge,  # noqa: F401
                   cache_key, parse_score)

# judge kind -> factory(arg, cache_path=..., prompt_version=..., **kwargs)
JUDGES: Registry[Callable] = Registry("judge")


def _hf(arg, **kw):
    from .hf import HFJudge
    return HFJudge(arg or "Qwen/Qwen3-14B", **kw)


def _api(arg, **kw):
    from .api import APIJudge
    return APIJudge(arg, **kw)


def _cache(arg, **kw):
    from .cached import CacheOnlyJudge
    kw.pop("cache_path", None)
    return CacheOnlyJudge(arg.split(","), **kw)


def _ensemble(arg, **kw):
    from .cached import EnsembleJudge
    kw.pop("cache_path", None)
    return EnsembleJudge(arg.split(","), **kw)


def _vdc(arg, **kw):
    from .vdc import VDCScorer
    vdc_kw = {k: kw.pop(k) for k in ("pairs_cache", "step_cache", "cache_dir") if k in kw}
    return VDCScorer(make_judge(arg, **kw), **vdc_kw)


JUDGES.register("hf", _hf)
JUDGES.register("api", _api)
JUDGES.register("cache", _cache)
JUDGES.register("ensemble", _ensemble)
JUDGES.register("vdc", _vdc)


def register_judge(name: str, factory: Callable | None = None, *, overwrite: bool = False):
    """Register a judge factory ``factory(arg, **kwargs) -> judge``. Usable as a decorator.

    ``arg`` is the text after ``<name>:`` in the spec. The returned object must be
    callable as ``judge(question, gt, pred) -> float``."""
    return JUDGES.register(name, factory, overwrite=overwrite)


def list_judges() -> list[str]:
    return JUDGES.names()


def make_judge(spec: str, cache_path: str | Path | None = None,
               prompt_version: str = DEFAULT_PROMPT_VERSION, **kwargs):
    """Build a judge from a spec (see the module docstring)."""
    kind, _, arg = spec.partition(":")
    if kind in JUDGES:
        return JUDGES.get(kind)(arg, cache_path=cache_path,
                                prompt_version=prompt_version, **kwargs)
    if ":" in spec:  # module:Class
        cls = import_object(spec)
        return cls(cache_path=cache_path, prompt_version=prompt_version, **kwargs)
    raise ValueError(f"unknown judge spec {spec!r}. kinds: {JUDGES.names()} or 'module:Class'")
