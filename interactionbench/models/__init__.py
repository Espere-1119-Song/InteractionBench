"""Systems under test: adapters, the model zoo, and ``build_model``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from ..registry import Registry, import_object
from .base import ChatModel, Generation, TurnBasedVLM, collect_images  # noqa: F401
from .zoo import ALIASES, MODEL_ZOO

ADAPTERS: Registry[Callable[..., ChatModel]] = Registry("model adapter")


def _hf_vlm(**cfg):
    from .hf_vlm import HFChatVLM
    return HFChatVLM(**cfg)


def _hf_text(**cfg):
    from .hf_text import HFChatLM
    return HFChatLM(**cfg)


def _minicpmo(**cfg):
    from .minicpmo import MiniCPMOChat
    return MiniCPMOChat(**cfg)


def _api(**cfg):
    from .api import APIChatModel
    return APIChatModel(**cfg)


ADAPTERS.register("hf-vlm", _hf_vlm)
ADAPTERS.register("hf", _hf_vlm)
ADAPTERS.register("hf-text", _hf_text)
ADAPTERS.register("minicpmo", _minicpmo)
ADAPTERS.register("api", _api)

_LOCAL_ADAPTERS = {"hf-vlm", "hf", "hf-text", "minicpmo"}
_SPEC_TARGET = {"api": "model"}


def register_adapter(name: str, factory: Callable[..., ChatModel] | None = None, *,
                     overwrite: bool = False):
    return ADAPTERS.register(name, factory, overwrite=overwrite)


def register_model(name: str, config: dict, *, overwrite: bool = False) -> None:
    if name in MODEL_ZOO and not overwrite:
        raise ValueError(f"model '{name}' is already registered")
    if "adapter" not in config:
        raise ValueError("a model config needs an 'adapter' key")
    MODEL_ZOO[name] = dict(config)


def load_model_configs(path: str | Path, *, overwrite: bool = True) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith((".yaml", ".yml")):
        import yaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    for name, cfg in data.items():
        register_model(name, cfg, overwrite=overwrite)
    return list(data)


def list_models() -> dict[str, dict]:
    return {k: dict(v) for k, v in sorted(MODEL_ZOO.items())}


def resolve_name(name: str) -> str:
    return ALIASES.get(name, name)


def _relax_cudnn_attention() -> None:
    try:
        import torch
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:
        pass


def resolve_config(spec: str, model_path: str | None = None, **overrides) -> dict:
    key = resolve_name(spec)
    if key in MODEL_ZOO:
        cfg = dict(MODEL_ZOO[key])
        cfg.setdefault("short_name", key)
    else:
        prefix, sep, value = spec.partition(":")
        if sep and prefix in ADAPTERS:
            cfg = {"adapter": prefix, _SPEC_TARGET.get(prefix, "repo"): value}
        elif sep:
            cfg = {"adapter_path": spec}
        else:
            raise ValueError(
                f"unknown model '{spec}'. Use a zoo name {sorted(MODEL_ZOO)}, "
                f"'<adapter>:<id>' with an adapter from {ADAPTERS.names()}, "
                f"or 'module:Class'.")
    if model_path:
        cfg["repo"] = model_path
    cfg.update(overrides)
    if cfg.get("adapter") == "api":
        cfg["model"] = cfg.get("model") or os.environ.get("IBENCH_API_MODEL")
        if not cfg["model"]:
            raise ValueError("API model name missing: use 'api:<model>', "
                             "--model-arg model=<name>, or set IBENCH_API_MODEL")
        cfg["base_url"] = (cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL")
                           or "https://api.openai.com/v1")
    return cfg


def build_model(spec: str, model_path: str | None = None, **overrides) -> ChatModel:
    cfg = resolve_config(spec, model_path=model_path, **overrides)
    path = cfg.pop("adapter_path", None)
    if path is not None:
        cls = import_object(path)
        return cls(**cfg)
    adapter = cfg.pop("adapter")
    if adapter in _LOCAL_ADAPTERS:
        _relax_cudnn_attention()
    return ADAPTERS.get(adapter)(**cfg)
