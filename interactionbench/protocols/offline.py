"""Offline temporal grounding: the non-real-time reference track.

The system sees the whole video in one call and lists every moment where it would have
responded. The claimed timestamps become emission times, so the same timing and silence
scores apply. The system has hindsight here, so these numbers are an upper reference
for the same model under polling and are not real-time results.
"""

from __future__ import annotations

from ..data import BenchItem
from ..frames import Frame, subsample
from ..models.base import ChatModel
from ..parsing import parse_offline
from ..prompts import OFFLINE_REVEAL_TEMPLATE, OFFLINE_SYSTEM, OFFLINE_TEMPLATE, hint_for
from .base import Protocol, ProtocolConfig, user_turn


class OfflineProtocol(Protocol):
    name = "offline"
    tag = "offline"

    def run_item(self, model: ChatModel, frames: list[Frame], item: BenchItem,
                 question: str, cfg: ProtocolConfig) -> list[dict]:
        window = subsample(frames, cfg.max_frames)
        hint = hint_for(item.capability, cfg.hint_set)
        tmpl = OFFLINE_REVEAL_TEMPLATE if item.time_type == "A" else OFFLINE_TEMPLATE
        prompt = tmpl.format(dur=item.duration_s, n=len(window), question=question, hint=hint)
        messages = [{"role": "system", "content": OFFLINE_SYSTEM}, user_turn(window, prompt)]
        gen = model.timed_chat(messages, max_new_tokens=cfg.max_new_tokens)
        emissions = parse_offline(gen.text, item)
        if cfg.verbose:
            print(f"    offline [{gen.n_images} frm {gen.latency_s:.2f}s] "
                  f"{len(emissions)} timed responses", flush=True)
        base = {"latency_s": round(gen.latency_s, 3), "n_images": gen.n_images, "raw": gen.text}
        if not emissions:
            return [{"t": 0.0, "spoke": False, "response": None, **base}]
        return [{"t": round(e["t"], 3), "spoke": True, "response": e["content"], **base}
                for e in emissions]
