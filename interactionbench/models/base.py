"""Interface every system under test implements.

All adapters speak one canonical message format, so a protocol is model-agnostic and
supports both single-turn modes (a fresh one-turn conversation per decision step) and
the interleaved mode (one growing multi-turn conversation that carries earlier frames
and the model's own earlier responses).

Canonical message format (what a protocol builds):

    [
      {"role": "system", "content": "<str>"},
      {"role": "user",   "content": [ {"type": "image", "image": <PIL.Image>},
                                      {"type": "text",  "text": "<str>"} ]},
      {"role": "assistant", "content": "<str>"},     # the model's earlier raw output
      ...
    ]

To add a system, subclass :class:`ChatModel` and implement :meth:`chat`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Pillow is only needed once frames are decoded
    from PIL import Image


@dataclass
class Generation:
    text: str
    latency_s: float        # wall-clock generation latency (reported, never scored)
    n_images: int           # images fed in this call


def collect_images(messages: list[dict]) -> "list[Image.Image]":
    """All images across the conversation, in order."""
    imgs = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for item in c:
                if item.get("type") == "image" and item.get("image") is not None:
                    imgs.append(item["image"])
    return imgs


class ChatModel:
    """A system that maps a canonical conversation to one text reply."""

    name: str = "base"

    def chat(self, messages: list[dict], max_new_tokens: int = 96) -> str:
        raise NotImplementedError

    def timed_chat(self, messages: list[dict], max_new_tokens: int = 96) -> Generation:
        n = len(collect_images(messages))
        t0 = time.perf_counter()
        text = self.chat(messages, max_new_tokens=max_new_tokens)
        return Generation(text=text, latency_s=time.perf_counter() - t0, n_images=n)


# backwards-compatible name
TurnBasedVLM = ChatModel
