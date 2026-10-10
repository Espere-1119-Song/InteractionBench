"""Interface every system under test implements."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class Generation:
    text: str
    latency_s: float
    n_images: int


def collect_images(messages: list[dict]) -> "list[Image.Image]":
    imgs = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for item in c:
                if item.get("type") == "image" and item.get("image") is not None:
                    imgs.append(item["image"])
    return imgs


class ChatModel:
    name: str = "base"

    def chat(self, messages: list[dict], max_new_tokens: int = 96) -> str:
        raise NotImplementedError

    def timed_chat(self, messages: list[dict], max_new_tokens: int = 96) -> Generation:
        n = len(collect_images(messages))
        t0 = time.perf_counter()
        text = self.chat(messages, max_new_tokens=max_new_tokens)
        return Generation(text=text, latency_s=time.perf_counter() - t0, n_images=n)


TurnBasedVLM = ChatModel
