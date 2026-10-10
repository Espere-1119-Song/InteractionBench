"""Protocol interface: how a system is driven over one item."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data import BenchItem
from ..frames import Frame
from ..models.base import ChatModel


@dataclass
class ProtocolConfig:
    interval: float = 1.0
    a_window: float = 10.0
    max_frames: int = 16
    max_new_per_turn: int = 8
    max_new_tokens: int = 96
    hint_set: str = "default"
    blind: bool = False
    verbose: bool = False
    extra: dict = field(default_factory=dict)


class Protocol:
    name: str = "base"
    tag: str = "{name}_iv{interval:g}"
    needs_frames: bool = True

    def run_item(self, model: ChatModel, frames: list[Frame], item: BenchItem,
                 question: str, cfg: ProtocolConfig) -> list[dict]:
        raise NotImplementedError

    def run_tag(self, cfg: ProtocolConfig) -> str:
        return self.tag.format(name=self.name, interval=cfg.interval)


def user_turn(frames: list[Frame], text: str) -> dict:
    content = [{"type": "image", "image": f.image} for f in frames]
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def make_poll(t: float, spoke: bool, response, gen) -> dict:
    return {"t": round(t, 3), "spoke": spoke, "response": response,
            "latency_s": round(gen.latency_s, 3), "n_images": gen.n_images,
            "raw": gen.text}
