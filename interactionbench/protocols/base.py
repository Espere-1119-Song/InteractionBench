"""Protocol interface: how a system is driven over one item.

A protocol decides what the system sees and when it is asked. It returns *polls*, one
dict per decision step:

    {"t": <stream seconds>, "spoke": bool, "response": str | None,
     "latency_s": float, "n_images": int, "raw": <raw model text>}

The runner turns the polls with ``spoke`` and a non-empty ``response`` into the
emissions that are scored. Everything else in a poll is kept for debugging.

To add a test method, subclass :class:`Protocol`, implement :meth:`run_item`, and
register it with ``register_protocol``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data import BenchItem
from ..frames import Frame
from ..models.base import ChatModel


@dataclass
class ProtocolConfig:
    interval: float = 1.0          # seconds of stream time between decision steps
    a_window: float = 10.0         # seconds of polling after an A-type question is revealed
    max_frames: int = 16           # frames visible per step (context cap when interleaved)
    max_new_per_turn: int = 8      # interleaved: new frames appended per step
    max_new_tokens: int = 96
    hint_set: str = "default"      # capability-hint paraphrase set
    blind: bool = False            # no frames at all: language-prior baseline
    verbose: bool = False
    extra: dict = field(default_factory=dict)   # free-form options for custom protocols


class Protocol:
    """Drives one model over one item."""

    name: str = "base"
    #: run tag suffix pattern; ``{interval}`` is available
    tag: str = "{name}_iv{interval:g}"
    #: False for protocols that need no decoded frames even when not blind
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
