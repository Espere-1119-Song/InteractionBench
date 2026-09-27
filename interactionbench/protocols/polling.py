"""Fixed-interval polling: make a turn-based model act as a real-time system.

At each tick the protocol exposes the visual context available "as of now" and asks
SPEAK or WAIT. The poll interval is the best timing resolution the system can reach.

Tick schedule
  time_type A    The question is NOT revealed before ``question_time_s`` (revealing it
                 earlier would leak a retrospective-memory task). The first poll is at
                 the reveal moment, then every ``interval`` seconds for ``a_window``
                 seconds or until the video ends.
  time_type B/C  The question is a standing request from t=0. Poll every ``interval``
                 seconds until the video ends.

Context regimes
  sliding      the most recent ``max_frames`` frames up to t. Each poll is an
               independent conversation. Constant cost; nothing older than the window
               is visible.
  cumulative   all frames 0..t, evenly subsampled to ``max_frames``. Each poll is
               independent; old detail thins out as the stream grows.
  interleaved  one growing conversation. Each poll appends the new frames since the
               previous poll plus a short nudge, and the model's reply stays in the
               context, so the model sees what it already said. Images in context are
               capped at ``max_frames`` by dropping the oldest, keeping all text turns.
"""

from __future__ import annotations

from ..data import BenchItem
from ..frames import Frame, frames_up_to, subsample
from ..models.base import ChatModel, collect_images
from ..parsing import parse_decision
from ..prompts import (FORMAT, INTERLEAVED_SUFFIX, INTERLEAVED_TURN, REVEAL_TEMPLATE,
                       STANDING_TEMPLATE, SYSTEM, hint_for)
from .base import Protocol, ProtocolConfig, make_poll, user_turn


def poll_ticks(item: BenchItem, interval: float, a_window: float) -> list[float]:
    """Decision times for one item."""
    ticks: list[float] = []
    if item.time_type == "A":
        q_t = item.question_time_s
        t_end = min(item.duration_s, q_t + a_window)
        t = q_t
        while t <= t_end + 1e-6:
            ticks.append(t)
            t += interval
    else:
        t = interval
        while t <= item.duration_s + 1e-6:
            ticks.append(t)
            t += interval
    return ticks


def trim_context_images(messages: list[dict], cap: int) -> None:
    """Drop the oldest image items in place so that at most ``cap`` remain."""
    total = len(collect_images(messages))
    if total <= cap:
        return
    to_drop = total - cap
    for m in messages:
        if to_drop <= 0:
            break
        c = m.get("content")
        if not isinstance(c, list):
            continue
        kept = []
        for item in c:
            if to_drop > 0 and item.get("type") == "image":
                to_drop -= 1
                continue
            kept.append(item)
        m["content"] = kept


class PollingProtocol(Protocol):
    """Shared tick loop. Subclasses choose the context regime through ``mode``."""

    mode = "sliding"

    def run_item(self, model: ChatModel, frames: list[Frame], item: BenchItem,
                 question: str, cfg: ProtocolConfig) -> list[dict]:
        is_reveal = item.time_type == "A"
        q_t = item.question_time_s if is_reveal else 0.0
        template = REVEAL_TEMPLATE if is_reveal else STANDING_TEMPLATE
        hint = hint_for(item.capability, cfg.hint_set)

        convo: list[dict] = []
        if self.mode == "interleaved":
            convo = [{"role": "system", "content":
                      SYSTEM + "\n\n" + template.format(t=q_t, question=question,
                                                        hint=hint, fmt=FORMAT)
                      + INTERLEAVED_SUFFIX}]

        polls = []
        prev_t = 0.0 if not is_reveal else max(0.0, q_t - 1e-6)
        for t in poll_ticks(item, cfg.interval, cfg.a_window):
            history = frames_up_to(frames, t) or frames[:1]
            if cfg.blind:
                history = []
            if self.mode == "cumulative":
                window = subsample(history, cfg.max_frames)
                messages = [{"role": "system", "content": SYSTEM},
                            user_turn(window, template.format(
                                t=t, question=question, hint=hint, fmt=FORMAT))]
            elif self.mode == "sliding":
                window = history[-cfg.max_frames:]
                messages = [{"role": "system", "content": SYSTEM},
                            user_turn(window, template.format(
                                t=t, question=question, hint=hint, fmt=FORMAT))]
            else:  # interleaved
                new = [f for f in history if prev_t < f.time <= t] or history[-1:]
                new = subsample(new, cfg.max_new_per_turn)
                convo.append(user_turn(new, INTERLEAVED_TURN.format(t=t)))
                trim_context_images(convo, cfg.max_frames)
                messages = convo

            gen = model.timed_chat(messages, max_new_tokens=cfg.max_new_tokens)
            spoke, response = parse_decision(gen.text)
            if self.mode == "interleaved":
                convo.append({"role": "assistant", "content": gen.text})
            polls.append(make_poll(t, spoke, response, gen))
            if cfg.verbose:
                tag = f"SPEAK: {response}" if spoke else "wait"
                print(f"    t={t:7.2f}s [{gen.n_images:2d} frm {gen.latency_s:5.2f}s] {tag}",
                      flush=True)
            prev_t = t
        return polls


class SlidingProtocol(PollingProtocol):
    name = mode = "sliding"


class CumulativeProtocol(PollingProtocol):
    name = mode = "cumulative"


class InterleavedProtocol(PollingProtocol):
    name = mode = "interleaved"
