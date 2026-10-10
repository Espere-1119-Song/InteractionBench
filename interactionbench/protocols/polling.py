"""Fixed-interval polling: make a turn-based model act as a real-time system."""

from __future__ import annotations

from ..data import BenchItem
from ..frames import Frame, frames_up_to, subsample
from ..models.base import ChatModel, collect_images
from ..parsing import parse_decision
from ..prompts import (FORMAT, INTERLEAVED_SUFFIX, INTERLEAVED_TURN, REVEAL_TEMPLATE,
                       STANDING_TEMPLATE, SYSTEM, hint_for)
from .base import Protocol, ProtocolConfig, make_poll, user_turn


def poll_ticks(item: BenchItem, interval: float, a_window: float) -> list[float]:
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
            else:
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
