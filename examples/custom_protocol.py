"""Example plugin: add a test method.

Load it with ``--plugin examples/custom_protocol.py`` and select it with
``--protocol single-look`` or ``--protocol sparse-poll``.

A protocol decides what the system sees and when it is asked. ``run_item`` returns one
poll per decision step; the runner converts the polls that spoke into emissions and the
evaluator scores them exactly like any other run.
"""

from interactionbench import Protocol, register_protocol
from interactionbench.frames import frames_up_to
from interactionbench.parsing import parse_decision
from interactionbench.prompts import (FORMAT, REVEAL_TEMPLATE, STANDING_TEMPLATE, SYSTEM,
                                      hint_for)
from interactionbench.protocols import make_poll, poll_ticks, user_turn


@register_protocol("single-look")
class SingleLookProtocol(Protocol):
    """At every tick the system sees only the current frame. No history at all."""

    name = "single-look"

    def run_item(self, model, frames, item, question, cfg):
        template = REVEAL_TEMPLATE if item.time_type == "A" else STANDING_TEMPLATE
        hint = hint_for(item.capability, cfg.hint_set)
        polls = []
        for t in poll_ticks(item, cfg.interval, cfg.a_window):
            visible = [] if cfg.blind else (frames_up_to(frames, t) or frames[:1])[-1:]
            messages = [
                {"role": "system", "content": SYSTEM},
                user_turn(visible, template.format(t=t, question=question, hint=hint,
                                                   fmt=FORMAT)),
            ]
            gen = model.timed_chat(messages, max_new_tokens=cfg.max_new_tokens)
            spoke, response = parse_decision(gen.text)
            polls.append(make_poll(t, spoke, response, gen))
        return polls


@register_protocol("sparse-poll")
class SparsePollProtocol(Protocol):
    """Sliding window, but the system is asked only every ``every`` ticks.

    Custom options arrive through ``--protocol-arg``, e.g. ``--protocol-arg every=5``."""

    name = "sparse-poll"

    def run_item(self, model, frames, item, question, cfg):
        every = int(cfg.extra.get("every", 5))
        template = REVEAL_TEMPLATE if item.time_type == "A" else STANDING_TEMPLATE
        hint = hint_for(item.capability, cfg.hint_set)
        polls = []
        for i, t in enumerate(poll_ticks(item, cfg.interval, cfg.a_window)):
            if i % every:
                continue
            window = (frames_up_to(frames, t) or frames[:1])[-cfg.max_frames:]
            messages = [
                {"role": "system", "content": SYSTEM},
                user_turn([] if cfg.blind else window,
                          template.format(t=t, question=question, hint=hint, fmt=FORMAT)),
            ]
            gen = model.timed_chat(messages, max_new_tokens=cfg.max_new_tokens)
            spoke, response = parse_decision(gen.text)
            polls.append(make_poll(t, spoke, response, gen))
        return polls
