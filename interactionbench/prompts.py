"""Prompts of the polling and offline protocols."""

from __future__ import annotations

from .data import BenchItem
from .mcq import LETTERS

SYSTEM = (
    "You are a real-time visual assistant watching a LIVE video stream. "
    "You receive frames as they arrive; the LAST image is the current moment. "
    "You must decide, right now, whether to speak or to keep watching silently. "
    "Only speak when this exact moment calls for it — do not speak early, and do "
    "not repeat what you already reported. If nothing needs saying, wait silently."
)

OFFLINE_SYSTEM = "You ground your answers precisely in video time."

FORMAT = (
    "Respond in EXACTLY this format, nothing else:\n"
    "DECISION: SPEAK or WAIT\n"
    "RESPONSE: <if SPEAK, one short line — the exact thing you would say out loud now; "
    "if WAIT, leave blank>"
)

INTERLEAVED_SUFFIX = (
    "\nYou can see your own previous turns. If you already reported "
    "this and nothing new happened, WAIT instead of repeating."
)

DEFAULT_HINT = "Speak only when it is the right moment."

CAPABILITY_HINTS: dict[str, dict[str, str]] = {
    "default": {
        "PTR": ("Speak only at the exact moment the requested event happens. If it never "
                "happens, stay silent for the whole stream."),
        "TOA": ("Answer at the earliest moment the question becomes decidable from what "
                "you have seen; before that, wait."),
        "CST": ("Track the requested state across frames. Each time it changes (e.g. the "
                "count increases), speak the update (for counts: just the new number). "
                "Otherwise wait."),
        "BRC": ("Track the requested state; speak whenever the correct answer changes, "
                "correcting your earlier answer if new evidence contradicts it."),
        "LCG": ("Follow the stream and speak one short line for each new step or scene "
                "as it happens, like a live commentator or guide."),
        "IVQA": "Answer the question as soon as you are asked, from what you can see.",
        "CIR": "Answer the question as soon as it becomes answerable from what you saw.",
        "LVM": "Answer from what you saw earlier in the stream. Answer immediately.",
    },
    "v2": {
        "PTR": ("Say something only when the event you were asked about actually occurs. "
                "If it never occurs, remain silent until the stream ends."),
        "TOA": ("Reply the moment the answer first becomes clear from the video so far; "
                "hold off until then."),
        "CST": ("Keep watching the state you were asked to track. Whenever it changes "
                "(for example the count goes up), announce the update (counts: only the "
                "new number). Stay quiet in between."),
        "BRC": ("Keep the tracked state in mind; whenever the correct answer becomes "
                "different, say the new answer, revising what you said before."),
        "LCG": ("Give a brief line of narration whenever a new step or scene appears, "
                "the way a live guide would."),
        "IVQA": "Reply right away when the question arrives, based on what is visible.",
        "CIR": "Reply as soon as the video has shown enough to answer.",
        "LVM": "Recall what appeared earlier in the stream and reply at once.",
    },
    "v3": {
        "PTR": ("Only break silence at the precise moment the requested event takes "
                "place; a stream without that event should get no response at all."),
        "TOA": ("Wait until the earliest point where what you have watched settles the "
                "question, then answer."),
        "CST": ("Monitor the quantity or state in question. On every change, state the "
                "update (just the number when counting); say nothing otherwise."),
        "BRC": ("Whenever new footage makes a different answer correct, speak up with "
                "the correction to your previous answer."),
        "LCG": ("Provide one short spoken line per new step or scene, in the style of "
                "real-time commentary."),
        "IVQA": "As soon as you are asked, answer using what the video has shown.",
        "CIR": "The moment the answer becomes determinable, give it.",
        "LVM": "Use your memory of earlier footage to answer without delay.",
    },
}

STANDING_TEMPLATE = """\
Current stream time: {t:.1f}s.

User's standing request (given at the start of the stream): "{question}"

{hint}

{fmt}"""

REVEAL_TEMPLATE = """\
Current stream time: {t:.1f}s. The user JUST asked:

"{question}"

{hint}

{fmt}"""

INTERLEAVED_TURN = "[stream time {t:.1f}s] New frames above. Decide now (DECISION / RESPONSE)."

OFFLINE_TEMPLATE = """\
You are watching a recorded video ({dur:.0f} seconds long, {n} frames sampled \
uniformly from 0s to {dur:.0f}s are shown in order).

User's request: "{question}"

{hint}

Report every moment where you would have responded, as if you had watched it \
live. Output one line per response, EXACTLY in this format (time in seconds):
[t=12.3] <the exact thing you would say at that moment>

Rules: use the video timeline for t; list moments in increasing time order; \
output NOTHING except these lines. If you would never respond, output exactly: \
NO_RESPONSE"""

OFFLINE_REVEAL_TEMPLATE = """\
You are watching a recorded video ({dur:.0f} seconds long, {n} frames sampled \
uniformly from 0s to {dur:.0f}s are shown in order).

At the end of the video the user asks: "{question}"

Answer the question, and also report the video time where the evidence appears. \
Output EXACTLY one line in this format (time in seconds):
[t=12.3] <your answer>"""


def hint_for(capability: str, hint_set: str = "default") -> str:
    if hint_set not in CAPABILITY_HINTS:
        raise KeyError(f"unknown hint set '{hint_set}'. known: {sorted(CAPABILITY_HINTS)}")
    hints = dict(CAPABILITY_HINTS["default"])
    hints.update(CAPABILITY_HINTS[hint_set])
    return hints.get(capability, DEFAULT_HINT)


def format_question(item: BenchItem, options) -> str:
    if not options:
        return item.question
    if isinstance(options, dict):
        stem = options.get("stem") or item.question
        opt_list = options["options"]
    else:
        stem, opt_list = item.question, options
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opt_list))
    return (f"{stem}\n{opts}\n"
            f"(When you speak, answer with just the option letter.)")
