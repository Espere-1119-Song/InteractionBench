"""Example plugin: evaluate your own system.

Load it with ``--plugin examples/custom_model.py``, then select the system with
``--model always-wait``, ``--model first-frame-talker`` or ``--model scripted:<policy>``.

The contract is one method. ``chat`` receives the canonical conversation

    [{"role": "system", "content": "<str>"},
     {"role": "user", "content": [{"type": "image", "image": <PIL.Image>}, ...,
                                  {"type": "text", "text": "<str>"}]},
     {"role": "assistant", "content": "<str>"}, ...]

and returns the raw reply text. Under the polling protocols the reply is parsed as

    DECISION: SPEAK or WAIT
    RESPONSE: <one line>

so a system that wraps a real model only has to convert the messages to its own input
format, generate, and return the text.
"""

from interactionbench import ChatModel, register_adapter, register_model
from interactionbench.models import collect_images


@register_adapter("scripted")
class ScriptedModel(ChatModel):
    """A model-free policy. Useful as a smoke test and as a metric sanity check.

    ``repo`` receives the text after ``scripted:`` in the model spec."""

    def __init__(self, repo: str = "wait", short_name: str | None = None, **_unused):
        self.policy = repo
        self.name = short_name or f"scripted:{repo}"

    def chat(self, messages, max_new_tokens=96):
        n_images = len(collect_images(messages))
        if self.policy == "wait":
            return "DECISION: WAIT\nRESPONSE:"
        if self.policy == "first-frame":
            # speaks while only one or two frames are visible, then stays silent
            if n_images <= 2:
                return "DECISION: SPEAK\nRESPONSE: A"
            return "DECISION: WAIT\nRESPONSE:"
        if self.policy == "always":
            return "DECISION: SPEAK\nRESPONSE: Something is happening now."
        raise ValueError(f"unknown policy {self.policy!r}")


# Named configurations, selectable with --model <name>
register_model("always-wait", {"adapter": "scripted", "repo": "wait"})
register_model("first-frame-talker", {"adapter": "scripted", "repo": "first-frame"})


# ---------------------------------------------------------------------------
# Template for a real model. Copy, fill in the three marked places, register.
# ---------------------------------------------------------------------------
class MyModelTemplate(ChatModel):
    def __init__(self, repo: str, short_name: str | None = None, **kwargs):
        self.name = short_name or repo
        # 1. load weights / open a client here
        # self.model = ...

    def chat(self, messages, max_new_tokens=96):
        # 2. convert `messages` to the input format of your model
        # 3. generate and return the reply text
        raise NotImplementedError
