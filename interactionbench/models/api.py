"""OpenAI-compatible chat-completions adapter.

Frames go inline as base64 JPEG. There is no agent loop and no scaffolding, which
isolates the model's own ability under the protocol. Any provider with an
OpenAI-compatible ``/chat/completions`` endpoint works, including a local vLLM or
SGLang server, and the compatibility endpoints of Anthropic and Google.

No SDK is required; requests use the standard library.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

from .base import ChatModel

RETRY_STATUS = (429, 500, 502, 503, 529)


def force_ipv4() -> None:
    """Resolve hosts to IPv4 first. Some compute nodes blackhole IPv6, which makes
    every request wait for the connect timeout. Enabled by IBENCH_FORCE_IPV4=1."""
    import socket

    if getattr(socket.getaddrinfo, "_ibench_ipv4", False):
        return
    orig = socket.getaddrinfo

    def _ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
        res = orig(host, port, family, type, proto, flags)
        v4 = [r for r in res if r[0] == socket.AF_INET]
        return v4 or res

    _ipv4_first._ibench_ipv4 = True
    socket.getaddrinfo = _ipv4_first


def image_to_b64(img, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """Canonical messages -> OpenAI chat-completions messages."""
    conv = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            conv.append({"role": m["role"], "content": c})
            continue
        parts = []
        for item in c:
            if item.get("type") == "image" and item.get("image") is not None:
                parts.append({"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_b64(item['image'])}"}})
            elif item.get("type") == "text":
                parts.append({"type": "text", "text": item["text"]})
        conv.append({"role": m["role"], "content": parts})
    return conv


class APIChatModel(ChatModel):
    """
    Args:
        model: model name sent in the request body.
        base_url: endpoint root, e.g. ``https://api.openai.com/v1`` or
            ``http://127.0.0.1:8000/v1`` for a local server.
        api_key_env: name of the environment variable holding the key. The key is
            read from the environment only and is never written to disk.
        max_tokens_field: ``max_completion_tokens`` or ``max_tokens``.
        extra_body: extra fields merged into every request body.
        timeout_s / max_retries: transport settings.
    """

    def __init__(self, model: str, base_url: str, api_key_env: str = "OPENAI_API_KEY",
                 short_name: str | None = None,
                 max_tokens_field: str = "max_completion_tokens",
                 extra_body: dict | None = None,
                 timeout_s: float = 180.0, max_retries: int = 6,
                 allow_empty_key: bool = False, **_ignored):
        if os.environ.get("IBENCH_FORCE_IPV4") == "1":
            force_ipv4()
        self.extra_body = extra_body or {}
        self.model = model
        self.max_tokens_field = max_tokens_field
        self.base_url = base_url.rstrip("/")
        self.name = short_name or f"api:{model}"
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key and not allow_empty_key:
            raise RuntimeError(f"environment variable {api_key_env} is not set")

    def chat(self, messages: list[dict], max_new_tokens: int = 96) -> str:
        body = json.dumps({"model": self.model, "messages": to_openai_messages(messages),
                           self.max_tokens_field: max_new_tokens,
                           **self.extra_body}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key or 'EMPTY'}"})
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    d = json.load(r)
                ch = d.get("choices") or []
                if not ch or "message" not in ch[0]:
                    # Some providers return a choice without a message when a safety
                    # filter blocks the reply; treat it as silence.
                    print(f"[api] empty/blocked choice: {json.dumps(d)[:400]}",
                          file=sys.stderr, flush=True)
                    return ""
                return ch[0]["message"].get("content") or ""
            except urllib.error.HTTPError as e:
                if e.code not in RETRY_STATUS or attempt == self.max_retries - 1:
                    raise
                time.sleep(8 * (2 ** attempt))
        return ""


# backwards-compatible name
APIChatVLM = APIChatModel
