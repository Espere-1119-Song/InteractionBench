"""Judge backed by an OpenAI-compatible ``/chat/completions`` endpoint (no SDK)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .base import DEFAULT_PROMPT_VERSION, CachedJudge


class APIJudge(CachedJudge):
    def __init__(self, model: str, base_url: str | None = None,
                 api_key_env: str = "IBENCH_JUDGE_API_KEY", cache_path=None,
                 prompt_version: str = DEFAULT_PROMPT_VERSION,
                 max_tokens: int = 2048, reasoning_effort: str | None = None,
                 timeout_s: float = 120.0, max_retries: int = 6, **_ignored):
        super().__init__(cache_path, prompt_version)
        self.name = f"api:{model}"
        self.model = model
        self.base_url = (base_url or os.environ.get("IBENCH_JUDGE_BASE_URL")
                         or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY", "")
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort or os.environ.get("IBENCH_JUDGE_REASONING_EFFORT")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def _generate(self, prompt: str) -> str:
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": self.max_tokens, "temperature": 0}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key or 'EMPTY'}"})
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    d = json.load(resp)
                return d["choices"][0]["message"].get("content") or ""
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503) or attempt == self.max_retries - 1:
                    raise
                time.sleep(10 * (2 ** attempt))
        return ""
