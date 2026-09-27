#!/usr/bin/env python3
"""OpenAI-compatible tool-calling agent, used as the "openai" harness.

What it does
  Command line replacement for `claude -p`: the prompt is argv[1] and the final answer
  is printed on stdout. The script runs a function-calling loop against an
  OpenAI-compatible chat-completions endpoint. The functions are the tools of the
  Qwen-MM-Plugins "core" MCP server (frame extraction with ffmpeg, no GPU), which the
  script starts through `uvx` and calls over stdio. Images returned by a tool are sent
  to the model in a following user message.
  Tools offered to the model: the server tools whose name is in ALLOWED. A name in
  ALLOWED that the server does not provide is ignored.
  The loop ends when the model replies without a tool call, or after OAI_MAX_TURNS
  model calls (then an empty line is printed). Each model call has
  max_completion_tokens = 1500. HTTP errors 429, 500, 502 and 503 are retried up to 5
  times with waits of 8, 16, 32, 64 and 128 seconds.

Upstream code
  Qwen-MM-Plugins, https://github.com/QwenLM/Qwen-MM-Plugins (capability "core").

Environment
  Python 3.10 or later (standard library only), uv (for uvx), ffmpeg.
  OAI_MODEL             model name on the endpoint (required)
  OAI_BASE_URL          default http://127.0.0.1:8199/v1 (agents/cost_proxy.py)
  OAI_API_KEY           default "proxy" (the real key is held by the cost proxy; set
                        this variable only when calling an endpoint directly)
  OAI_MAX_TURNS         default 12
  QWEN_MM_PLUGINS_ROOT  path of the Qwen-MM-Plugins checkout (required)

Command
  The runners call this script; it is not started by hand for the paper runs:
    python agents/run_polling.py --harness openai ...
  Stand-alone use:
    python agents/openai_agent.py "<prompt>"

Output
  The final answer on stdout.
  Exit codes: 0 success; 1 API error; 3 spending limit reached at the cost proxy
  (stdout is empty, so the runner records no answer for the item).
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ALLOWED = ("read_video", "read_image", "media_info", "crop_image", "save_view")


class MCP:
    """Minimal stdio JSON-RPC client for the qwen-mm MCP server."""

    def __init__(self, root):
        self.p = subprocess.Popen(
            ["uvx", "--from", f"{root}[core]", "qwen-mm-plugins-core"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._id = 0
        self.rpc("initialize", {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "oai-agent", "version": "1.0"}})
        self.p.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self.p.stdin.flush()
        self.tools = self.rpc("tools/list", {})["tools"]

    def rpc(self, method, params):
        self._id += 1
        self.p.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "id": self._id,
             "method": method, "params": params}) + "\n")
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP server died during {method}")
            msg = json.loads(line)
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"MCP {method}: {msg['error']}")
                return msg["result"]

    def call(self, name, args):
        r = self.rpc("tools/call", {"name": name, "arguments": args})
        texts, images = [], []
        for c in r.get("content", []):
            if c.get("type") == "text":
                texts.append(c["text"])
            elif c.get("type") == "image":
                images.append({"type": "image_url", "image_url": {
                    "url": f"data:{c.get('mimeType', 'image/jpeg')};"
                           f"base64,{c['data']}"}})
        return "\n".join(texts) or f"({len(images)} frame(s) attached)", images

    def close(self):
        try:
            self.p.terminate()
        except OSError:
            pass


def api(base, key, payload):
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            if "BUDGET EXHAUSTED" in body:
                sys.exit(3)
            if e.code not in (429, 500, 502, 503) or attempt == 5:
                print(f"api error {e.code}: {body}", file=sys.stderr)
                sys.exit(1)
            time.sleep(8 * (2 ** attempt))


def main():
    if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    if len(sys.argv) < 2:
        sys.exit("usage: openai_agent.py '<prompt>'")
    prompt = sys.argv[1]
    model = os.environ.get("OAI_MODEL") or sys.exit("set OAI_MODEL")
    base = os.environ.get("OAI_BASE_URL", "http://127.0.0.1:8199/v1")
    key = os.environ.get("OAI_API_KEY", "proxy")
    root = os.environ.get("QWEN_MM_PLUGINS_ROOT") \
        or sys.exit("set QWEN_MM_PLUGINS_ROOT")
    mcp = MCP(root)
    tools = [{"type": "function", "function": {
        "name": t["name"], "description": t.get("description", "")[:1000],
        "parameters": t.get("inputSchema", {"type": "object"})}}
        for t in mcp.tools if t["name"] in ALLOWED]
    msgs = [{"role": "user", "content": prompt}]
    try:
        for _ in range(int(os.environ.get("OAI_MAX_TURNS", "12"))):
            d = api(base, key, {"model": model, "messages": msgs,
                                "tools": tools, "max_completion_tokens": 1500})
            m = d["choices"][0]["message"]
            calls = m.get("tool_calls") or []
            if not calls:
                print((m.get("content") or "").strip())
                return
            msgs.append({"role": "assistant", "content": m.get("content"),
                         "tool_calls": calls})
            frames = []
            for c in calls:
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                    text, images = mcp.call(c["function"]["name"], args)
                except Exception as e:  # noqa: BLE001 — tool fault -> tell model
                    text, images = f"tool error: {e}", []
                msgs.append({"role": "tool", "tool_call_id": c["id"],
                             "content": text})
                frames.extend(images)
            if frames:
                msgs.append({"role": "user", "content":
                             [{"type": "text", "text":
                               "Frames returned by the tool call(s) above:"}]
                             + frames})
        print("")  # turn cap: no final answer
    finally:
        mcp.close()


if __name__ == "__main__":
    main()
