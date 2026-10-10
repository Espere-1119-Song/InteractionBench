#!/usr/bin/env python3
"""OpenAI-compatible tool-calling agent, used as the "openai" harness."""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ALLOWED = ("read_video", "read_image", "media_info", "crop_image", "save_view")


class MCP:
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
        print("")
    finally:
        mcp.close()


if __name__ == "__main__":
    main()
