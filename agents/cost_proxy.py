#!/usr/bin/env python3
"""Local metering proxy with a spending limit, for the "openai" harness.

What it does
  agents/openai_agent.py sends its chat-completions requests to this proxy. The proxy
  forwards each request to the real endpoint, computes the price of the response from
  its usage block, and appends one row to a ledger file. When the total in the ledger
  has reached the limit, the proxy answers every further request with HTTP 429 and a
  BUDGET EXHAUSTED message; openai_agent.py then exits with code 3 and the run stops
  producing answers. Start one proxy per run (own --port and --line label). Proxies
  that share one ledger file share one limit.
  The total is read from the ledger before a request is forwarded, so requests that
  are already in progress when the limit is reached are still completed and charged.
  The proxy listens on 127.0.0.1 only. The API key is held by the proxy; the runners
  send a placeholder key.

Upstream code
  None. Works with any OpenAI-compatible chat-completions endpoint that returns a
  usage block (prompt_tokens, completion_tokens, prompt_tokens_details.cached_tokens).

Environment (all required; standard library only)
  UPSTREAM_BASE_URL   real endpoint, e.g. https://api.openai.com/v1
  UPSTREAM_API_KEY    real key, read from the environment only
  GPT_PRICE_IN        price of uncached input tokens, USD per 1M tokens
  GPT_PRICE_CACHED    price of cached input tokens, USD per 1M tokens
  GPT_PRICE_OUT       price of output tokens, USD per 1M tokens
  GPT_BUDGET_USD      spending limit over all runs that share the ledger
  The three prices are list prices of the model. No price is stored in this file:
  take the current values from the pricing page of the provider and update them when
  the model or the price list changes. The ledger is only as accurate as these values.

Command
  python agents/cost_proxy.py --port 8199 --line agent-poll [--ledger PATH]
  Spend so far:  curl -s localhost:8199/spend | python -m json.tool

Output
  Ledger, default results/runs/cost_ledger.jsonl, one row per response:
  {"t", "line", "prompt", "cached", "completion", "usd"}.
"""
import argparse
import fcntl
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CFG = {}


def spent_total(ledger):
    tot, per = 0.0, {}
    try:
        with open(ledger) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    tot += r["usd"]
                    per[r["line"]] = per.get(r["line"], 0.0) + r["usd"]
    except FileNotFoundError:
        pass
    return tot, per


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        tot, per = spent_total(CFG["ledger"])
        self._send(200, {"spent_usd": round(tot, 4), "budget_usd": CFG["budget"],
                         "remaining_usd": round(CFG["budget"] - tot, 4),
                         "by_line": {k: round(v, 4) for k, v in per.items()}})

    def do_POST(self):
        tot, _ = spent_total(CFG["ledger"])
        if tot >= CFG["budget"]:
            return self._send(429, {"error": {"message":
                f"BUDGET EXHAUSTED: ${tot:.2f} >= ${CFG['budget']:.2f}. "
                f"Raise GPT_BUDGET_USD and restart the proxy to continue."}})
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        req = urllib.request.Request(
            CFG["upstream"].rstrip("/") + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {CFG['key']}"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
                code = r.status
        except urllib.error.HTTPError as e:
            return self._send(e.code, {"error": {"message": e.read().decode()[:400]}})
        except Exception as e:  # noqa: BLE001 — network faults become 502
            return self._send(502, {"error": {"message": str(e)[:200]}})
        try:
            u = json.loads(data).get("usage") or {}
            pt = u.get("prompt_tokens", 0)
            ct = u.get("completion_tokens", 0)
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            usd = ((pt - cached) * CFG["p_in"] + cached * CFG["p_cached"]
                   + ct * CFG["p_out"]) / 1e6
            row = {"t": round(time.time(), 1), "line": CFG["line"],
                   "prompt": pt, "cached": cached, "completion": ct,
                   "usd": round(usd, 6)}
            with open(CFG["ledger"], "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(json.dumps(row) + "\n")
        except (json.JSONDecodeError, KeyError):
            pass  # non-JSON upstream reply: forward it, meter nothing
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser(
        description="Local metering proxy with a spending limit.")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--line", required=True,
                    help="label of this run in the ledger, e.g. agent-poll")
    ap.add_argument("--ledger", default="results/runs/cost_ledger.jsonl")
    a = ap.parse_args()
    for k in ("UPSTREAM_BASE_URL", "UPSTREAM_API_KEY", "GPT_PRICE_IN",
              "GPT_PRICE_CACHED", "GPT_PRICE_OUT", "GPT_BUDGET_USD"):
        if not os.environ.get(k):
            sys.exit(f"missing env {k}")
    os.makedirs(os.path.dirname(a.ledger) or ".", exist_ok=True)
    CFG.update(upstream=os.environ["UPSTREAM_BASE_URL"],
               key=os.environ["UPSTREAM_API_KEY"],
               p_in=float(os.environ["GPT_PRICE_IN"]),
               p_cached=float(os.environ["GPT_PRICE_CACHED"]),
               p_out=float(os.environ["GPT_PRICE_OUT"]),
               budget=float(os.environ["GPT_BUDGET_USD"]),
               line=a.line, ledger=a.ledger)
    print(f"cost proxy :{a.port} line={a.line} budget=${CFG['budget']}"
          f" ledger={a.ledger}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
