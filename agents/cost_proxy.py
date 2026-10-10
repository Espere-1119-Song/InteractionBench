#!/usr/bin/env python3
"""Local metering proxy with a spending limit, for the "openai" harness."""
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
            pass
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
