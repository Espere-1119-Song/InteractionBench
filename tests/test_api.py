"""API model and API judge against a local stand-in server. No external request is made."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from interactionbench import build_model, make_judge


class _Handler(BaseHTTPRequestHandler):
    requests = []
    fail_first = 0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                    "body": body})
        if type(self).fail_first > 0:
            type(self).fail_first -= 1
            self.send_response(503)
            self.end_headers()
            return
        text = body["messages"][-1]["content"]
        if isinstance(text, list):
            text = " ".join(p.get("text", "") for p in text if p["type"] == "text")
        reply = "SCORE: 1" if "grading answers" in text else "DECISION: SPEAK\nRESPONSE: B"
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": reply}}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out.encode())

    def log_message(self, *args):
        pass


@pytest.fixture()
def server():
    _Handler.requests, _Handler.fail_first = [], 0
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_api_model_request_and_reply(server, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    monkeypatch.setenv("UNIT_KEY", "k123")
    m = build_model("api:unit-model", base_url=server, api_key_env="UNIT_KEY",
                    max_tokens_field="max_tokens", extra_body={"temperature": 0})
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (8, 8))},
                                             {"type": "text", "text": "decide"}]}]
    gen = m.timed_chat(messages, max_new_tokens=32)
    assert gen.text == "DECISION: SPEAK\nRESPONSE: B" and gen.n_images == 1
    req = _Handler.requests[-1]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer k123"
    b = req["body"]
    assert b["model"] == "unit-model" and b["max_tokens"] == 32 and b["temperature"] == 0
    parts = b["messages"][1]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[1] == {"type": "text", "text": "decide"}


def test_api_model_retries_on_server_error(server, monkeypatch):
    monkeypatch.setenv("UNIT_KEY", "k")
    monkeypatch.setattr("time.sleep", lambda s: None)
    _Handler.fail_first = 2
    m = build_model("api:unit-model", base_url=server, api_key_env="UNIT_KEY")
    assert "SPEAK" in m.chat([{"role": "user", "content": "hi"}])
    assert len(_Handler.requests) == 3


def test_api_judge(server, tmp_path, monkeypatch):
    monkeypatch.setenv("IBENCH_JUDGE_API_KEY", "jk")
    cache = tmp_path / "j.jsonl"
    j = make_judge("api:judge-model", cache_path=cache, base_url=server)
    assert j("q", "a blue cup", "the cup is blue") == 1.0
    assert j("q", "a blue cup", "the cup is blue") == 1.0          # second call served by the cache
    assert len(_Handler.requests) == 1 and _Handler.requests[0]["auth"] == "Bearer jk"
    assert "Ground-truth answer:\na blue cup" in _Handler.requests[0]["body"]["messages"][0]["content"]
    assert len(cache.read_text().splitlines()) == 1
