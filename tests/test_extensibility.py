"""Registering models, protocols and judges; model spec resolution; the judge cache."""

import json

import pytest

from interactionbench import (ChatModel, Protocol, build_model, get_protocol, make_judge,
                              register_adapter, register_judge, register_model,
                              register_protocol)
from interactionbench.judges import CachedJudge, cache_key, parse_score
from interactionbench.models import load_model_configs, resolve_config
from interactionbench.registry import load_plugins


def test_zoo_name_and_alias_resolve():
    cfg = resolve_config("qwen3vl")
    assert cfg["adapter"] == "hf-vlm" and cfg["repo"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert cfg["short_name"] == "qwen3vl-8b"

    mage = resolve_config("mage-vl")
    assert mage == {
        "adapter": "hf-vlm",
        "repo": "microsoft/Mage-VL",
        "model_cls": "AutoModelForCausalLM",
        "image_style": "placeholder",
        "trust_remote_code": True,
        "revision": "d88b153285f1633a61b2f693c59c8576693af185",
        "attn_implementation": "sdpa",
        "short_name": "mage-vl-4b",
    }


def test_model_path_and_overrides():
    cfg = resolve_config("qwen3vl-8b", model_path="/ckpt/finetuned", attn_implementation="eager")
    assert cfg["repo"] == "/ckpt/finetuned" and cfg["attn_implementation"] == "eager"
    assert cfg["model_cls"] == "Qwen3VLForConditionalGeneration"


def test_generic_specs():
    assert resolve_config("hf:org/some-vlm") == {"adapter": "hf", "repo": "org/some-vlm"}
    cfg = resolve_config("api:my-model", base_url="http://127.0.0.1:8000/v1")
    assert cfg == {"adapter": "api", "model": "my-model", "base_url": "http://127.0.0.1:8000/v1"}
    with pytest.raises(ValueError):
        resolve_config("no-such-model")


def test_api_model_reads_key_from_environment(monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_model("api:m", base_url="http://x/v1", api_key_env="MY_KEY")
    monkeypatch.setenv("MY_KEY", "secret")
    m = build_model("api:m", base_url="http://x/v1/", api_key_env="MY_KEY")
    assert m.api_key == "secret" and m.base_url == "http://x/v1" and m.name == "api:m"


def test_register_adapter_and_model():
    @register_adapter("unit-echo", overwrite=True)
    class Echo(ChatModel):
        def __init__(self, repo="x", short_name=None, **kw):
            self.name = short_name or repo
            self.kw = kw

        def chat(self, messages, max_new_tokens=96):
            return "DECISION: WAIT\nRESPONSE:"

    assert build_model("unit-echo:abc").name == "abc"
    register_model("unit-echo-named", {"adapter": "unit-echo", "repo": "r", "temperature": 0}, overwrite=True)
    m = build_model("unit-echo-named")
    assert m.name == "unit-echo-named" and m.kw == {"temperature": 0}
    g = m.timed_chat([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    assert g.n_images == 0 and g.latency_s >= 0


def test_model_class_path(tmp_path):
    f = tmp_path / "mymodel.py"
    f.write_text("from interactionbench import ChatModel\n"
                 "class M(ChatModel):\n"
                 "    def __init__(self, **kw): self.name = 'm'\n"
                 "    def chat(self, messages, max_new_tokens=96): return 'x'\n")
    assert build_model(f"{f}:M").chat([]) == "x"


def test_model_config_file(tmp_path):
    f = tmp_path / "models.json"
    f.write_text(json.dumps({"cfg-model": {"adapter": "hf-vlm", "repo": "org/m",
                                           "image_style": "payload"}}))
    assert load_model_configs(f) == ["cfg-model"]
    assert resolve_config("cfg-model")["repo"] == "org/m"


def test_register_protocol_and_plugin_file(tmp_path):
    @register_protocol("unit-proto", overwrite=True)
    class P(Protocol):
        name = "unit-proto"

        def run_item(self, model, frames, item, question, cfg):
            return []

    assert isinstance(get_protocol("unit-proto"), P)
    assert get_protocol("unit-proto").run_tag(type("C", (), {"interval": 2.0})()) == "unit-proto_iv2"

    plugin = tmp_path / "plug.py"
    plugin.write_text("from interactionbench import register_model\n"
                      "register_model('from-plugin', {'adapter': 'api', 'model': 'x',"
                      " 'base_url': 'http://h/v1'}, overwrite=True)\n")
    load_plugins([str(plugin)])
    assert resolve_config("from-plugin")["model"] == "x"
    with pytest.raises(KeyError):
        get_protocol("missing-protocol")


def test_parse_score_scales():
    assert parse_score("SCORE: 1") == 1.0 and parse_score("SCORE: 0") == 0.0
    assert parse_score("8/10") == 1.0 and parse_score("score: 30") == 0.0
    assert parse_score("no number") == 0.0


class CountingJudge(CachedJudge):
    name = "counting"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.prompts = []

    def _generate(self, prompt):
        self.prompts.append(prompt)
        return "SCORE: 1"


def test_cached_judge_calls_once_and_persists(tmp_path):
    cache = tmp_path / "c.jsonl"
    j = CountingJudge(cache_path=cache, prompt_version="v2")
    assert j("q", "blue cup", "a blue cup") == 1.0
    assert j("q", "blue cup", "a blue cup") == 1.0
    assert j.n_calls == 1 and "contain the information" in j.prompts[0]
    assert j("q", "", "") == 1.0 and j("q", "x", "") == 0.0 and j.n_calls == 1
    j2 = CountingJudge(cache_path=cache)
    assert j2("q", "blue cup", "a blue cup") == 1.0 and j2.n_calls == 0
    assert "same information" in CountingJudge(prompt_version="v1").prompt


def test_cache_only_and_ensemble(tmp_path):
    k = cache_key("q", "gt", "pred")
    for name, s in (("a_v2_shard0", 1.0), ("b_v2_shard0", 1.0), ("c_v2_shard0", 0.0)):
        (tmp_path / f"{name}.jsonl").write_text(json.dumps({"k": k, "s": s}) + "\n")
    j = make_judge(f"cache:{tmp_path}/a_v2_shard*.jsonl")
    assert j("q", "gt", "pred") == 1.0
    assert j("q", "gt", "other") == 0.0 and j.n_missing == 1
    with pytest.raises(KeyError):
        make_judge(f"cache:{tmp_path}/a_v2_shard0.jsonl", strict=True)("q", "gt", "other")
    e = make_judge("ensemble:a,b,c", cache_dir=str(tmp_path))
    assert e("q", "gt", "pred") == 1.0
    assert make_judge("ensemble:c,a", cache_dir=str(tmp_path))("q", "gt", "pred") == 0.0


def test_register_judge():
    @register_judge("unit-const", overwrite=True)
    def make(arg, **kw):
        return lambda q, g, p: float(arg)

    assert make_judge("unit-const:1")("q", "a", "b") == 1.0
    with pytest.raises(ValueError):
        make_judge("nothing-like-this")
