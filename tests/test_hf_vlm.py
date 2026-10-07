"""Generic Hugging Face VLM adapter contracts that do not load model weights."""

from types import SimpleNamespace

import pytest


def test_revision_reaches_model_and_processor_and_placeholders_keep_order(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from interactionbench.models import hf_vlm

    calls = {}

    class FakeModel:
        device = "cpu"

        def eval(self):
            return self

    class FakeModelClass:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            calls["model"] = (repo, kwargs)
            return FakeModel()

    class FakeProcessorClass:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            calls["processor"] = (repo, kwargs)
            return SimpleNamespace(tokenizer=None)

    monkeypatch.setattr(hf_vlm, "_load_class", lambda _name: FakeModelClass)
    monkeypatch.setattr(hf_vlm, "_compat_shims", lambda: None)
    monkeypatch.setattr(hf_vlm.transformers, "AutoProcessor", FakeProcessorClass)

    model = hf_vlm.HFChatVLM(
        repo="org/model",
        revision="abc123",
        trust_remote_code=True,
        from_pretrained_kwargs={"low_cpu_mem_usage": False},
        processor_kwargs={"use_fast": False},
    )

    assert calls["model"][0] == calls["processor"][0] == "org/model"
    assert calls["model"][1]["revision"] == "abc123"
    assert calls["processor"][1] == {
        "trust_remote_code": True,
        "use_fast": False,
        "revision": "abc123",
    }

    first, second = object(), object()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": first},
            {"type": "text", "text": "between"},
            {"type": "image", "image": second},
        ],
    }]
    native = model._to_native(messages)
    assert native[0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "between"},
        {"type": "image"},
    ]
    assert hf_vlm.collect_images(messages) == [first, second]
