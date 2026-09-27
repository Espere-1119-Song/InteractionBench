"""Judge backed by a local Hugging Face chat model (text only)."""

from __future__ import annotations

from .base import DEFAULT_PROMPT_VERSION, CachedJudge


class HFJudge(CachedJudge):
    """Loads ``model_id`` with transformers. Works with text LMs and with VLMs used
    text-only. The paper judge is ``Qwen/Qwen3-14B``."""

    def __init__(self, model_id: str = "Qwen/Qwen3-14B", device: str | None = None,
                 cache_path=None, prompt_version: str = DEFAULT_PROMPT_VERSION,
                 max_new_tokens: int = 512, **_ignored):
        super().__init__(cache_path, prompt_version)
        self.name = f"hf:{model_id}"
        self.max_new_tokens = max_new_tokens
        import torch
        from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                                  AutoProcessor)

        self.processor = AutoProcessor.from_pretrained(model_id)
        kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa",
                      device_map=device or "auto")
        self.model = None
        last: Exception | None = None
        for attn in ("sdpa", "eager"):  # some architectures have no sdpa path
            kwargs["attn_implementation"] = attn
            for cls in (AutoModelForImageTextToText, AutoModelForCausalLM):
                try:
                    self.model = cls.from_pretrained(model_id, **kwargs)
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
            if self.model is not None:
                break
        if self.model is None:
            raise last  # type: ignore[misc]
        self.model.eval()

    def _generate(self, prompt: str) -> str:
        import torch
        # Plain-string content is valid for multimodal processors and required by
        # text-only tokenizers (a content list renders incorrectly in their templates).
        messages = [{"role": "user", "content": prompt}]
        # Hybrid-thinking models must answer directly; templates that do not know the
        # keyword raise TypeError and get the plain call.
        for tpl_kw in ({"enable_thinking": False, "reasoning_effort": "low"},
                       {"enable_thinking": False}, {}):
            try:
                inputs = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt", **tpl_kw)
                break
            except TypeError:
                continue
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        text = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        if "</think>" in text:  # keep the answer after a reasoning block
            text = text.rsplit("</think>", 1)[1]
        return text.strip()
