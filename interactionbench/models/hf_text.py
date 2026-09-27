"""Text-only HF chat-LM adapter (no vision tower).

Used for the language-only baseline (e.g. Qwen/Qwen3-8B) under a polling
protocol with ``--blind``: the canonical messages carry no images, so we flatten
each content list to its text parts, apply the tokenizer chat template
(thinking disabled for reasoning-capable LMs) and decode greedily, exactly
matching HFChatVLM's generation settings.
"""

from __future__ import annotations

import torch
import transformers

from .base import ChatModel, collect_images


def _text_only(messages: list[dict]) -> list[dict]:
    native = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(item.get("text", "") for item in c if item.get("type") == "text")
        native.append({"role": m["role"], "content": c})
    return native


class HFChatLM(ChatModel):
    def __init__(
        self,
        repo: str,
        model_cls: str = "AutoModelForCausalLM",
        short_name: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = False,
        chat_template_kwargs: dict | None = None,
        from_pretrained_kwargs: dict | None = None,
        **_ignored,
    ):
        self.repo = repo
        self.name = short_name or repo.split("/")[-1]
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        cls = getattr(transformers, model_cls)
        self.model = cls.from_pretrained(
            repo,
            dtype=getattr(torch, dtype),
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            **(from_pretrained_kwargs or {}),
        ).eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            repo, trust_remote_code=trust_remote_code)

    @torch.inference_mode()
    def chat(self, messages: list[dict], max_new_tokens: int = 96) -> str:
        if collect_images(messages):
            raise ValueError(f"{self.name} is text-only; run it with --blind")
        text = self.tokenizer.apply_chat_template(
            _text_only(messages), tokenize=False, add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if "</think>" in decoded:
            decoded = decoded.rsplit("</think>", 1)[1]
        return decoded.strip()
