"""Generic Hugging Face vision-language adapter driven by a config dict.

Covers most chat VLMs through two knobs:
  - ``model_cls``     which auto/explicit class to load with
  - ``image_style``   "payload"  -> image data kept inline in message content
                                    (Qwen-VL family expects this)
                      "placeholder" -> content carries {"type":"image"} markers and
                                    the PIL images are passed separately to the
                                    processor (standard HF multimodal path; LLaVA-OV2)

Both styles funnel into: apply_chat_template -> processor(text, images) -> generate.
"""

from __future__ import annotations

import torch
import transformers

from .base import ChatModel, collect_images


def _compat_shims() -> None:
    """Inject aliases so remote code written for newer transformers imports cleanly."""
    import transformers.configuration_utils as cu
    if not hasattr(cu, "PreTrainedConfig") and hasattr(cu, "PretrainedConfig"):
        cu.PreTrainedConfig = cu.PretrainedConfig
    if not hasattr(transformers, "PreTrainedConfig") and hasattr(transformers, "PretrainedConfig"):
        transformers.PreTrainedConfig = transformers.PretrainedConfig
    # VideoChat3 remote code imports a docstring constant that transformers 5.x
    # dropped; it is only consumed by an @add_start_docstrings decorator
    import transformers.video_processing_utils as vpu
    if not hasattr(vpu, "BASE_VIDEO_PROCESSOR_DOCSTRING"):
        vpu.BASE_VIDEO_PROCESSOR_DOCSTRING = ""
    # Nemotron-Omni remote code calls create_causal_mask(input_embeds=...); transformers 5.x
    # renamed the kwarg to inputs_embeds. Wrap once so the old name is accepted.
    try:
        import inspect as _inspect
        from transformers import masking_utils as _mu
        _orig = _mu.create_causal_mask
        if "input_embeds" not in _inspect.signature(_orig).parameters and not getattr(_orig, "_ibench_shim", False):
            _params = set(_inspect.signature(_orig).parameters)
            def _create_causal_mask(*a, **kw):
                if "input_embeds" in kw:
                    kw["inputs_embeds"] = kw.pop("input_embeds")
                kw = {k: v for k, v in kw.items() if k in _params}  # e.g. cache_position (4.x-only)
                return _orig(*a, **kw)
            _create_causal_mask._ibench_shim = True
            _mu.create_causal_mask = _create_causal_mask
    except ImportError:
        pass
    # Molmo2 remote code looks up ROPE_INIT_FUNCTIONS["default"], a key that
    # transformers 5.x renamed away; map it back to the standard initializer
    try:
        from transformers import modeling_rope_utils as mru
        if "default" not in mru.ROPE_INIT_FUNCTIONS:
            fn = getattr(mru, "_compute_default_rope_parameters", None)
            if fn is not None:
                mru.ROPE_INIT_FUNCTIONS["default"] = fn
    except ImportError:
        pass


def _load_class(name: str):
    return getattr(transformers, name)


class HFChatVLM(ChatModel):
    def __init__(
        self,
        repo: str,
        model_cls: str = "AutoModelForImageTextToText",
        image_style: str = "placeholder",
        trust_remote_code: bool = False,
        short_name: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        chat_template_kwargs: dict | None = None,
        from_pretrained_kwargs: dict | None = None,
        drop_input_keys: list[str] | None = None,
        **_ignored,
    ):
        self.repo = repo
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.drop_input_keys = list(drop_input_keys or [])  # processor outputs the model's generate() rejects
        self.name = short_name or repo.split("/")[-1]
        self.image_style = image_style
        if trust_remote_code:
            _compat_shims()
        cls = _load_class(model_cls)
        self.model = cls.from_pretrained(
            repo,
            dtype=getattr(torch, dtype),
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            **(from_pretrained_kwargs or {}),
        ).eval()
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(repo, trust_remote_code=trust_remote_code)
        self.tokenizer = getattr(self.processor, "tokenizer", None)

    def _to_native(self, messages: list[dict]) -> list[dict]:
        if self.image_style == "payload":
            return messages
        # placeholder: strip image payloads from content
        native = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                new_c = []
                for item in c:
                    if item.get("type") == "image":
                        new_c.append({"type": "image"})
                    else:
                        new_c.append(item)
                native.append({"role": m["role"], "content": new_c})
            else:
                native.append(m)
        return native

    @torch.inference_mode()
    def chat(self, messages: list[dict], max_new_tokens: int = 96) -> str:
        images = collect_images(messages)
        native = self._to_native(messages)
        text = self.processor.apply_chat_template(
            native, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs
        )
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
        ).to(self.model.device)

        for k in self.drop_input_keys:
            inputs.pop(k, None)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        in_len = inputs["input_ids"].shape[1]
        trimmed = out[:, in_len:]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        if "</think>" in decoded:  # reasoning models: keep the answer after the thinking block
            decoded = decoded.rsplit("</think>", 1)[1]
        return decoded.strip()
