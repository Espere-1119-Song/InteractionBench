"""Built-in model configurations."""

from __future__ import annotations

MODEL_ZOO: dict[str, dict] = {
    "qwen3vl-8b": {
        "adapter": "hf-vlm",
        "repo": "Qwen/Qwen3-VL-8B-Instruct",
        "model_cls": "Qwen3VLForConditionalGeneration",
        "image_style": "payload",
    },
    "qwen3vl-4b": {
        "adapter": "hf-vlm",
        "repo": "Qwen/Qwen3-VL-4B-Instruct",
        "model_cls": "Qwen3VLForConditionalGeneration",
        "image_style": "payload",
    },
    "llava-ov2-8b": {
        "adapter": "hf-vlm",
        "repo": "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
        "model_cls": "AutoModelForImageTextToText",
        "image_style": "placeholder",
        "trust_remote_code": True,
    },
    "llava-ov-7b": {
        "adapter": "hf-vlm",
        "repo": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "model_cls": "AutoModelForImageTextToText",
        "image_style": "placeholder",
    },
    "gemma-3n-e4b": {
        "adapter": "hf-vlm",
        "repo": "google/gemma-3n-E4B-it",
        "model_cls": "AutoModelForImageTextToText",
        "image_style": "placeholder",
    },
    "qwen3-omni-30b": {
        "adapter": "hf-vlm",
        "repo": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "model_cls": "Qwen3OmniMoeThinkerForConditionalGeneration",
        "image_style": "payload",
    },
    "qwen2.5-omni-7b": {
        "adapter": "hf-vlm",
        "repo": "Qwen/Qwen2.5-Omni-7B",
        "model_cls": "Qwen2_5OmniThinkerForConditionalGeneration",
        "image_style": "payload",
    },
    "minicpm-o-4.5": {
        "adapter": "minicpmo",
        "repo": "openbmb/MiniCPM-o-4_5",
        "from_pretrained_kwargs": {"init_audio": False, "init_tts": False},
    },
    "qwen3-8b-text": {
        "adapter": "hf-text",
        "repo": "Qwen/Qwen3-8B",
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "mage-vl-4b": {
        "adapter": "hf-vlm",
        "repo": "microsoft/Mage-VL",
        "model_cls": "AutoModelForCausalLM",
        "image_style": "placeholder",
        "trust_remote_code": True,
        "revision": "d88b153285f1633a61b2f693c59c8576693af185",
        "attn_implementation": "sdpa",
    },
    "videochat3-4b": {
        "adapter": "hf-vlm",
        "repo": "MCG-NJU/VideoChat3-4B",
        "model_cls": "AutoModelForCausalLM",
        "image_style": "placeholder",
        "trust_remote_code": True,
    },
    "keye-vl-8b": {
        "adapter": "hf-vlm",
        "repo": "Kwai-Keye/Keye-VL-8B-Preview",
        "model_cls": "AutoModelForCausalLM",
        "image_style": "payload",
        "trust_remote_code": True,
    },
    "keye-vl2-30b": {
        "adapter": "hf-vlm",
        "repo": "Kwai-Keye/Keye-VL-2.0-30B-A3B",
        "model_cls": "AutoModelForCausalLM",
        "image_style": "payload",
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2",
    },
    "molmo2-8b": {
        "adapter": "hf-vlm",
        "repo": "allenai/Molmo2-8B",
        "model_cls": "AutoModelForImageTextToText",
        "image_style": "placeholder",
        "trust_remote_code": True,
    },
    "nemotron-omni-30b": {
        "adapter": "hf-vlm",
        "repo": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        "model_cls": "AutoModelForCausalLM",
        "image_style": "payload",
        "trust_remote_code": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "attn_implementation": "eager",
        "drop_input_keys": ["num_patches", "num_tokens", "imgs_sizes"],
    },
    "openai-api": {
        "adapter": "api",
        "model": None,
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "claude-api": {
        "adapter": "api",
        "model": "claude-sonnet-5",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "gemini-api": {
        "adapter": "api",
        "model": None,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GOOGLE_API_KEY",
        "max_tokens_field": "max_tokens",
        "extra_body": {"reasoning_effort": "none"},
    },
}

ALIASES: dict[str, str] = {
    "qwen3vl": "qwen3vl-8b",
    "llava-ov2": "llava-ov2-8b",
    "mage-vl": "mage-vl-4b",
    "videochat3": "videochat3-4b",
    "keye": "keye-vl2-30b",
}
