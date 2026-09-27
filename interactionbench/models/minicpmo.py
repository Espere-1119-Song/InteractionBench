"""MiniCPM-o adapter: the remote code exposes ``model.chat(msgs=...)`` (no processor /
apply_chat_template path), so the generic HFChatVLM cannot drive it. Vision-only load
(init_audio/init_tts False). Needs its own environment with transformers 4.51."""
from __future__ import annotations

import torch

from .base import ChatModel


class MiniCPMOChat(ChatModel):
    def __init__(self, repo: str, short_name: str | None = None, dtype: str = "bfloat16",
                 attn_implementation: str = "sdpa", from_pretrained_kwargs: dict | None = None,
                 max_slice_nums: int = 1, history_budget_tokens: int = 12288, **_ignored):
        from transformers import AutoModel, AutoTokenizer
        self.repo = repo
        self.name = short_name or repo.split("/")[-1]
        kw = {"init_audio": False, "init_tts": False}
        kw.update(from_pretrained_kwargs or {})
        self.model = AutoModel.from_pretrained(
            repo, trust_remote_code=True, attn_implementation=attn_implementation,
            torch_dtype=getattr(torch, dtype), **kw).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        self.max_slice_nums = max_slice_nums  # streaming frames are small; no tiling
        # Interleaved mode grows the text history without bound (images are already capped by the
        # runner at --max-frames). Prefill materialises logits for the whole prompt (bf16, 0.3 MB/token
        # per beam), so bound the history: drop the oldest user/assistant pairs over this budget.
        # Measured: 67 tokens/image, ~30 tokens/user turn, ~27 tokens/reply.
        self.history_budget_tokens = history_budget_tokens

    _IMAGE_TOKENS = 64 + 8  # one slice (max_slice_nums=1) + wrapper tokens

    def _estimate_tokens(self, msgs: list[dict]) -> int:
        n = 0
        for m in msgs:
            for p in m["content"]:
                n += len(self.tokenizer.encode(p, add_special_tokens=False)) if isinstance(p, str) else self._IMAGE_TOKENS
            n += 4  # role/turn delimiters
        return n

    def _trim_history(self, msgs: list[dict]) -> list[dict]:
        """msgs = [user, assistant, ..., user] (system already stripped). Drop the oldest
        user+assistant pair while over budget; always keep the final user turn."""
        dropped = 0
        while len(msgs) > 2 and self._estimate_tokens(msgs) > self.history_budget_tokens:
            msgs = msgs[2:]; dropped += 1
        if dropped:
            print(f"  [minicpmo] history budget: dropped {dropped} oldest turn pair(s)", flush=True)
        return msgs

    @staticmethod
    def _to_msgs(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                out.append({"role": m["role"], "content": [c]}); continue
            parts = []
            for item in c or []:
                if item.get("type") == "image":
                    parts.append(item["image"])
                elif item.get("type") == "text":
                    parts.append(item["text"])
            out.append({"role": m["role"], "content": parts})
        return out

    @torch.inference_mode()
    def chat(self, messages: list[dict], max_new_tokens: int = 96) -> str:
        msgs = self._to_msgs(messages)
        system = ""
        if msgs and msgs[0]["role"] == "system":
            system = " ".join(p for p in msgs[0]["content"] if isinstance(p, str)); msgs = msgs[1:]
        msgs = self._trim_history(msgs)
        if system and msgs:  # remote chat() has no system slot: prepend to the first user turn
            msgs[0]["content"] = [system + "\n\n"] + msgs[0]["content"]
        # chat() defaults to max_inp_length=8192 and TRUNCATES input_ids before counting image
        # placeholders -> "Expected size 16 but got size 15" once interleaved history exceeds 8k tokens.
        # The model's context is 40960; pass it explicitly.
        # do_sample=False defaults to num_beams=3 (+repetition_penalty 1.2) in the remote
        # prepare_generation_config(): prefill logits/KV are allocated x3, so a ~12k-token
        # interleaved history runs out of memory on a 45 GB GPU. Greedy decoding (num_beams=1)
        # matches every other adapter (HFChatVLM: do_sample=False, num_beams=1).
        out = self.model.chat(msgs=msgs, tokenizer=self.tokenizer, do_sample=False, num_beams=1,
                              max_new_tokens=max_new_tokens, max_slice_nums=self.max_slice_nums,
                              use_image_id=False, enable_thinking=False, max_inp_length=40960)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return str(out).strip()
