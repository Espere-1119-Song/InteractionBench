"""OneStreamer model engine and causal streaming session.

Adapted from OneStreamer's Eval/Proactive_Eval/shared/inference_fast.py.
Copyright 2026 OneStreamer contributors. Licensed under Apache-2.0; see LICENSE.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import re

MODEL_ID = "MCG-NJU/OneStreamer-4B"
MODEL_REVISION = "fba29e66908b424877951fa8de46459537c8aa23"
TAG = re.compile(r"</(Silence|Standby|Response)>", re.IGNORECASE)


def decision_ticks(sample: dict, a_window: float = 10.0) -> list[float]:
    duration, question_time = float(sample["duration_s"]), float(sample["question_time_s"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video duration must be finite and positive")
    if not math.isfinite(a_window) or a_window <= 0:
        raise ValueError("A window must be finite and positive")
    if sample["time_type"] not in ("A", "B", "C"):
        raise ValueError("Unsupported time type")
    if sample["time_type"] == "A" and not (0 <= question_time <= duration):
        raise ValueError("Question time lies outside the video")
    end = min(duration, question_time + a_window) if sample["time_type"] == "A" else duration
    ticks = [float(t) for t in range(1, math.floor(end) + 1)]
    if not ticks or end - ticks[-1] > 1e-8:
        ticks.append(end)
    return ticks


def parse_action(raw: str) -> dict:
    """Accept exactly one leading tag. Keep invalid output without a retry."""
    raw = raw.strip()
    tags = list(TAG.finditer(raw))
    if len(tags) != 1 or tags[0].start() != 0:
        return dict(kind="invalid", content=None, format_error="missing_or_multiple_control_tags")
    tag, body = tags[0].group(1).lower(), raw[tags[0].end():].strip()
    if tag == "response" and not body:
        return dict(kind="invalid", content=None, format_error="empty_response")
    if tag != "response" and body:
        return dict(kind="invalid", content=None, format_error="text_after_wait_tag")
    return dict(kind=tag, content=body if tag == "response" else None, format_error=None)


@dataclass
class Turn:
    index: int
    start: float
    end: float
    images: list
    answer: str | None = None


class NativeSession:
    def __init__(self, engine, system, question, time_type, question_time_s,
                 max_rounds=32, max_tokens=128):
        if max_rounds < 1 or max_tokens < 1:
            raise ValueError("Context and generation limits must be positive")
        if time_type not in ("A", "B", "C"):
            raise ValueError("Unsupported time type")
        if "<image>" in question or "<image>" in system:
            raise ValueError("Prompt contains a reserved image placeholder")
        self.engine, self.system, self.question = engine, system, question
        self.question_time = float(question_time_s)
        self.max_tokens = max_tokens
        self.turns = deque(maxlen=max_rounds)
        self.counter = 0
        self.reveal_turn = None if time_type == "A" else 0
        self.last_input = None

    def _data(self):
        messages, images = [{"role": "system", "content": self.system}], []
        q_index = None
        if self.reveal_turn is not None:
            q_index = next(i for i, t in enumerate(self.turns) if t.index >= self.reveal_turn)
        for i, turn in enumerate(self.turns):
            text = f"<{turn.start:g}s-{turn.end:g}s>\n" + "\n".join(["<image>"] * len(turn.images))
            if i == q_index:
                text = self.question + "\n" + text
            messages.append({"role": "user", "content": text})
            images.extend(turn.images)
            if turn.answer is not None:
                messages.append({"role": "assistant", "content": turn.answer})
        return {"messages": messages, "images": images}

    def step(self, images, frame_times, start, end):
        if not (math.isfinite(start) and math.isfinite(end) and end > start >= 0
                and len(images) == len(frame_times) <= 4):
            raise ValueError("Invalid streaming interval")
        if (list(frame_times) != sorted(set(frame_times))
                or any(not (start < t <= end + 1e-8) for t in frame_times)):
            raise ValueError("Noncausal, unordered or duplicated frame")
        previous_end = self.turns[-1].end if self.turns else 0.0
        if abs(previous_end - start) > 1e-7:
            raise ValueError("Non-contiguous decision intervals")
        if self.reveal_turn is None and end + 1e-8 >= self.question_time:
            self.reveal_turn = self.counter
        turn = Turn(self.counter, float(start), float(end), images)
        self.turns.append(turn)
        self.counter += 1
        data = self._data()
        self.last_input = data
        generated = self.reveal_turn is not None
        raw = (self.engine.infer(data, max_tokens=self.max_tokens,
                                 temperature=0.0, top_p=1.0, top_k=0)
               if generated else "</Silence>")
        turn.answer = raw
        return dict(raw=raw, generated=generated, question_visible=generated,
                    history_rounds=len(self.turns), history_frames=len(data["images"]),
                    **parse_action(raw))


class FrameResizer:
    """Resize a new frame once with the same kernel as the model processor."""

    def __init__(self, processor, min_pixels, max_pixels):
        from torchvision.transforms import InterpolationMode
        from transformers.image_utils import SizeDict
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        self.processor = processor.image_processor
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self.factor = self.processor.patch_size * self.processor.merge_size
        if self.processor.resample != 3:
            raise ValueError("OneStreamer frame preprocessing requires bicubic interpolation")
        self.interpolation = InterpolationMode.BICUBIC
        self.smart_resize, self.SizeDict = smart_resize, SizeDict

    def __call__(self, image):
        from torchvision.transforms import functional as F

        h, w = self.smart_resize(image.height, image.width, factor=self.factor,
                                 min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        if (w, h) == image.size:
            return image
        resized = self.processor.resize(
            F.pil_to_tensor(image), self.SizeDict(height=h, width=w),
            interpolation=self.interpolation)
        return F.to_pil_image(resized)


class OneStreamerEngine:
    def __init__(self, model, min_pixels=3136, max_pixels=100352,
                 attn_implementation="flash_attention_2", device="cuda:0"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model, min_pixels=min_pixels, max_pixels=max_pixels)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model, dtype=torch.bfloat16, attn_implementation=attn_implementation,
            device_map=device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.prepare_frame = FrameResizer(self.processor, min_pixels, max_pixels)

    @staticmethod
    def to_chat_messages(messages, images):
        """Preserve the reference engine's image-first ordering within each turn."""
        image_iter, converted = iter(images), []
        for message in messages:
            role, content = message["role"], message["content"]
            segments = content.split("<image>")
            text_parts, image_parts = [], []
            for idx, segment in enumerate(segments):
                # Pure text turns keep their whitespace, as in the reference engine.
                text = segment if len(segments) == 1 else segment.strip()
                if text or len(segments) == 1:
                    text_parts.append({"type": "text", "text": text})
                if idx < len(segments) - 1:
                    image_parts.append({"type": "image", "image": next(image_iter)})
            converted.append({"role": role, "content": image_parts + text_parts})
        if next(image_iter, None) is not None:
            raise ValueError("More images than placeholders")
        return converted

    def infer(self, data, max_tokens=128, temperature=0.0, top_p=1.0, top_k=0):
        import torch

        if (temperature, top_p, top_k) != (0.0, 1.0, 0):
            raise ValueError("This runner uses greedy decoding")
        chat = self.to_chat_messages(data["messages"], data["images"])
        inputs = self.processor.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True,
                temperature=1.0, top_p=1.0, top_k=0,
                pad_token_id=self.processor.tokenizer.pad_token_id
                or self.processor.tokenizer.eos_token_id)
        text = self.processor.batch_decode(
            output[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=False)[0]
        for end_token in ("<|im_end|>", "<|endoftext|>"):
            while text.rstrip().endswith(end_token):
                text = text.rstrip()[:-len(end_token)]
        return text.strip()
