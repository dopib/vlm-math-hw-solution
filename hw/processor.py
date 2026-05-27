from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


def _to_float_array(image: Image.Image) -> np.ndarray:
    """Return HxWx3 float32 array scaled to [0, 1]."""
    return np.asarray(image, dtype=np.float32) / 255.0


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size]."""
        image = image.convert("RGB")
        size = self.config.image_size
        num_tiles = max(1, int(self.config.num_tiles))

        side = max(1, math.ceil(math.sqrt(num_tiles)))
        canvas = image.resize((side * size, side * size), Image.BILINEAR)

        mean = torch.tensor(self.IMAGE_MEAN).view(3, 1, 1)
        std = torch.tensor(self.IMAGE_STD).view(3, 1, 1)

        tiles: list[torch.Tensor] = []
        for row in range(side):
            for col in range(side):
                if len(tiles) >= num_tiles:
                    break
                box = (col * size, row * size, (col + 1) * size, (row + 1) * size)
                crop = canvas.crop(box)
                arr = torch.from_numpy(_to_float_array(crop))
                chw = arr.permute(2, 0, 1).contiguous()
                tiles.append((chw - mean) / std)

        return torch.stack(tiles, dim=0).float()

    def _visual_token_block(self) -> str:
        image_tokens = " ".join([IMAGE_TOKEN] * self.config.num_image_tokens)
        return f"{IMAGE_START_TOKEN} {image_tokens} {IMAGE_END_TOKEN}"

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options."""
        options_text = "\n".join(sample.options)
        prompt = (
            f"{self._visual_token_block()}\n"
            "Реши визуально-математическую задачу. "
            "Выбери один вариант ответа и напиши только букву.\n"
            f"Вопрос: {sample.question}\n"
            f"Варианты:\n{options_text}\n"
            "Ответ:"
        )
        if include_answer:
            prompt = f"{prompt} {sample.answer}"
        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample.

        labels are IGNORE_INDEX for prompt tokens and real token ids only for
        the assistant answer (so loss is computed on the answer alone).
        """
        cfg = self.config
        prompt = self.build_prompt(sample, include_answer=False)
        prompt_ids = list(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        answer_ids = list(
            self.tokenizer(f" {sample.answer}", add_special_tokens=False)["input_ids"]
        )

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            answer_ids = answer_ids + [int(eos_id)]

        input_ids = (prompt_ids + answer_ids)[: cfg.max_length]
        labels = ([cfg.ignore_index] * len(prompt_ids) + answer_ids)[: cfg.max_length]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values.

        Pads text fields to the longest sequence in the batch and stacks
        pixel_values into [B, T, 3, H, W].
        """
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
        ignore = self.config.ignore_index

        max_len = max(item["input_ids"].shape[0] for item in batch)

        def _pad(seq: torch.Tensor, value: int) -> torch.Tensor:
            pad_amount = max_len - seq.shape[0]
            if pad_amount == 0:
                return seq
            padding = torch.full((pad_amount,), value, dtype=seq.dtype)
            return torch.cat([seq, padding], dim=0)

        input_ids = torch.stack([_pad(b["input_ids"], int(pad_id)) for b in batch])
        attention_mask = torch.stack([_pad(b["attention_mask"], 0) for b in batch])
        labels = torch.stack([_pad(b["labels"], ignore) for b in batch])
        pixel_values = torch.stack([b["pixel_values"] for b in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }
