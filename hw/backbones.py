"""Lightweight CPU-only backbones for the Track A smoke pipeline.

These are deliberately tiny mocks so that training and benchmarking run
end-to-end on CPU without downloading any external vision/LLM checkpoints.
Track B/C can swap these for real Hugging Face models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hw.constants import (
    IGNORE_INDEX,
    IMAGE_END_TOKEN,
    IMAGE_START_TOKEN,
    IMAGE_TOKEN,
)


class SimpleWhitespaceTokenizer:
    """Deterministic whitespace tokenizer with a growable vocabulary."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}
        for token in ("<pad>", "<eos>", IMAGE_TOKEN, IMAGE_START_TOKEN, IMAGE_END_TOKEN):
            self._add(token)
        self.pad_token_id = self.vocab["<pad>"]
        self.eos_token_id = self.vocab["<eos>"]
        self._special = {"<pad>", "<eos>", IMAGE_TOKEN, IMAGE_START_TOKEN, IMAGE_END_TOKEN}

    def _add(self, token: str) -> int:
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inv[idx] = token
        return self.vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self._add(tok) for tok in text.replace("\n", " ").split()]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        for i in ids:
            token = self.inv.get(int(i), "")
            if skip_special_tokens and token in self._special:
                continue
            if token:
                tokens.append(token)
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.vocab)


class MockVisionEncoder(nn.Module):
    """Patch-embedding stand-in for a ViT encoder."""

    def __init__(self, hidden_size: int, patch_size: int = 16) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.proj(pixel_values)
        return x.flatten(2).transpose(1, 2)


@dataclass
class CausalLMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor


class MockLanguageModel(nn.Module):
    """Single-layer transformer LM used only for the CPU smoke pipeline."""

    def __init__(self, vocab_size: int, hidden_size: int, num_heads: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(
        self,
        inputs_embeds: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)

        pad_mask = None
        if attention_mask is not None:
            pad_mask = attention_mask == 0
        hidden = self.encoder(inputs_embeds, src_key_padding_mask=pad_mask)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 8,
        **_: Any,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)

        embeds = inputs_embeds
        generated: list[torch.Tensor] = []
        for _step in range(max_new_tokens):
            hidden = self.encoder(embeds)
            next_logits = self.lm_head(hidden[:, -1, :])
            next_token = next_logits.argmax(dim=-1)
            generated.append(next_token)
            embeds = torch.cat([embeds, self.embed(next_token).unsqueeze(1)], dim=1)
        return torch.stack(generated, dim=1)
