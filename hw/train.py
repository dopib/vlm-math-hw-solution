from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from hw.backbones import MockLanguageModel, MockVisionEncoder, SimpleWhitespaceTokenizer
from hw.dataset import MathVQADataset
from hw.model import MathVLM, ModelConfig
from hw.processor import MathVLMProcessor, ProcessorConfig


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_loss(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        return outputs["loss"]
    return outputs.loss


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return the scalar loss."""
    model.train()
    optimizer.zero_grad()
    outputs = model(batch)
    loss = _extract_loss(outputs)
    if not torch.isfinite(loss):
        raise ValueError(f"Loss is not finite: {loss}")
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def build_processor_config(config: dict[str, Any]) -> ProcessorConfig:
    p = config.get("processor", {})
    return ProcessorConfig(
        image_size=int(p.get("image_size", 224)),
        num_tiles=int(p.get("num_tiles", 1)),
        tile_overlap=float(p.get("tile_overlap", 0.0)),
        num_image_tokens=int(p.get("num_image_tokens", 49)),
        max_length=int(p.get("max_length", 512)),
        ignore_index=int(p.get("ignore_index", -100)),
    )


def build_pipeline(
    config: dict[str, Any], dataset: MathVQADataset
) -> tuple[MathVLM, MathVLMProcessor]:
    """Build a CPU-runnable VLM (mock backbones) wired to a shared tokenizer."""
    proc_cfg = build_processor_config(config)
    tokenizer = SimpleWhitespaceTokenizer()
    processor = MathVLMProcessor(tokenizer, proc_cfg)

    for i in range(len(dataset)):
        processor.tokenize_sample(dataset[i])

    vision_hidden = 64
    text_hidden = 64
    vision_encoder = MockVisionEncoder(hidden_size=vision_hidden)
    language_model = MockLanguageModel(vocab_size=len(tokenizer), hidden_size=text_hidden)

    model_cfg = ModelConfig(
        vision_hidden_size=vision_hidden,
        text_hidden_size=text_hidden,
        num_image_tokens=proc_cfg.num_image_tokens,
        image_token_id=tokenizer.vocab["<image>"],
    )
    model = MathVLM(vision_encoder, language_model, model_cfg)

    model_conf = config.get("model", {})
    if model_conf.get("freeze_vision", True) and model_conf.get("freeze_llm", True):
        model.freeze_backbones()

    return model, processor


def run_training(config: dict[str, Any], fast_train: bool = False) -> dict[str, Any]:
    """Main training entry point. Returns a small summary dict."""
    data_cfg = config.get("data", {})
    trainer = config.get("trainer", {})

    dataset = MathVQADataset(
        data_cfg["train_manifest"],
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples"),
    )

    model, processor = build_pipeline(config, dataset)
    device = torch.device(trainer.get("device", "cpu"))
    model.to(device)

    local_batch_size = int(trainer.get("local_batch_size", 1))
    global_batch_size = int(trainer.get("global_batch_size", local_batch_size))
    grad_accum = max(1, global_batch_size // max(1, local_batch_size))

    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=int(trainer.get("num_workers", 0)),
        collate_fn=lambda b: processor.collate([processor(s) for s in b]),
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(trainer.get("learning_rate", 5e-4)),
        weight_decay=float(trainer.get("weight_decay", 0.0)),
    )

    max_steps = int(trainer.get("max_steps", 0)) or None
    if fast_train:
        max_steps = min(max_steps or 2, 2)

    model.train()
    step = 0
    micro = 0
    last_loss = float("nan")
    optimizer.zero_grad()
    done = False
    for epoch in range(int(trainer.get("num_train_epochs", 1))):
        if done:
            break
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            loss = _extract_loss(outputs)
            if not torch.isfinite(loss):
                raise ValueError(f"Loss is not finite: {loss}")
            (loss / grad_accum).backward()
            micro += 1
            last_loss = float(loss.detach().item())

            if micro % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                print(f"step {step} | loss {last_loss:.4f}")
                if max_steps is not None and step >= max_steps:
                    done = True
                    break

    save_path = trainer.get("save_checkpoint_path")
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.adapter.state_dict(), save_path)
        print(f"Saved adapter to {save_path}")

    return {"steps": step, "last_loss": last_loss}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
