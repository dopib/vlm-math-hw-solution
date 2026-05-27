from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from hw.constants import CHOICES


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract a multiple-choice answer letter from model output.

    Handles "A", "(B)", "Answer: C", "The correct answer is D." and returns
    None when no choice letter is present.
    """
    if not text:
        return None
    letters = "".join(c.upper() for c in choices)
    match = re.search(rf"\b([{letters}])\b", text)
    return match.group(1) if match else None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run the evaluation loop and return accuracy metrics."""
    import torch

    from hw.dataset import MathVQADataset
    from hw.train import build_pipeline

    data_cfg = config.get("data", {})
    inference = config.get("inference", {})

    dataset = MathVQADataset(
        data_cfg["eval_manifest"],
        split=data_cfg.get("split", "dev"),
        max_samples=data_cfg.get("max_samples"),
    )

    model, processor = build_pipeline(config, dataset)
    device = torch.device("cpu" if toy else inference.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    tokenizer = processor.tokenizer
    max_new_tokens = int(inference.get("max_new_tokens", 8))

    rows: list[dict[str, Any]] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        prompt = processor.build_prompt(sample, include_answer=False)
        input_ids = torch.tensor(
            [tokenizer(prompt, add_special_tokens=False)["input_ids"]], device=device
        )
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": processor.preprocess_image(sample.image).unsqueeze(0).to(device),
        }
        out_ids = model.generate(batch, max_new_tokens=max_new_tokens)
        decoded = tokenizer.decode(out_ids[0].tolist())
        prediction = parse_mc_answer(decoded)
        rows.append(
            {
                "id": sample.id,
                "question": sample.question,
                "prediction": prediction,
                "answer": sample.answer,
                "subject": sample.subject,
                "raw_output": decoded,
            }
        )

    output_path = inference.get("output_path")
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return compute_accuracy(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
