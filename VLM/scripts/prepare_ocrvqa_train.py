#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def convert(records: List[Dict], dataset_root: Path) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for item in records:
        conversations = item.get("conversations") or []
        if len(conversations) < 2:
            continue

        prompt = str(conversations[0].get("value", "")).strip()
        answer = str(conversations[1].get("value", "")).strip()
        sample_id = str(item.get("id", "")).strip()
        image_path = dataset_root / "train" / sample_id / "image.jpg"

        prompt_lines = [line for line in prompt.splitlines() if "<img>" not in line]
        question = "\n".join(line.strip() for line in prompt_lines if line.strip())
        if question.startswith("Picture 1:"):
            question = question[len("Picture 1:") :].strip()

        if not sample_id or not question or not answer or not image_path.exists():
            continue

        output.append(
            {
                "id": sample_id,
                "image": str(image_path.resolve()),
                "text": question,
                "answer": answer,
            }
        )

    return output


def sample_records(records: List[Dict[str, str]], sample_size: int, seed: int) -> List[Dict[str, str]]:
    if sample_size is None:
        return records
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(records):
        raise ValueError(f"sample_size={sample_size} exceeds available records={len(records)}")
    rng = random.Random(seed)
    indexes = rng.sample(range(len(records)), sample_size)
    return [records[i] for i in indexes]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OCR-VQA Qwen-format training data to alpaca-style JSON for Qwen2.5-VL training."
    )
    parser.add_argument(
        "--input-json",
        default="/ssd/lincan/datasets/ocrvqa/train_ocrvqa_qwen.json",
    )
    parser.add_argument(
        "--dataset-root",
        default="/ssd/lincan/datasets/ocrvqa",
    )
    parser.add_argument(
        "--output-json",
        default="/home/lincan/Finetune-Qwen2.5-VL/data/train_ocrvqa_qwen_for_vl.json",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with Path(args.input_json).open("r", encoding="utf-8") as f:
        records = json.load(f)

    converted = convert(records, Path(args.dataset_root))
    converted = sample_records(converted, args.sample_size, args.seed)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"input={len(records)} converted={len(converted)} output={output_path}")
    if converted:
        print("sample=", converted[0])


if __name__ == "__main__":
    main()
