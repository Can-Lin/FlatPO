#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def load_questions(path: Path) -> Dict[int, Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(item["question_id"]): item for item in data["questions"]}


def convert(
    question_map: Dict[int, Dict],
    annotations_path: Path,
    image_root: Path,
) -> List[Dict[str, str]]:
    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]

    output: List[Dict[str, str]] = []
    for ann in annotations:
        question_id = int(ann["question_id"])
        image_id = int(ann["image_id"])
        question_obj = question_map.get(question_id)
        if question_obj is None:
            continue

        image_path = image_root / f"COCO_train2014_{image_id:012d}.jpg"
        if not image_path.exists():
            continue

        question = str(question_obj.get("question", "")).strip()
        answer = str(ann.get("multiple_choice_answer", "")).strip()
        if not question or not answer:
            continue

        output.append(
            {
                "id": str(question_id),
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
        description="Convert VQAv2 train questions and annotations to alpaca-style JSON for Qwen2.5-VL training."
    )
    parser.add_argument(
        "--questions-json",
        default="/ssd/lincan/datasets/vqav2/questions/v2_OpenEnded_mscoco_train2014_questions.json",
    )
    parser.add_argument(
        "--annotations-json",
        default="/ssd/lincan/datasets/vqav2/annotations/v2_mscoco_train2014_annotations.json",
    )
    parser.add_argument(
        "--image-root",
        default="/ssd/lincan/datasets/vqav2/train2014",
    )
    parser.add_argument(
        "--output-json",
        default="/home/lincan/Finetune-Qwen2.5-VL/data/train_vqav2_qwen_for_vl.json",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    question_map = load_questions(Path(args.questions_json))
    converted = convert(
        question_map=question_map,
        annotations_path=Path(args.annotations_json),
        image_root=Path(args.image_root),
    )
    converted = sample_records(converted, args.sample_size, args.seed)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"questions={len(question_map)} converted={len(converted)} output={output_path}")
    if converted:
        print("sample=", converted[0])


if __name__ == "__main__":
    main()
