#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def option_letters(n: int) -> List[str]:
    letters = []
    for i in range(n):
        letters.append(chr(ord("A") + i) if i < 26 else f"Option-{i}")
    return letters


def format_choose_txt(question: str, choices: List[str], answer_idx: int) -> Tuple[str, str]:
    letters = option_letters(len(choices))
    options_text = "\n".join(
        f"{letters[i]}. {str(choice).strip()}" for i, choice in enumerate(choices)
    )
    prompt = f"{question.strip()}\n{options_text}"
    if 0 <= answer_idx < len(choices):
        answer = str(choices[answer_idx]).strip()
    else:
        answer = str(answer_idx)
    return prompt, answer


def format_fill_blank(question: str, answer: str) -> Tuple[str, str]:
    return question.strip(), str(answer).strip()


def format_choose_img(question: str, answer_idx: int, choices: List[str]) -> Tuple[str, str]:
    letters = option_letters(len(choices))
    options_text = "\n".join(
        f"{letters[i]}. choice_{i}" for i in range(len(choices))
    )
    prompt = (
        f"{question.strip()}\n"
        f"Options are image candidates in the sample folder:\n{options_text}"
    )
    answer = letters[answer_idx] if 0 <= answer_idx < len(letters) else str(answer_idx)
    return prompt, answer


def convert_split(
    iconqa_root: Path,
    split: str,
    include_choose_img: bool,
) -> Tuple[List[Dict], Dict[str, int]]:
    records: List[Dict] = []
    stats = {"choose_img": 0, "choose_txt": 0, "fill_in_blank": 0, "skipped": 0}

    type_dirs = ["choose_txt", "fill_in_blank"]
    if include_choose_img:
        type_dirs = ["choose_img"] + type_dirs

    for ques_type in type_dirs:
        base = iconqa_root / split / ques_type
        if not base.is_dir():
            continue

        for sample_dir in sorted(base.iterdir(), key=lambda p: int(p.name)):
            if not sample_dir.is_dir():
                continue

            data_file = sample_dir / "data.json"
            image_file = sample_dir / "image.png"
            if not data_file.exists() or not image_file.exists():
                stats["skipped"] += 1
                continue

            with data_file.open("r", encoding="utf-8") as f:
                item = json.load(f)

            question = str(item.get("question", "")).strip()
            if not question:
                stats["skipped"] += 1
                continue

            raw_answer = item.get("answer")
            choices = item.get("choices") or []

            if ques_type == "choose_txt":
                try:
                    answer_idx = int(raw_answer)
                except Exception:
                    stats["skipped"] += 1
                    continue
                text, answer = format_choose_txt(question, choices, answer_idx)
            elif ques_type == "fill_in_blank":
                text, answer = format_fill_blank(question, raw_answer)
            elif ques_type == "choose_img":
                try:
                    answer_idx = int(raw_answer)
                except Exception:
                    stats["skipped"] += 1
                    continue
                text, answer = format_choose_img(question, answer_idx, choices)
            else:
                stats["skipped"] += 1
                continue

            records.append(
                {
                    "id": f"{split}_{ques_type}_{sample_dir.name}",
                    "image": str(image_file.resolve()),
                    "text": text,
                    "answer": answer,
                }
            )
            stats[ques_type] += 1

    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert IconQA train split to alpaca-style JSON for Qwen2.5-VL training."
    )
    parser.add_argument(
        "--iconqa-root",
        default="/ssd/lincan/datasets/iconqa/raw/iconqa_data/iconqa",
        help="Root folder that contains train/val/test directories.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Which split to convert.",
    )
    parser.add_argument(
        "--output-json",
        default="/home/lincan/Finetune-Qwen2.5-VL/data/train_iconqa_qwen_for_vl.json",
    )
    parser.add_argument(
        "--include-choose-img",
        action="store_true",
        help="Also include choose_img samples. Disabled by default because candidate images are not passed as separate multimodal inputs.",
    )
    args = parser.parse_args()

    iconqa_root = Path(args.iconqa_root)
    records, stats = convert_split(
        iconqa_root=iconqa_root,
        split=args.split,
        include_choose_img=args.include_choose_img,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total = len(records)
    print(f"wrote {total} samples to {output_path}")
    print(f"stats: {stats}")
    if total > 0:
        print("sample:", records[0])


if __name__ == "__main__":
    main()
