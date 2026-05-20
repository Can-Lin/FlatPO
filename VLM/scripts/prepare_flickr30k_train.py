#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_PROMPT = "Describe this image in one sentence."


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_captions(raw_field: str) -> List[str]:
    if raw_field is None:
        return []
    try:
        data = json.loads(raw_field)
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    captions: List[str] = []
    for item in data:
        text = str(item).strip()
        if text:
            captions.append(text)
    return captions


def convert(
    rows: List[Dict[str, str]],
    image_root: Path,
    split: str,
    prompt: str,
) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []

    for row in rows:
        if str(row.get("split", "")).strip().lower() != split:
            continue

        filename = str(row.get("filename", "")).strip()
        img_id = str(row.get("img_id", "")).strip()
        if not filename:
            continue

        image_path = (image_root / filename).resolve()
        if not image_path.exists():
            continue

        captions = parse_captions(row.get("raw", ""))
        for cap_idx, caption in enumerate(captions):
            output.append(
                {
                    "id": f"flickr30k_{split}_{img_id}_{cap_idx}",
                    "image": str(image_path),
                    "text": prompt,
                    "answer": caption,
                }
            )

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Flickr30k CSV into alpaca-style JSON for VL SFT."
    )
    parser.add_argument(
        "--csv",
        default="/ssd/lincan/datasets/flickr30k/flickr_annotations_30k.csv",
    )
    parser.add_argument(
        "--image-root",
        default="/ssd/lincan/datasets/flickr30k/images/flickr30k-images",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text used as instruction for all samples.",
    )
    parser.add_argument(
        "--output-json",
        default="/home/lincan/Finetune-Qwen2.5-VL/data/train_flickr30k_qwen_for_vl.json",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    image_root = Path(args.image_root)

    rows = load_rows(csv_path)
    converted = convert(rows, image_root, args.split, args.prompt)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"rows={len(rows)} split={args.split} converted={len(converted)}")
    if converted:
        print("sample=", converted[0])


if __name__ == "__main__":
    main()
