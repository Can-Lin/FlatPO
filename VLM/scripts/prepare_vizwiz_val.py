#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path
from typing import Iterable, List

import pyarrow.parquet as pq


def choose_answer(answers: Iterable[str]) -> str:
    cleaned: List[str] = []
    for ans in answers or []:
        if isinstance(ans, str):
            ans = ans.strip()
            if ans:
                cleaned.append(ans)

    if not cleaned:
        return "unanswerable"

    counts = {}
    first_pos = {}
    for idx, ans in enumerate(cleaned):
        counts[ans] = counts.get(ans, 0) + 1
        if ans not in first_pos:
            first_pos[ans] = idx

    best_count = max(counts.values())
    candidates = [ans for ans, cnt in counts.items() if cnt == best_count]
    candidates.sort(key=lambda x: first_pos[x])
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VizWiz val parquet shards to alpaca-style JSON."
    )
    parser.add_argument(
        "--parquet-glob",
        type=str,
        default="/ssd/lincan/datasets/VizWiz/data/val-*.parquet",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/lincan/Finetune-Qwen2.5-VL/data/train_vizwiz_qwen_for_vl.json",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/ssd/lincan/datasets/VizWiz/val_images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Rewrite image files even if they already exist.",
    )
    args = parser.parse_args()

    parquet_files = sorted(glob.glob(args.parquet_glob))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files matched: {args.parquet_glob}")

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    total = 0

    for parquet_file in parquet_files:
        pf = pq.ParquetFile(parquet_file)
        for batch in pf.iter_batches(
            batch_size=args.batch_size,
            columns=["question_id", "image", "question", "answers"],
        ):
            for row in batch.to_pylist():
                question_id = str(row.get("question_id", "")).strip()
                question = str(row.get("question", "")).strip()
                answer = choose_answer(row.get("answers", []))

                image_obj = row.get("image", {}) or {}
                image_name = image_obj.get("path") or f"{question_id}.jpg"
                image_bytes = image_obj.get("bytes")
                image_path = image_dir / image_name
                image_path.parent.mkdir(parents=True, exist_ok=True)

                if image_bytes is None:
                    continue

                if args.overwrite_images or (not image_path.exists()):
                    image_path.write_bytes(image_bytes)

                samples.append(
                    {
                        "id": question_id,
                        "image": str(image_path),
                        "text": question,
                        "answer": answer,
                    }
                )
                total += 1

    output_json.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {total} samples to {output_json}")
    print(f"Images saved under {image_dir}")


if __name__ == "__main__":
    main()
