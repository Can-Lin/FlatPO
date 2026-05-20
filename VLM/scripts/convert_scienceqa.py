#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Any, Dict, List

IMG_TAG_PATTERN = re.compile(r"<img>(.*?)</img>", flags=re.IGNORECASE | re.DOTALL)


def normalize_path(p: str, root: str) -> str:
    p = p.strip()
    if not p:
        return p
    if p.startswith(("http://", "https://", "file://", "data:image")):
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p

    candidates = [
        os.path.join(root, p.lstrip("/")),
        os.path.join("/ssd/lincan", p.lstrip("/")),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    return os.path.abspath(os.path.join(root, p))


def parse_user_text(value: str, root: str):
    m = IMG_TAG_PATTERN.search(value)
    image = None
    if m is not None:
        image = normalize_path(m.group(1), root)
        value = IMG_TAG_PATTERN.sub("", value)
    return image, value.strip()


def convert_record(record: Dict[str, Any], root: str) -> Dict[str, Any]:
    conv = record.get("conversations") or record.get("messages")
    if not isinstance(conv, list) or len(conv) < 2:
        raise ValueError("invalid conversations/messages")

    user_turn = None
    assistant_turn = None
    for t in conv:
        role = (t.get("from") or t.get("role") or "").strip().lower()
        if role in ["user", "human"] and user_turn is None:
            user_turn = t
        if role in ["assistant", "gpt", "bot"]:
            assistant_turn = t

    if user_turn is None or assistant_turn is None:
        raise ValueError("cannot find user/assistant turns")

    user_text = str(user_turn.get("value", user_turn.get("content", "")))
    answer = str(assistant_turn.get("value", assistant_turn.get("content", ""))).strip()

    image, question = parse_user_text(user_text, root)
    if not question:
        raise ValueError("empty question")

    return {
        "id": record.get("id"),
        "image": image,
        "text": question,
        "answer": answer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default="/ssd/lincan")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("input json must be a list")

    converted: List[Dict[str, Any]] = []
    skipped = 0
    for i, rec in enumerate(data):
        try:
            converted.append(convert_record(rec, args.root))
        except Exception:
            skipped += 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False)

    print(f"input={len(data)} converted={len(converted)} skipped={skipped}")
    if converted:
        sample = converted[0]
        print("sample_keys=", sorted(sample.keys()))
        print("sample_image_exists=", os.path.exists(sample["image"]) if sample.get("image") else False)


if __name__ == "__main__":
    main()
