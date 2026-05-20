import torch
import os
import re

from dataclasses import dataclass
from typing import Any, Dict, List
from PIL import Image
from transformers import Qwen2_5_VLProcessor


# https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune.py
# https://github.com/roboflow/notebooks/blob/main/notebooks/how-to-finetune-qwen2-5-vl-for-json-data-extraction.ipynb
IMG_TAG_PATTERN = re.compile(r"<img>(.*?)</img>", flags=re.IGNORECASE | re.DOTALL)


def _normalize_image_path(image: Any) -> Any:
    if not isinstance(image, str):
        return image

    image = image.strip()
    if not image:
        return None

    if image.startswith(("http://", "https://", "file://", "data:image")):
        return image

    if os.path.exists(image):
        return os.path.abspath(image)

    prefixed = os.path.join("/ssd/lincan", image.lstrip("/"))
    if os.path.exists(prefixed):
        return prefixed

    return image


def _ensure_min_image_size(image: Any, min_side: int = 28) -> Any:
    if not isinstance(image, Image.Image):
        return image

    width, height = image.size
    if width >= min_side and height >= min_side:
        return image

    # Qwen2/2.5-VL image processor requires both sides >= factor(28).
    scale = max(min_side / max(width, 1), min_side / max(height, 1))
    resized_width = max(min_side, int(round(width * scale)))
    resized_height = max(min_side, int(round(height * scale)))
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    return image.resize((resized_width, resized_height), resampling)


def _load_image(image: Any) -> Any:
    image = _normalize_image_path(image)
    if not isinstance(image, str):
        return image

    if not os.path.exists(image):
        return image

    with Image.open(image) as img:
        return _ensure_min_image_size(img.convert("RGB"))


def _prepare_images_for_processor(images: Any) -> Any:
    if images is None:
        return None

    if isinstance(images, list):
        if not images:
            return None
        if isinstance(images[0], list):
            return [[_load_image(image) for image in sample] for sample in images]
        return [_load_image(image) for image in images]

    return _load_image(images)


def _extract_image_from_text(text: str):
    if not isinstance(text, str):
        return None, text

    match = IMG_TAG_PATTERN.search(text)
    if match is None:
        return None, text.strip()

    image_path = _normalize_image_path(match.group(1).strip())
    clean_text = IMG_TAG_PATTERN.sub("", text).strip()
    return image_path, clean_text


def _map_role(role: str) -> str:
    role = (role or "").strip().lower()
    if role in ["user", "human"]:
        return "user"
    if role in ["assistant", "gpt", "bot"]:
        return "assistant"
    if role in ["system"]:
        return "system"
    return "user"


def _collect_vision_inputs(messages: List[Dict[str, Any]]):
    images = []
    videos = []
    for message in messages:
        for item in message.get("content", []):
            if not isinstance(item, dict):
                continue

            if item.get("type") == "image":
                image_value = _normalize_image_path(item.get("image", item.get("url")))
                if image_value is not None:
                    images.append(_load_image(image_value))
            elif item.get("type") == "video":
                video_value = _normalize_image_path(item.get("video", item.get("url")))
                if video_value is not None:
                    videos.append(video_value)

    image_inputs = images[0] if len(images) == 1 else (images if images else None)
    video_inputs = videos[0] if len(videos) == 1 else (videos if videos else None)
    return image_inputs, video_inputs


def create_message_template(batch: Dict[str, Any]):
    if (
        "messages" in batch
        and isinstance(batch["messages"], list)
        and len(batch["messages"]) > 0
    ):
        conversations = batch["messages"]
    elif (
        "conversations" in batch
        and isinstance(batch["conversations"], list)
        and len(batch["conversations"]) > 0
    ):
        conversations = batch["conversations"]
    else:
        conversations = None

    if conversations is not None:
        messages = []
        for turn in conversations:
            if not isinstance(turn, dict):
                continue

            role = _map_role(turn.get("role", turn.get("from", "")))
            content = turn.get("content", turn.get("value", ""))

            if isinstance(content, list):
                normalized_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue

                    item_type = item.get("type")
                    if item_type == "image":
                        image_value = item.get("image", item.get("url"))
                        image_value = _normalize_image_path(image_value)
                        if image_value is not None:
                            normalized_content.append({"type": "image", "image": _load_image(image_value)})
                    elif item_type == "text":
                        normalized_content.append(
                            {"type": "text", "text": str(item.get("text", ""))}
                        )

                if normalized_content:
                    messages.append({"role": role, "content": normalized_content})
                continue

            text = str(content)
            image_path, clean_text = _extract_image_from_text(text)
            multimodal_content = []
            if image_path is not None:
                multimodal_content.append({"type": "image", "image": image_path})
            if clean_text:
                multimodal_content.append({"type": "text", "text": clean_text})
            if not multimodal_content:
                multimodal_content.append({"type": "text", "text": ""})

            messages.append({"role": role, "content": multimodal_content})

        if messages:
            return messages

    if "prompt" in batch and "response" in batch:
        prompt = str(batch.get("prompt", ""))
        query = str(batch.get("query", "")).strip()
        user_text = f"{prompt}\n{query}".strip() if query else prompt
        image_value = _normalize_image_path(batch.get("image"))

        user_content = []
        if image_value is not None:
            user_content.append({"type": "image", "image": _load_image(image_value)})
        user_content.append({"type": "text", "text": user_text})

        return [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(batch.get("response", ""))}],
            },
        ]

    image_value = _normalize_image_path(batch.get("image"))
    user_content = [{"type": "text", "text": "請列出圖片中的文字"}]
    if image_value is not None:
        user_content.insert(0, {"type": "image", "image": _load_image(image_value)})

    return [
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(batch.get("text", ""))}],
        },
    ]


def find_assistant_content_sublist_indexes(label):
    """
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, \
                    {'type': 'text', 'text': '描述一下这个图片'}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': \
                    '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，\
                        坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广\
                            阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|>\
            <|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场\
                景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋\
                    和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    """
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(label) - 2):
        # Check if the current and next elements form the start sequence
        if label[i] == 151644 and label[i + 1] == 77091 and label[i + 2] == 198:
            start_indexes.append(i + 3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i + 3, len(label) - 1):
                if label[j] == 151645 and label[j + 1] == 198:
                    end_indexes.append(j + 2)
                    # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model \can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))


@dataclass
class DataCollatorForQwenVL:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        payloads = []
        for batch in features:
            messages = create_message_template(batch)
            image_inputs = batch.get("image_inputs")
            video_inputs = batch.get("video_inputs")
            if image_inputs is None and video_inputs is None:
                image_inputs, video_inputs = _collect_vision_inputs(messages)

            payloads.append(
                (
                    self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    ),
                    image_inputs,
                    video_inputs,
                )
            )

        text, image_inputs, video_inputs = map(list, zip(*payloads))
        image_inputs = _prepare_images_for_processor(image_inputs)
        batch = self.processor(
            text=text,
            images=image_inputs if any(image is not None for image in image_inputs) else None,
            videos=video_inputs if video_inputs[0] is not None else None,
            padding=True,
            return_tensors="pt",
        )
        # labels = batch["input_ids"].clone()
        # if self.processor.tokenizer.pad_token_id is not None:
        #     labels[labels == self.processor.tokenizer.pad_token_id] = -100
        # batch["labels"] = labels

        labels_list = batch["input_ids"].clone()  # .tolist()
        labels_list[labels_list == self.processor.tokenizer.pad_token_id] = -100

        if isinstance(self.processor, Qwen2_5_VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [
                self.processor.tokenizer.convert_tokens_to_ids(
                    self.processor.image_token
                )
            ]

        # labels_list = []
        # for ids_list in input_ids_lists:
        # label_ids = [-100] * len(ids_list)
        # for begin_end_indexs in find_assistant_content_sublist_indexes(
        #     ids_list, image_tokens
        # ):
        #     label_ids[begin_end_indexs[0] : begin_end_indexs[1]] = ids_list[
        #         begin_end_indexs[0] : begin_end_indexs[1]
        #     ]
        for image_token_id in image_tokens:
            labels_list[labels_list == image_token_id] = -100

        batch["labels"] = torch.tensor(labels_list, dtype=torch.int64)

        return batch
