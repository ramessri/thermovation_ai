"""
Shared Qwen vision-language-model helpers for the VLM comparison scripts in
this folder. Everything runs locally/offline via transformers on a 10GB GPU:
Qwen2-VL-2B in fp16, Qwen2.5-VL-7B in 4-bit NF4 (fp16 7B doesn't fit).

max_pixels caps the vision tokens per image: a full 4032x3024 photo at
native resolution OOMs a 10GB card in Qwen's vision tower.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import cv2
import torch
from PIL import Image

QWEN2_2B = "Qwen/Qwen2-VL-2B-Instruct"
QWEN25_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28   # ~896x896 effective
RESIZE_MAX_DIM = 1024


def load_qwen(model_id: str, max_pixels: int = MAX_PIXELS):
    """Returns (model, processor). The 7B model is loaded 4-bit."""
    from transformers import AutoProcessor

    print(f"Loading {model_id} (local, offline inference)...")
    if model_id == QWEN2_2B:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.float16).to("cuda")
    else:
        from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="cuda")
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=MIN_PIXELS, max_pixels=max_pixels)
    print("Loaded.\n")
    return model, processor


def load_pil(path: Path, max_dim: Optional[int] = RESIZE_MAX_DIM) -> Image.Image:
    """Photo as RGB PIL, downscaled to max_dim (None = native resolution)."""
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read {path}")
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if max_dim is not None:
        pil_img.thumbnail((max_dim, max_dim))
    return pil_img


def ask(model, processor, images: list[Image.Image], prompt: str, max_new_tokens: int) -> str:
    """One greedy chat turn: all images, then the prompt."""
    content = [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}]
    text = processor.apply_chat_template([{"role": "user", "content": content}],
                                         tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


# Single-object pointing — the only box-prompting mode that worked on the 2B
# model. Asking for a full list in one shot degenerated into repetition loops.
SINGLE_OBJECT_PROMPT = (
    'Where is the {obj} in this image? Respond with ONLY one JSON object: '
    '{{"box_2d": [x1, y1, x2, y2]}}, integers normalized 0-1000 relative to '
    "this image's width and height. If there is no {obj} visible, respond {{\"box_2d\": null}}."
)
_BOX_RE = re.compile(r'"box_2d"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]')


def norm_box_to_px(box_1000, img_w: int, img_h: int) -> Optional[list[float]]:
    """0-1000 normalized box -> original-image px, or None if degenerate."""
    x1n, y1n, x2n, y2n = box_1000
    box = [x1n / 1000.0 * img_w, y1n / 1000.0 * img_h, x2n / 1000.0 * img_w, y2n / 1000.0 * img_h]
    return box if box[2] > box[0] and box[3] > box[1] else None


def query_single_object(model, processor, pil_img: Image.Image, obj: str,
                        img_w: int, img_h: int) -> tuple[Optional[list[float]], str]:
    """Ask for one named object's box. Returns (box in original px or None, raw response)."""
    resp = ask(model, processor, [pil_img], SINGLE_OBJECT_PROMPT.format(obj=obj), max_new_tokens=60)
    m = _BOX_RE.search(resp)
    if m is None:
        return None, resp
    return norm_box_to_px([int(v) for v in m.groups()], img_w, img_h), resp


def parse_json_object(raw_text: str) -> Optional[dict]:
    """Outermost {...} in a response, tolerating ``` fences and // comments
    (the pointing prompts' own schemas contain comments the model echoes)."""
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    candidate = re.sub(r"//[^\n]*", "", m.group(0) if m else cleaned)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None
