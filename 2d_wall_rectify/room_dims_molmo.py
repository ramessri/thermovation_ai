"""
room_dims_qwen_pointing.py's homography variant with allenai/Molmo-7B-D-0924
as the pointer instead of Qwen: wall height from the 4 wall corners plus the
tank (and the boiler, where visible). Molmo is trained specifically for 2D
pointing (PixMo), so this re-tests the step Qwen failed: its "wall corners"
were just the image frame edges. One landmark per call, no cm value in any
prompt; all arithmetic in room_dims_solver.py.

Molmo answers in its own format, with coordinates as PERCENT of image size:
    <point x="41.3" y="52.7" alt="boiler">boiler</point>

Runs in a separate venv: Molmo's trust_remote_code model file (Sept 2024)
breaks against transformers 5.x in three places (bnb quantizer internals,
tie_weights() signature, GenerationMixin resolution), so it needs
transformers==4.45.2 — see README §2.

Run: <molmo-venv>/Scripts/python.exe 2d_wall_rectify/room_dims_molmo.py
"""

from __future__ import annotations

import json
import re
from typing import Optional

import cv2
import torch
from PIL import Image, ImageDraw

from config import IMAGES_DIR, OUTPUT_DIR, WALLS
from room_dims_solver import error_report, solve_wall_height

MODEL_ID = "allenai/Molmo-7B-D-0924"
OUT_DIR = OUTPUT_DIR / "room_dims_molmo"

POINT_QUERIES = {
    "tank_top": "Point to the highest point of the gray cylindrical hot water storage tank's lid.",
    "tank_base": "Point to the lowest point of the gray cylindrical hot water storage tank, where it touches the floor.",
    "boiler_top": "Point to the top edge of the white Wolf gas boiler unit.",
    "boiler_base": "Point to the bottom edge of the white Wolf gas boiler unit, where it touches the floor.",
    "wall_top_left": "Point to the corner where the back wall meets the ceiling and the left wall.",
    "wall_top_right": "Point to the corner where the back wall meets the ceiling and the right wall.",
    "wall_bottom_left": "Point to the corner where the back wall meets the floor and the left wall.",
    "wall_bottom_right": "Point to the corner where the back wall meets the floor and the right wall.",
}
CORNER_KEYS = ["wall_top_left", "wall_top_right", "wall_bottom_right", "wall_bottom_left"]
_POINT_RE = re.compile(r'x="([\d.]+)"\s+y="([\d.]+)"')


def load_molmo():
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    print(f"Loading {MODEL_ID} (4-bit NF4, vision tower unquantized)...")
    # The vision backbone stays unquantized: its LayerNorm weights are pinned
    # to fp32 and a 4-bit (fp16-compute) residual stream crashes against them.
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                                    llm_int8_skip_modules=["vision_backbone"])
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype="auto", device_map="cuda")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
                                                 quantization_config=bnb_config, device_map="cuda")
    print("Loaded.\n")
    return model, processor


def point_to(model, processor, pil_img: Image.Image, prompt: str) -> tuple[Optional[tuple[float, float]], str]:
    from transformers import GenerationConfig

    inputs = processor.process(images=[pil_img], text=prompt)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    inputs["images"] = inputs["images"].to(torch.float16)
    with torch.no_grad():
        output = model.generate_from_batch(
            inputs, GenerationConfig(max_new_tokens=100, stop_strings="<|endoftext|>", use_cache=True),
            tokenizer=processor.tokenizer)
    resp = processor.tokenizer.decode(output[0, inputs["input_ids"].size(1):], skip_special_tokens=True)
    m = _POINT_RE.search(resp)
    if m is None:
        return None, resp
    return (float(m.group(1)) / 100.0 * pil_img.width, float(m.group(2)) / 100.0 * pil_img.height), resp


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, processor = load_molmo()

    results = {}
    for wall in (w for w in WALLS if w.has_tank):
        pil_img = Image.fromarray(cv2.cvtColor(cv2.imread(str(IMAGES_DIR / wall.file)), cv2.COLOR_BGR2RGB))
        print(f"=== {wall.name} ({wall.file}) — native {pil_img.width}x{pil_img.height} ===")

        points: dict[str, Optional[tuple[float, float]]] = {}
        for key, prompt in POINT_QUERIES.items():
            if key.startswith("boiler") and not wall.has_boiler:
                continue
            points[key], resp = point_to(model, processor, pil_img, prompt)
            print(f"  {key}: {resp.strip()!r} -> {points[key]}")
        results[wall.name] = {"points": points}

        annotated = pil_img.copy()
        draw = ImageDraw.Draw(annotated)
        r = max(8, pil_img.width // 150)
        for key, pt in points.items():
            if pt is not None:
                draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], outline="red", width=4)
                draw.text((pt[0] + r + 4, pt[1] - r), key, fill="yellow")
        annotated.save(OUT_DIR / f"{wall.name}_points.png")

        if not all(points.get(k) for k in CORNER_KEYS):
            print("  Missing wall corner point(s) — skipping the solve.\n")
            continue
        result = solve_wall_height([points[k] for k in CORNER_KEYS], points)
        if result is None:
            print("  No usable reference object point — skipping the solve.\n")
            continue
        results[wall.name]["solve"] = result
        if result["scale_disagreement_pct"] is not None:
            print(f"  Rectified boiler/tank scale disagreement: {result['scale_disagreement_pct']:.1f}%")
        print(f"  Height: {error_report(result['height_cm'], wall.height_cm)}\n")

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
