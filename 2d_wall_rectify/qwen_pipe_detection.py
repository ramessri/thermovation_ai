"""
Can a local VLM (Qwen2-VL-2B) find the pipes GroundingDINO missed? On wall4
grounding-dino-base found zero pipes at any threshold down to 0.15, though
pipes are clearly visible. Two prompting modes on walls 3 and 4:

  - list mode: "list every pipe" as one JSON list. Degenerates on a 2B model
    (the same box repeated, or mechanically incrementing coordinates);
    repetition penalties stop the loop but break the JSON instead.
  - single-object mode: one "where is the {obj}" query per term. Clean boxes,
    and on wall4 they land on the real copper pipe + gas hose.

Single-object boxes looked convincing in isolation; qwen_obstacle_placement.py
is the full-pipeline test that reversed that verdict (README §5.1).

Run: python 2d_wall_rectify/qwen_pipe_detection.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from config import IMAGES_DIR, OUTPUT_DIR, WALLS_BY_NAME
from vlm import QWEN2_2B, ask, load_pil, load_qwen, norm_box_to_px, query_single_object

OUT_DIR = OUTPUT_DIR / "qwen_pipe_detection"

# grounding-dino-base pipe detections (box_threshold=0.20, "pipe" + "heating pipe" prompts).
GDINO_PIPE_COUNT = {"wall3": 8, "wall4": 0}
SINGLE_OBJECTS = ["pipe", "heating pipe", "gas hose"]

LIST_PROMPT = (
    "Look carefully at this photo of a boiler/utility room wall. List EVERY distinct pipe "
    "segment you can see — there are usually several (copper pipes, black/insulated heating "
    "pipes, gas hoses). Respond ONLY with a JSON list, one entry per pipe, like: "
    '[{"label": "pipe", "box_2d": [x1, y1, x2, y2]}, ...]. '
    "Coordinates must be integers normalized to a 0-1000 scale relative to this image's own "
    "width and height (0,0 = top-left, 1000,1000 = bottom-right). Do not include any text "
    "outside the JSON list."
)
_LABELED_BOX_RE = re.compile(
    r'"label"\s*:\s*"([^"]*)"\s*,\s*"box_2d"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)'
)


def detect_list_mode(model, processor, pil_img, img_w: int, img_h: int) -> tuple[list[dict], str]:
    resp = ask(model, processor, [pil_img], LIST_PROMPT, max_new_tokens=400)
    # Regex rather than json.loads: the 2B model's JSON is often slightly malformed.
    boxes = [(label or "pipe", [int(v) for v in coords])
             for label, *coords in _LABELED_BOX_RE.findall(resp)]
    # Collapse the near-identical boxes a repetition loop produces, so the
    # count isn't inflated by the same box emitted 29 times.
    kept: list[tuple[str, list[int]]] = []
    for label, box in boxes:
        if not any(all(abs(a - b) < 15 for a, b in zip(box, k)) for _, k in kept):
            kept.append((label, box))
    dets = []
    for label, box in kept:
        px = norm_box_to_px(box, img_w, img_h)
        if px is not None:
            dets.append({"label": label, "box": px})
    return dets, resp


def detect_single_mode(model, processor, pil_img, img_w: int, img_h: int) -> tuple[list[dict], list[str]]:
    dets, raw = [], []
    for obj in SINGLE_OBJECTS:
        box, resp = query_single_object(model, processor, pil_img, obj, img_w, img_h)
        raw.append(f"{obj!r} -> {resp!r}")
        if box is not None:
            dets.append({"label": obj, "box": box})
    return dets, raw


def render(img_bgr: np.ndarray, dets: list[dict], out_path: Path) -> None:
    vis = img_bgr.copy()
    for i, d in enumerate(dets, start=1):
        x0, y0, x1, y1 = (int(v) for v in d["box"])
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 255), 4)
        cv2.putText(vis, f"{i}:{d['label']}", (x0 + 4, max(20, y0 + 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, processor = load_qwen(QWEN2_2B)

    summary = {}
    for wall_name, gdino_count in GDINO_PIPE_COUNT.items():
        wall = WALLS_BY_NAME[wall_name]
        img_bgr = cv2.imread(str(IMAGES_DIR / wall.file))
        img_h, img_w = img_bgr.shape[:2]
        pil_img = load_pil(IMAGES_DIR / wall.file)
        print(f"=== {wall_name} ({wall.file}) — grounding-dino-base: {gdino_count} pipe detection(s) ===")

        list_dets, list_raw = detect_list_mode(model, processor, pil_img, img_w, img_h)
        print(f"  list mode raw response: {list_raw!r}")
        print(f"  list mode: {len(list_dets)} distinct box(es)")
        render(img_bgr, list_dets, OUT_DIR / f"{wall_name}_list_mode.jpg")

        single_dets, single_raw = detect_single_mode(model, processor, pil_img, img_w, img_h)
        print(f"  single-object mode: {len(single_dets)} box(es)")
        for d in single_dets:
            print(f"    {d['label']!r} box (orig px) = {[round(v) for v in d['box']]}")
        for r in single_raw:
            print(f"    raw: {r}")
        render(img_bgr, single_dets, OUT_DIR / f"{wall_name}_single_object.jpg")
        print()

        summary[wall_name] = {
            "gdino_pipe_count": gdino_count,
            "list_mode": {"count": len(list_dets), "boxes": list_dets, "raw": list_raw},
            "single_object_mode": {"count": len(single_dets), "boxes": single_dets, "raw": single_raw},
        }

    (OUT_DIR / "comparison.json").write_text(json.dumps(summary, indent=2))
    print("=" * 60)
    print("Pipe count: grounding-dino-base vs Qwen2-VL-2B single-object mode")
    for wall_name, s in summary.items():
        print(f"  {wall_name}: GDINO={s['gdino_pipe_count']}  Qwen2-VL={s['single_object_mode']['count']}")
    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
