"""
auto_label_pipes.py — GDINO+SAM2 auto-generates 'pipe' mask proposals over
the Boilers COCO dataset for human review in Roboflow.

Never touches _annotations.coco.json directly — writes proposals to a
sibling file per split; see PIPE_LABELING.md for the review workflow.

Usage:
  python experiments/auto_label_pipes.py --split test
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from experiment_pipeline import load_coco_split, load_gdino, load_sam2, detect_gdino, sam2_from_boxes


def mask_to_coco_polygon(mask: np.ndarray) -> list[list[float]] | None:
    """Largest external contour of a binary mask -> one COCO polygon (flat x,y,x,y,...)."""
    mask = mask.squeeze().astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 20 or len(largest) < 3:
        return None
    return [largest.reshape(-1).astype(float).tolist()]


def bbox_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask.squeeze() > 0)
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return [x0, y0, x1 - x0, y1 - y0]


def main():
    parser = argparse.ArgumentParser(description="Auto-generate pipe mask proposals")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Boilers.coco"))
    parser.add_argument("--split", default="test", choices=["train", "test", "valid"])
    parser.add_argument("--prompt", default="pipe")
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, _, _ = load_coco_split(args.dataset_dir, args.split)
    img_dir = args.dataset_dir / args.split

    print("Loading models...")
    gdino_proc, gdino_model = load_gdino(device)
    sam2_pred = load_sam2(device)

    annotations = []
    ann_id = 1
    for img_info in images:
        img_path = img_dir / img_info["file_name"]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        boxes, labels, scores = detect_gdino(rgb, gdino_proc, gdino_model,
                                             args.prompt, args.threshold, device)
        masks = sam2_from_boxes(sam2_pred, rgb, boxes)

        for mask, score in zip(masks, scores):
            poly = mask_to_coco_polygon(mask)
            if poly is None:
                continue
            annotations.append(dict(
                id=ann_id, image_id=img_info["id"], category_id=1,
                segmentation=poly, bbox=bbox_from_mask(mask),
                area=float(mask.sum()), iscrowd=0, score=round(float(score), 3),
            ))
            ann_id += 1
        print(f"  {img_info['file_name']}: {len(masks)} pipe proposals")

    out = dict(
        images=images,
        annotations=annotations,
        categories=[dict(id=1, name="pipe", supercategory="none")],
    )
    out_path = args.dataset_dir / args.split / "_annotations.pipe_proposals.coco.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n{len(annotations)} pipe proposals across {len(images)} images -> {out_path}")
    print("Review these in Roboflow before merging into the real GT — see PIPE_LABELING.md")


if __name__ == "__main__":
    main()
