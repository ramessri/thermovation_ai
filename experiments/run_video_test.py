"""Qualitative gdino + yolo-world detection test on raw (unannotated) video frames."""

import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from experiment_pipeline import (
    load_sam2, load_gdino, load_yolo_world,
    detect_gdino, detect_yolo_world, sam2_from_boxes, save_result,
)


def main():
    parser = argparse.ArgumentParser(description="Run gdino+yolo-world on raw frames")
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("--gdino-prompt", default="boiler. pipe. valve. pressure gauge.")
    parser.add_argument("--yolo-classes", default="boiler,pipe,valve,pressure gauge")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("output/video_test"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    images = sorted(args.frames_dir.glob("*.jpg"))
    print(f"Frames: {len(images)}")

    sam2_pred = load_sam2(device)
    gdino_proc, gdino_model = load_gdino(device)
    yolo_model = load_yolo_world(device)
    yolo_classes = [c.strip() for c in args.yolo_classes.split(",")]

    summary = {"gdino": [], "yoloworld": []}

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        stem = img_path.stem

        t0 = time.perf_counter()
        boxes, labels, scores = detect_gdino(rgb, gdino_proc, gdino_model,
                                             args.gdino_prompt, args.threshold, device)
        det_ms = (time.perf_counter() - t0) * 1000
        masks = sam2_from_boxes(sam2_pred, rgb, boxes) if len(boxes) else []
        if len(boxes):
            save_result(bgr, boxes, masks, labels, scores,
                        args.output / "gdino" / f"{stem}_gdino.jpg")
        summary["gdino"].append(dict(file=img_path.name, detections=len(boxes), det_ms=round(det_ms, 1)))
        print(f"  [gdino] {img_path.name}: {len(boxes)} det  det={det_ms:.0f}ms")

        t0 = time.perf_counter()
        boxes, labels, scores = detect_yolo_world(rgb, yolo_model, yolo_classes, args.threshold)
        det_ms = (time.perf_counter() - t0) * 1000
        masks = sam2_from_boxes(sam2_pred, rgb, boxes) if len(boxes) else []
        if len(boxes):
            save_result(bgr, boxes, masks, labels, scores,
                        args.output / "yoloworld" / f"{stem}_yolo.jpg")
        summary["yoloworld"].append(dict(file=img_path.name, detections=len(boxes), det_ms=round(det_ms, 1)))
        print(f"  [yolo-world] {img_path.name}: {len(boxes)} det  det={det_ms:.0f}ms")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
