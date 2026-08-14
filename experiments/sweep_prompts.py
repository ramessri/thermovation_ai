"""
sweep_prompts.py — sweep pipe-class prompt phrasing and confidence threshold
across the 3 open-vocab arms, to diagnose why YOLO-World/YOLOE returned zero
'pipe' detections at --threshold 0.25 in the first compare_arms.py run
(GDINO found 21/40 pipe instances on the same images at the same threshold,
so pipes are visually findable — the failure is threshold/phrasing-specific
to YOLO-World's and YOLOE's zero-shot class matching, not the images).

Boiler class is held constant at the real GT name ("Boiler") rather than
swept: YOLO-World returned 0/18 Boiler detections even with that exact
wording, which already points at threshold rather than phrasing for that
class — sweeping pipe phrasing is where the open question actually is.

Usage:
  python experiments/sweep_prompts.py --split test
  python experiments/sweep_prompts.py --split test --arms yoloworld,yoloe --thresholds 0.02,0.05,0.1
"""

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from experiment_pipeline import load_coco_split, load_sam2, load_gdino, load_yolo_world, load_yoloe
from compare_arms import gt_masks_and_labels, run_arm, match_predictions, precision_recall_miou

DEFAULT_PIPE_PHRASINGS = ["pipe", "metal pipe", "heating pipe", "pipeline", "copper pipe"]
DEFAULT_THRESHOLDS = [0.05, 0.1, 0.25]


def main():
    parser = argparse.ArgumentParser(description="Sweep pipe-prompt phrasing + threshold per arm")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Boilers.coco"))
    parser.add_argument("--split", default="test", choices=["train", "test", "valid"])
    parser.add_argument("--arms", default="gdino,yoloworld,yoloe")
    parser.add_argument("--boiler-class", default="Boiler",
                        help="Held constant — this sweep targets the pipe-detection failure specifically")
    parser.add_argument("--pipe-phrasings", default=",".join(DEFAULT_PIPE_PHRASINGS))
    parser.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    parser.add_argument("--yoloworld-checkpoint", default="yolov8s-worldv2.pt",
                        help="e.g. yolov8l-worldv2.pt or yolov8x-worldv2.pt to test a bigger variant")
    parser.add_argument("--output", type=Path, default=Path("output/experiment/sweep_prompts.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arms = [a.strip() for a in args.arms.split(",")]
    pipe_phrasings = [p.strip() for p in args.pipe_phrasings.split(",")]
    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    images, img_to_anns, id_to_cat = load_coco_split(args.dataset_dir, args.split)
    img_dir = args.dataset_dir / args.split

    print("Loading models...")
    models = dict(sam2=load_sam2(device))
    if "gdino" in arms:
        models["gdino_proc"], models["gdino_model"] = load_gdino(device)
    if "yoloworld" in arms:
        models["yoloworld"] = load_yolo_world(device, checkpoint=args.yoloworld_checkpoint)
    if "yoloe" in arms:
        models["yoloe"] = load_yoloe(device)

    # load images + GT once, reused across every phrasing/threshold combo
    loaded = []
    for img_info in images:
        img_path = img_dir / img_info["file_name"]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        gt_masks, gt_labels = gt_masks_and_labels(img_info, h, w, img_to_anns, id_to_cat)
        if gt_masks:
            loaded.append((rgb, gt_masks, gt_labels))
    print(f"{len(loaded)} images with GT loaded\n")

    results = {}
    for arm in arms:
        results[arm] = {}
        for pipe_phrase, thresh in itertools.product(pipe_phrasings, thresholds):
            classes = [pipe_phrase, args.boiler_class]
            key = f"pipe='{pipe_phrase}' thresh={thresh}"
            stats_acc = {}
            for rgb, gt_masks, gt_labels in loaded:
                pred_masks, pred_labels, pred_scores = run_arm(arm, rgb, models, classes, thresh, device)
                # GT uses the dataset's real category name ('pipe'); remap
                # this run's swept phrasing back onto it before matching.
                remapped = ["pipe" if l.lower() == pipe_phrase.lower() else l for l in pred_labels]
                stats = match_predictions(pred_masks, remapped, pred_scores, gt_masks, gt_labels, 0.5)
                for label, s in stats.items():
                    acc = stats_acc.setdefault(label, dict(tp=0, fp=0, fn=0, ious=[]))
                    acc["tp"] += s["tp"]; acc["fp"] += s["fp"]; acc["fn"] += s["fn"]
                    acc["ious"].extend(s["ious"])
            metrics = precision_recall_miou(stats_acc)
            results[arm][key] = metrics
            pipe_m = metrics.get("pipe", dict(precision="-", recall="-", tp="-"))
            print(f"[{arm}] {key:35s} -> pipe: P={pipe_m['precision']} "
                  f"R={pipe_m['recall']} (tp={pipe_m['tp']})")
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
