"""
compare_arms.py — IoU-matched per-class metrics across all 4 detection arms,
to pick the detector Stream A builds on (Foundation item 3).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def polygon_to_mask(segmentation: list, height: int, width: int) -> np.ndarray:
    """COCO polygon segmentation (list of flat [x,y,x,y,...] lists) -> binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def match_predictions(pred_masks: list, pred_labels: list, pred_scores: list,
                      gt_masks: list, gt_labels: list, iou_thresh: float = 0.5) -> dict:
    """Greedy best-IoU matching, predictions considered highest-confidence first.

    Labels are matched case-insensitively: open-vocab detectors don't
    reliably preserve GT casing (e.g. GroundingDINO's bert-base-uncased text
    backbone lowercases everything, so a "Boiler" prompt comes back labeled
    "boiler"). Ground truth's casing is kept as the canonical, reported key.
    """
    canonical: dict[str, str] = {}
    for l in gt_labels:
        canonical.setdefault(l.lower(), l)
    for l in pred_labels:
        canonical.setdefault(l.lower(), l)

    stats = {canonical[k]: dict(tp=0, fp=0, fn=0, ious=[]) for k in canonical}
    order = sorted(range(len(pred_masks)), key=lambda i: -pred_scores[i])
    used_gt = set()
    for pi in order:
        pm, pl = pred_masks[pi], pred_labels[pi].lower()
        best_iou, best_gi = 0.0, -1
        for gi, (gm, gl) in enumerate(zip(gt_masks, gt_labels)):
            if gi in used_gt or gl.lower() != pl:
                continue
            iou = mask_iou(pm, gm)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_thresh:
            used_gt.add(best_gi)
            stats[canonical[pl]]["tp"] += 1
            stats[canonical[pl]]["ious"].append(best_iou)
        else:
            stats[canonical[pl]]["fp"] += 1
    for gi, gl in enumerate(gt_labels):
        if gi not in used_gt:
            stats[gl]["fn"] += 1
    return stats


def precision_recall_miou(stats: dict) -> dict:
    out = {}
    for label, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        miou = float(np.mean(s["ious"])) if s["ious"] else 0.0
        out[label] = dict(precision=round(precision, 3), recall=round(recall, 3),
                          miou=round(miou, 3), tp=tp, fp=fp, fn=fn)
    return out


from experiment_pipeline import (
    load_coco_split, load_sam2, load_gdino, load_yolo_world, load_yoloe,
    detect_gdino, detect_yolo_world, detect_yoloe, sam2_from_boxes,
)
import torch


def gt_masks_and_labels(img_info: dict, height: int, width: int,
                        img_to_anns: dict, id_to_cat: dict) -> tuple[list, list]:
    """height/width must come from the actually-loaded image array, not COCO
    metadata — some Roboflow re-exports record height/width that doesn't
    match the real pixel dimensions (EXIF-rotation quirk), which silently
    produces a transposed mask and a broadcast error in mask_iou."""
    anns = img_to_anns.get(img_info["id"], [])
    masks, labels = [], []
    for a in anns:
        if not a.get("segmentation"):
            continue
        masks.append(polygon_to_mask(a["segmentation"], height, width))
        labels.append(id_to_cat.get(a["category_id"], "?"))
    return masks, labels


def run_arm(name: str, image_rgb: np.ndarray, models: dict, classes: list[str],
           threshold: float, device: str) -> tuple[list, list, list]:
    """Returns (masks, labels, scores) for one arm on one image."""
    if name == "gdino":
        # One class per GDINO call, not a joined multi-class prompt: GDINO's
        # phrase grounding can merge adjacent class words into a single
        # detected phrase (e.g. "pipe. Boiler." -> a bogus "pipe boiler"
        # detection) when they're queried together. Querying separately also
        # lets us force the label to the exact class searched for, instead of
        # trusting GDINO's decoded phrase text.
        all_masks, all_labels, all_scores = [], [], []
        for cls in classes:
            boxes, _, scores = detect_gdino(image_rgb, models["gdino_proc"], models["gdino_model"],
                                            cls, threshold, device)
            masks = sam2_from_boxes(models["sam2"], image_rgb, boxes)
            all_masks.extend(masks)
            all_labels.extend([cls] * len(masks))
            all_scores.extend(scores)
        return all_masks, all_labels, all_scores
    if name == "yoloworld":
        boxes, labels, scores = detect_yolo_world(image_rgb, models["yoloworld"], classes, threshold)
        masks = sam2_from_boxes(models["sam2"], image_rgb, boxes)
        return masks, labels, scores
    if name == "yoloe":
        boxes, labels, scores, masks = detect_yoloe(image_rgb, models["yoloe"], classes, threshold)
        return masks, labels, scores
    raise ValueError(f"Unknown arm: {name}")


def main():
    parser = argparse.ArgumentParser(description="Compare all detection arms against GT")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Boilers.coco"))
    parser.add_argument("--split", default="test", choices=["train", "test", "valid"])
    parser.add_argument("--classes", default="pipe,Boiler",
                        help="Comma-separated class list for YOLO-World/YOLOE")
    # Per-arm thresholds, not one shared value: sweep_prompts.py found each
    # arm's confidence scores live on a different scale (GDINO's real
    # operating point is ~0.25; YOLO-World's and YOLOE's are an order of
    # magnitude lower) — one global --threshold silently zeroed out the
    # weaker-calibrated arms instead of comparing each at its own best
    # setting.
    parser.add_argument("--gdino-threshold", type=float, default=0.25)
    parser.add_argument("--yoloworld-threshold", type=float, default=0.01)
    parser.add_argument("--yoloe-threshold", type=float, default=0.05)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--yoloworld-checkpoint", default="yolov8s-worldv2.pt",
                        help="e.g. yolov8l-worldv2.pt or yolov8x-worldv2.pt to test a bigger variant")
    parser.add_argument("--output", type=Path, default=Path("output/experiment/compare_arms.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arm_thresholds = dict(gdino=args.gdino_threshold, yoloworld=args.yoloworld_threshold,
                          yoloe=args.yoloe_threshold)
    classes = [c.strip() for c in args.classes.split(",")]
    images, img_to_anns, id_to_cat = load_coco_split(args.dataset_dir, args.split)
    img_dir = args.dataset_dir / args.split

    print("Loading models...")
    models = dict(
        sam2=load_sam2(device),
    )
    models["gdino_proc"], models["gdino_model"] = load_gdino(device)
    models["yoloworld"] = load_yolo_world(device, checkpoint=args.yoloworld_checkpoint)
    models["yoloe"] = load_yoloe(device)

    arm_stats = {arm: {} for arm in ("gdino", "yoloworld", "yoloe")}
    for img_info in images:
        img_path = img_dir / img_info["file_name"]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        gt_masks, gt_labels = gt_masks_and_labels(img_info, h, w, img_to_anns, id_to_cat)
        if not gt_masks:
            continue
        for arm in arm_stats:
            pred_masks, pred_labels, pred_scores = run_arm(arm, rgb, models, classes,
                                                            arm_thresholds[arm], device)
            stats = match_predictions(pred_masks, pred_labels, pred_scores,
                                      gt_masks, gt_labels, args.iou_thresh)
            for label, s in stats.items():
                acc = arm_stats[arm].setdefault(label, dict(tp=0, fp=0, fn=0, ious=[]))
                acc["tp"] += s["tp"]; acc["fp"] += s["fp"]; acc["fn"] += s["fn"]
                acc["ious"].extend(s["ious"])

    report = {arm: dict(threshold=arm_thresholds[arm], metrics=precision_recall_miou(stats))
             for arm, stats in arm_stats.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {args.output}")
    for arm, entry in report.items():
        print(f"\n=== {arm} (threshold={entry['threshold']}) ===")
        for label, m in entry["metrics"].items():
            print(f"  {label}: precision={m['precision']} recall={m['recall']} "
                  f"mIoU={m['miou']} (tp={m['tp']} fp={m['fp']} fn={m['fn']})")


if __name__ == "__main__":
    main()
