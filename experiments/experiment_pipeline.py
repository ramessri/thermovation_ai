"""
Three-mode segmentation experiment for boiler-room MEP detection.

Mode 1 (gdino):      GroundingDINO detects text prompt → SAM2 masks
Mode 2 (yoloworld):  YOLO-World detects class prompt  → SAM2 masks
Mode 3 (manual):     GT COCO boxes used as SAM2 box prompts (proxy for manual click)
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch


# ── colour palette ────────────────────────────────────────────────────────────
PALETTE = [
    (255, 64,  64),
    (64,  200, 64),
    (64,  64,  255),
    (255, 200, 0),
    (200, 0,   255),
    (0,   220, 220),
]


# ── visualisation helpers ─────────────────────────────────────────────────────

def overlay_masks(image: np.ndarray, masks: list[np.ndarray],
                  labels: list[str], alpha: float = 0.45) -> np.ndarray:
    out = image.copy().astype(np.float32)
    for i, (mask, label) in enumerate(zip(masks, labels)):
        mask = mask.squeeze()
        color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        out[mask > 0] = out[mask > 0] * (1 - alpha) + color * alpha

    out = out.astype(np.uint8)
    for i, (mask, label) in enumerate(zip(masks, labels)):
        mask = mask.squeeze()
        ys, xs = np.where(mask > 0)
        if len(xs):
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(out, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_boxes(image: np.ndarray, boxes_xyxy: np.ndarray,
               labels: list[str], scores: list[float] | None = None) -> np.ndarray:
    out = image.copy()
    for i, (box, label) in enumerate(zip(boxes_xyxy, labels)):
        color = PALETTE[i % len(PALETTE)]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {scores[i]:.2f}" if scores else label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def save_result(image: np.ndarray, boxes: np.ndarray, masks: list[np.ndarray],
                labels: list[str], scores: list[float] | None,
                out_path: Path):
    canvas = draw_boxes(image, boxes, labels, scores)
    canvas = overlay_masks(canvas, masks, labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


# ── model loaders ─────────────────────────────────────────────────────────────

def load_sam2(device: str):
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2.1-hiera-small", device=device
    )
    print("  SAM2 loaded (sam2.1-hiera-small)")
    return predictor


def load_gdino(device: str):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    print("  GroundingDINO loaded (grounding-dino-tiny)")
    return processor, model


def load_yolo_world():
    from ultralytics import YOLO
    model = YOLO("yolov8s-worldv2.pt")
    print("  YOLO-World loaded (yolov8s-worldv2)")
    return model


def load_yoloe(device: str):
    from ultralytics import YOLOE
    model = YOLOE("yoloe-11s-seg.pt")
    model.to(device)
    print("  YOLOE loaded (yoloe-11s-seg.pt)")
    return model


# ── detectors ────────────────────────────────────────────────────────────────

def detect_gdino(image_rgb: np.ndarray, processor, model,
                 text_prompt: str, threshold: float, device: str) -> tuple[np.ndarray, list[str], list[float]]:
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(image_rgb)
    # GroundingDINO expects "." at end of each phrase
    prompt = text_prompt.rstrip(".") + "."
    inputs = processor(images=pil_img, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=threshold, text_threshold=threshold,
        target_sizes=[pil_img.size[::-1]],
    )[0]

    boxes = results["boxes"].cpu().numpy()       # xyxy, absolute
    scores = results["scores"].cpu().tolist()
    labels = results["labels"]
    return boxes, labels, scores


def detect_yolo_world(image_rgb: np.ndarray, model,
                      classes: list[str], threshold: float) -> tuple[np.ndarray, list[str], list[float]]:
    model.set_classes(classes)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(bgr, conf=threshold, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().tolist()
    class_ids = results.boxes.cls.cpu().tolist()
    labels = [classes[int(c)] for c in class_ids]
    return boxes, labels, scores


def detect_yoloe(image_rgb: np.ndarray, model, classes: list[str],
                 threshold: float) -> tuple[np.ndarray, list[str], list[float], list[np.ndarray]]:
    model.set_classes(classes, model.get_text_pe(classes))
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(bgr, conf=threshold, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().tolist()
    class_ids = results.boxes.cls.cpu().tolist()
    labels = [classes[int(c)] for c in class_ids]
    masks = []
    if results.masks is not None:
        h, w = image_rgb.shape[:2]
        for m in results.masks.data.cpu().numpy():
            masks.append(cv2.resize(m.astype(np.uint8), (w, h),
                                    interpolation=cv2.INTER_NEAREST))
    return boxes, labels, scores, masks


# ── SAM2 prediction ───────────────────────────────────────────────────────────

def sam2_from_boxes(predictor, image_rgb: np.ndarray,
                    boxes_xyxy: np.ndarray) -> list[np.ndarray]:
    if len(boxes_xyxy) == 0:
        return []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        predictor.set_image(image_rgb)
        masks_out = []
        for box in boxes_xyxy:
            masks, _, _ = predictor.predict(
                box=box[None],          # (1, 4)
                multimask_output=False,
            )
            masks_out.append(masks[0].astype(np.uint8))
    return masks_out


# ── COCO dataset loader ───────────────────────────────────────────────────────

def load_coco_split(dataset_dir: Path, split: str) -> tuple[list[dict], dict, dict]:
    ann_file = dataset_dir / split / "_annotations.coco.json"
    data = json.loads(ann_file.read_text())
    id_to_img = {img["id"]: img for img in data["images"]}
    id_to_cat = {cat["id"]: cat["name"] for cat in data["categories"]}
    img_to_anns: dict[int, list] = {}
    for ann in data["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)
    return data["images"], img_to_anns, id_to_cat


# ── per-mode runners ──────────────────────────────────────────────────────────

def run_gdino_mode(images: list[dict], img_dir: Path, img_to_anns,
                   gdino_proc, gdino_model, sam2_pred,
                   text_prompt: str, threshold: float,
                   output_dir: Path, device: str) -> list[dict]:
    results = []
    for img_info in images:
        img_path = img_dir / img_info["file_name"]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        boxes, labels, scores = detect_gdino(rgb, gdino_proc, gdino_model,
                                             text_prompt, threshold, device)
        det_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        masks = sam2_from_boxes(sam2_pred, rgb, boxes)
        seg_ms = (time.perf_counter() - t1) * 1000

        stem = Path(img_info["file_name"]).stem
        if len(boxes):
            save_result(bgr, boxes, masks, labels, scores,
                        output_dir / "gdino" / f"{stem}_gdino.jpg")

        record = dict(file=img_info["file_name"], detections=len(boxes),
                      det_ms=round(det_ms, 1), seg_ms=round(seg_ms, 1))
        results.append(record)
        print(f"  [gdino] {img_info['file_name']}: {len(boxes)} det  "
              f"det={det_ms:.0f}ms  seg={seg_ms:.0f}ms")
    return results


def run_yoloworld_mode(images: list[dict], img_dir: Path, img_to_anns,
                       yolo_model, sam2_pred,
                       classes: list[str], threshold: float,
                       output_dir: Path) -> list[dict]:
    results = []
    for img_info in images:
        img_path = img_dir / img_info["file_name"]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        boxes, labels, scores = detect_yolo_world(rgb, yolo_model, classes, threshold)
        det_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        masks = sam2_from_boxes(sam2_pred, rgb, boxes)
        seg_ms = (time.perf_counter() - t1) * 1000

        stem = Path(img_info["file_name"]).stem
        if len(boxes):
            save_result(bgr, boxes, masks, labels, scores,
                        output_dir / "yoloworld" / f"{stem}_yolo.jpg")

        record = dict(file=img_info["file_name"], detections=len(boxes),
                      det_ms=round(det_ms, 1), seg_ms=round(seg_ms, 1))
        results.append(record)
        print(f"  [yolo-world] {img_info['file_name']}: {len(boxes)} det  "
              f"det={det_ms:.0f}ms  seg={seg_ms:.0f}ms")
    return results


def run_yoloe_mode(images: list[dict], img_dir: Path, img_to_anns,
                   yoloe_model, classes: list[str], threshold: float,
                   output_dir: Path) -> list[dict]:
    """YOLOE is single-model open-vocab detect+segment — no SAM2 step,
    its own masks are what gets scored as the 4th arm."""
    results = []
    for img_info in images:
        img_path = img_dir / img_info["file_name"]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        boxes, labels, scores, masks = detect_yoloe(rgb, yoloe_model, classes, threshold)
        det_ms = (time.perf_counter() - t0) * 1000

        stem = Path(img_info["file_name"]).stem
        if len(boxes):
            save_result(bgr, boxes, masks, labels, scores,
                        output_dir / "yoloe" / f"{stem}_yoloe.jpg")

        record = dict(file=img_info["file_name"], detections=len(boxes),
                      det_ms=round(det_ms, 1), seg_ms=0.0)
        results.append(record)
        print(f"  [yoloe] {img_info['file_name']}: {len(boxes)} det  "
              f"det+seg={det_ms:.0f}ms")
    return results


def run_manual_mode(images: list[dict], img_dir: Path, img_to_anns: dict,
                    id_to_cat: dict, sam2_pred,
                    output_dir: Path) -> list[dict]:
    """Uses GT COCO bounding boxes as box prompts — proxy for 'user clicked here'."""
    results = []
    for img_info in images:
        img_path = img_dir / img_info["file_name"]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        anns = img_to_anns.get(img_info["id"], [])
        if not anns:
            continue

        # COCO bbox is [x, y, w, h] → convert to xyxy
        boxes = np.array([[a["bbox"][0], a["bbox"][1],
                           a["bbox"][0] + a["bbox"][2],
                           a["bbox"][1] + a["bbox"][3]] for a in anns], dtype=np.float32)
        labels = [id_to_cat.get(a["category_id"], "?") for a in anns]

        t0 = time.perf_counter()
        masks = sam2_from_boxes(sam2_pred, rgb, boxes)
        seg_ms = (time.perf_counter() - t0) * 1000

        stem = Path(img_info["file_name"]).stem
        save_result(bgr, boxes, masks, labels, None,
                    output_dir / "manual" / f"{stem}_manual.jpg")

        record = dict(file=img_info["file_name"], detections=len(boxes),
                      seg_ms=round(seg_ms, 1))
        results.append(record)
        print(f"  [manual/GT] {img_info['file_name']}: {len(boxes)} boxes  "
              f"seg={seg_ms:.0f}ms")
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Three-mode MEP segmentation experiment")
    parser.add_argument("--dataset", default="boilers",
                        choices=["boilers"],
                        help="Which dataset to use")
    parser.add_argument("--split", default="test",
                        choices=["train", "test", "valid"],
                        help="Dataset split")
    parser.add_argument("--mode", default="all",
                        choices=["all", "gdino", "yoloworld", "yoloe", "manual"],
                        help="Which pipeline mode(s) to run")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N images (useful for quick tests)")
    parser.add_argument("--gdino-prompt", default="pipe",
                        help="Text prompt for GroundingDINO (default: 'pipe')")
    parser.add_argument("--yolo-classes", default="boiler",
                        help="Comma-separated class names for YOLO-World (default: 'boiler')")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--output", default="output/experiment",
                        help="Output directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    dataset_roots = {
        "boilers": Path(__file__).parent
    }
    dataset_dir = dataset_roots[args.dataset]
    img_dir = dataset_dir / args.split
    output_dir = Path(args.output) / args.dataset / args.split
    yolo_classes = [c.strip() for c in args.yolo_classes.split(",")]

    print(f"Dataset : {dataset_dir.name} / {args.split}")
    print(f"Mode    : {args.mode}")
    print(f"Output  : {output_dir}\n")

    images, img_to_anns, id_to_cat = load_coco_split(dataset_dir, args.split)
    if args.n:
        images = images[: args.n]
    print(f"Images to process: {len(images)}\n")

    run_gdino = args.mode in ("all", "gdino")
    run_yolo  = args.mode in ("all", "yoloworld")
    run_yoloe = args.mode in ("all", "yoloe")
    run_manual = args.mode in ("all", "manual")

    # Load SAM2 once (shared across modes)
    print("Loading models...")
    sam2_pred = load_sam2(device)

    gdino_proc = gdino_model = yolo_model = yoloe_model = None
    if run_gdino:
        gdino_proc, gdino_model = load_gdino(device)
    if run_yolo:
        yolo_model = load_yolo_world()
    if run_yoloe:
        yoloe_model = load_yoloe(device)
    print()

    summary = {}

    if run_gdino:
        print(f"=== Mode 1: GroundingDINO → SAM2  (prompt: '{args.gdino_prompt}') ===")
        summary["gdino"] = run_gdino_mode(
            images, img_dir, img_to_anns,
            gdino_proc, gdino_model, sam2_pred,
            args.gdino_prompt, args.threshold, output_dir, device,
        )
        print()

    if run_yolo:
        print(f"=== Mode 2: YOLO-World → SAM2  (classes: {yolo_classes}) ===")
        summary["yoloworld"] = run_yoloworld_mode(
            images, img_dir, img_to_anns,
            yolo_model, sam2_pred,
            yolo_classes, args.threshold, output_dir,
        )
        print()

    if run_yoloe:
        print(f"=== Mode 4: YOLOE (single-model detect+segment)  (classes: {yolo_classes}) ===")
        summary["yoloe"] = run_yoloe_mode(
            images, img_dir, img_to_anns,
            yoloe_model, yolo_classes, args.threshold, output_dir,
        )
        print()

    if run_manual:
        print("=== Mode 3: Manual / GT boxes → SAM2 ===")
        summary["manual"] = run_manual_mode(
            images, img_dir, img_to_anns, id_to_cat, sam2_pred, output_dir,
        )
        print()

    # Save summary JSON
    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved to {summary_path}")

    # Print aggregate stats
    for mode, records in summary.items():
        total_det = sum(r.get("detections", 0) for r in records)
        avg_det = round(sum(r.get("det_ms", 0) for r in records) / max(len(records), 1), 1)
        avg_seg = round(sum(r.get("seg_ms", 0) for r in records) / max(len(records), 1), 1)
        print(f"  [{mode}] total detections={total_det}  "
              f"avg det={avg_det}ms  avg seg={avg_seg}ms")


if __name__ == "__main__":
    main()
