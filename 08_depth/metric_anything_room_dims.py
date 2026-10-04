"""
metric_anything_room_dims.py — heuristic room L x W x H from a single-image
depth model alone (MetricAnything by default, DepthPro via --depth-model):
no Zhang/marker camera calibration, no SfM, no scale.json, no other file
from this project's pipeline. Single-image metric depth in, a room-size
guess out, per video.

This is deliberately a much cruder estimate than room_dims.py's SfM-based
one, and depends on two heuristic assumptions instead of measured ones:

  1. Focal length: MetricAnything's own model card documents a fallback of
     f_px = image width in pixels when no real camera intrinsics are known
     (see depth_models.py's predict_metric_anything) — used here for both
     the model's own metric-depth scaling AND our pinhole back-projection,
     so the two stay internally consistent even though neither is a real
     calibration.
  2. Level camera: with no multi-view camera poses (no SfM) there is no
     gravity direction to solve for, so we assume the image's vertical
     pixel axis IS the world's vertical axis. Any frame shot at a tilt
     silently biases that frame's height/width estimate.

Each sampled frame is backprojected independently (no cross-frame pose
alignment attempted — that would require SfM, which this script is
explicitly avoiding) into a per-frame heuristic L x W x H using robust
percentiles (not raw min/max — this project's SfM room_dims.py hit real
problems from un-trimmed outliers, see README). A video's final estimate
is the MEDIAN across its sampled frames, with the full per-frame spread
kept in the output JSON so a wide spread (a real, informative result, not
noise to hide) is visible rather than silently averaged away.

Usage (single video, default model = metric_anything):
  python 08_depth/metric_anything_room_dims.py \
      --frames-dir "C:\\thermovation-output\\sfm 1\\IMG_3126\\frames" \
      --out 08_depth/results_metric_anything/IMG_3126

Usage (DepthPro instead):
  python 08_depth/metric_anything_room_dims.py --depth-model depthpro \
      --frames-dir "C:\\thermovation-output\\sfm 1\\IMG_3126\\frames" \
      --out 08_depth/results_depthpro/IMG_3126

Usage (batch — every "<video>/frames" folder under a root):
  python 08_depth/metric_anything_room_dims.py \
      --frames-root "C:\\thermovation-output\\sfm 1" \
      --out 08_depth/results_metric_anything
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
from depth_models import DEPTH_MODELS


def sample_frames(frames_dir: Path, n_frames: int) -> list[Path]:
    frames = sorted(frames_dir.glob("*.jpg"))
    if len(frames) <= n_frames:
        return frames
    idx = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
    return [frames[i] for i in sorted(set(idx.tolist()))]


def estimate_dims_single_frame(depth_cm: np.ndarray, fx: float, fy: float,
                               cx: float, cy: float, pixel_stride: int) -> dict:
    """Backproject one frame's depth map (assumed-level camera, assumed
    focal length) and pull a crude L x W x H out of it via percentile
    extents — the single-image analog of room_dims.py's floor/ceiling/
    footprint logic, without gravity-from-camera-poses or a real scale."""
    h, w = depth_cm.shape
    ys, xs = np.mgrid[0:h:pixel_stride, 0:w:pixel_stride]
    d = depth_cm[ys, xs]
    valid = d > 1e-3
    xs, ys, d = xs[valid].astype(np.float64), ys[valid].astype(np.float64), d[valid]

    X = (xs - cx) / fx * d
    Y = (ys - cy) / fy * d   # image-down = world-down under the level-camera assumption
    Z = d

    height_cm = float(np.percentile(Y, 99.5) - np.percentile(Y, 0.5))
    length_cm = float(np.percentile(Z, 99.5) - np.percentile(Z, 0.5))

    far_mask = Z > np.percentile(Z, 80)
    width_cm = float(np.percentile(X[far_mask], 99) - np.percentile(X[far_mask], 1)) \
        if far_mask.sum() >= 20 else float("nan")

    return dict(height_cm=height_cm, length_cm=length_cm, width_cm=width_cm,
               n_points=int(len(d)))


def process_video(frames_dir: Path, out_dir: Path, model, predict_fn,
                  device: str, n_frames: int, pixel_stride: int, depth_model_name: str) -> dict:
    frames = sample_frames(frames_dir, n_frames)
    if not frames:
        return dict(error="no frames found")

    per_frame = []
    for fp in frames:
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # f_px=image_width is MetricAnything's own documented no-intrinsics
        # fallback, and used here as the pinhole back-projection focal
        # length regardless of which model produced the depth — DepthPro's
        # own metric output doesn't consume focal_px the way MetricAnything's
        # does (see predict_hf/predict_metric_anything in depth_models.py),
        # but the back-projection step needs SOME assumed focal length
        # either way, so the same fallback keeps both paths comparable.
        fx = fy = float(w)
        cx, cy = w / 2.0, h / 2.0
        depth_cm = predict_fn(rgb, model, device, focal_px=fx)
        dims = estimate_dims_single_frame(depth_cm, fx, fy, cx, cy, pixel_stride)
        dims["frame"] = fp.name
        per_frame.append(dims)
        print(f"    {fp.name}: H={dims['height_cm']:.0f}cm L={dims['length_cm']:.0f}cm "
              f"W={dims['width_cm']:.0f}cm", flush=True)
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()

    if not per_frame:
        return dict(error="no frames produced depth")

    def med(key):
        vals = [f[key] for f in per_frame if np.isfinite(f[key])]
        return float(np.median(vals)) if vals else None

    result = dict(
        method=f"{depth_model_name}_native_heuristic",
        assumptions="f_px=image_width (no calibration), level camera (no gravity solve)",
        n_frames_used=len(per_frame),
        height_cm=med("height_cm"), length_cm=med("length_cm"), width_cm=med("width_cm"),
        per_frame=per_frame,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Named per depth model (not always "metric_anything_room_dims.json")
    # so a depthpro run doesn't silently overwrite a metric_anything run in
    # the same --out dir, or vice versa.
    (out_dir / f"{depth_model_name}_room_dims.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Heuristic room L x W x H from MetricAnything alone (no calibration, no SfM)")
    parser.add_argument("--depth-model", choices=["metric_anything", "depthpro"],
                        default="metric_anything",
                        help="Which single-image depth model drives the heuristic (default: "
                             "metric_anything, matching this script's original scope). depthpro "
                             "uses the same level-camera/no-calibration heuristics — only the "
                             "depth source changes, not the back-projection assumptions.")
    parser.add_argument("--frames-dir", type=Path, help="Single video's frames folder")
    parser.add_argument("--frames-root", type=Path,
                        help="Batch mode: root containing <video>/frames subfolders")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output dir (single mode) or output root (batch mode)")
    parser.add_argument("--n-frames", type=int, default=10,
                        help="Evenly-sampled frames per video (default 10)")
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--force", action="store_true",
                        help="Batch mode: redo videos that already have output (default: skip/resume)")
    args = parser.parse_args()
    if not args.frames_dir and not args.frames_root:
        parser.error("pass --frames-dir (single video) or --frames-root (batch)")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_fn_ctor, predict_fn = DEPTH_MODELS[args.depth_model]
    print(f"Loading {args.depth_model}...")
    model = load_fn_ctor(device)

    if args.frames_dir:
        jobs = [(args.frames_dir, args.out)]
    else:
        jobs = sorted(
            (p, args.out / p.parent.name)
            for p in args.frames_root.glob("*/frames") if p.is_dir()
        )
        print(f"Batch mode: {len(jobs)} videos found under {args.frames_root}")

    summary = []
    for frames_dir, out_dir in jobs:
        video_name = out_dir.name
        existing_path = out_dir / f"{args.depth_model}_room_dims.json"
        if args.frames_root and not args.force and existing_path.exists():
            result = json.loads(existing_path.read_text())
            result["video"] = video_name
            summary.append(result)
            print(f"\n=== {video_name} === (skipped, already done: "
                  f"H={result.get('height_cm', 0):.0f}cm L={result.get('length_cm', 0):.0f}cm "
                  f"W={result.get('width_cm', 0):.0f}cm)", flush=True)
            continue
        print(f"\n=== {video_name} ===", flush=True)
        result = process_video(frames_dir, out_dir, model, predict_fn, device,
                               args.n_frames, args.pixel_stride, args.depth_model)
        result["video"] = video_name
        summary.append(result)
        if "error" in result:
            print(f"  FAILED: {result['error']}", flush=True)
        else:
            print(f"  -> H={result['height_cm']:.0f}cm L={result['length_cm']:.0f}cm "
                  f"W={result['width_cm']:.0f}cm (median over {result['n_frames_used']} frames)",
                  flush=True)

    if args.frames_root:
        summary_path = args.out / "summary.json"
        args.out.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved batch summary: {summary_path}")
        print("\n" + "=" * 72)
        print(f"{'video':<45} {'H(cm)':>8} {'L(cm)':>8} {'W(cm)':>8}")
        for r in summary:
            if "error" in r:
                print(f"{r['video']:<45} FAILED: {r['error']}")
            else:
                print(f"{r['video']:<45} {r['height_cm']:>8.0f} {r['length_cm']:>8.0f} {r['width_cm']:>8.0f}")


if __name__ == "__main__":
    main()
