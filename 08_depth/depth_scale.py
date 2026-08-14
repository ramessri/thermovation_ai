"""
depth_scale.py — metric scale (cm per model unit) from a monocular metric
depth model, as an alternative to marker-based (Zhang) calibration and the
existing depth_ratio_scale marker-footprint fallback.

Same ratio mechanism as 04_scale/scale_sfm.py's depth_ratio_scale (metric
reference depth vs SfM's own model-unit depth, ratioed and robustly
aggregated) — but the metric reference here is a depth MODEL's per-pixel
prediction instead of the marker's known physical geometry, so it works on
every well-triangulated SfM point in every frame, not just marker-sighting
frames near the marker footprint. This is what makes it directly comparable
to Zhang/marker calibration on the same video: same underlying math, two
different sources of "what is the true metric depth here."

Usage (library — see 08_depth/README.md for the CLI wiring into
04_scale/scale_sfm.py's --scale-method flag):
  from depth_scale import monocular_depth_scale
  result = monocular_depth_scale(rec, frames_dir, "depthanything_v2_metric", "cuda")
"""

import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_models import DEPTH_MODELS

INVALID_P3D = 2**63 - 1


def _point_ratios_in_frame(rec: pycolmap.Reconstruction, img, depth_cm: np.ndarray,
                           max_points: int = 300) -> list[float]:
    """metric_depth_cm(pixel) / model_unit_depth(pixel) for well-triangulated
    points visible in this frame, sampled to at most max_points for speed."""
    cam = rec.cameras[img.camera_id]
    P = np.asarray(img.cam_from_world().matrix())
    h, w = cam.height, cam.width

    candidates = []
    for p2d in img.points2D:
        pid = p2d.point3D_id
        if pid == INVALID_P3D or pid not in rec.points3D:
            continue
        pt3d = rec.points3D[pid]
        if pt3d.track.length() < 3 or pt3d.error > 1.5:
            continue
        x, y = int(round(p2d.xy[0])), int(round(p2d.xy[1]))
        if not (0 <= y < h and 0 <= x < w):
            continue
        Xc = P @ np.append(pt3d.xyz, 1.0)
        if Xc[2] <= 1e-6:
            continue
        candidates.append((x, y, float(Xc[2])))

    if len(candidates) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(candidates), max_points, replace=False)
        candidates = [candidates[i] for i in idx]

    ratios = []
    for x, y, model_depth in candidates:
        metric_depth_cm = float(depth_cm[y, x])
        if metric_depth_cm > 1e-3:
            ratios.append(metric_depth_cm / model_depth)
    return ratios


def _robust_mode(ratios: np.ndarray) -> tuple[float, float]:
    """Same 'tightest-window' robust aggregation depth_ratio_scale uses:
    median of the densest 50% cluster, plus its coefficient of variation."""
    r = np.array(sorted(ratios))
    k = max(int(0.5 * len(r)), 3)
    _, best = min((r[i + k - 1] - r[i], i) for i in range(len(r) - k + 1))
    core = r[best:best + k]
    cm_per_unit = float(np.median(core))
    cv_pct = float(np.std(core) / np.mean(core) * 100)
    return cm_per_unit, cv_pct


def monocular_depth_scale(rec: pycolmap.Reconstruction, frames_dir: Path,
                          model_name: str, device: str,
                          every_n_frames: int = 3,
                          max_points_per_frame: int = 300) -> dict | None:
    """cm_per_unit from a monocular depth model, aggregated across every
    `every_n_frames`-th registered frame. Returns None if too few usable
    frames/points survive to give a robust estimate."""
    import cv2

    if model_name not in DEPTH_MODELS:
        raise ValueError(f"Unknown depth model '{model_name}', choose from {list(DEPTH_MODELS)}")
    load_fn, predict_fn = DEPTH_MODELS[model_name]
    print(f"Loading {model_name}...")
    loaded = load_fn(device)

    images = sorted(rec.images.values(), key=lambda im: im.name)[::every_n_frames]
    all_ratios = []
    frames_used = 0
    for img in images:
        bgr = cv2.imread(str(frames_dir / img.name))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        focal_px = float(rec.cameras[img.camera_id].params[0])
        depth_cm = predict_fn(rgb, loaded, device, focal_px=focal_px)
        ratios = _point_ratios_in_frame(rec, img, depth_cm, max_points_per_frame)
        if ratios:
            all_ratios.extend(ratios)
            frames_used += 1

    print(f"{model_name}: {frames_used}/{len(images)} sampled frames contributed "
          f"{len(all_ratios)} point ratios")
    if len(all_ratios) < 30:
        return None

    cm_per_unit, cv_pct = _robust_mode(np.array(all_ratios))
    return dict(cm_per_unit=cm_per_unit, model=model_name, frames_used=frames_used,
               points_used=len(all_ratios), core_cv_pct=round(cv_pct, 2))
