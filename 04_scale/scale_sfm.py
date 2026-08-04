"""
scale_sfm.py — solve the metric scale of a COLMAP sparse model using the
fiducial marker (marker→metric bridge).

The marker may be MOVED between walls during filming, so sightings are first
split into temporal segments (one segment = one physical placement). Each
segment is triangulated independently and yields its own scale estimate;
segments must internally agree (low spread) to be accepted, and the final
scale is the median across accepted segments — giving built-in
cross-validation.

Per segment:
  1. Grid-validated marker detection on the registered frames.
  2. 180° ordering ambiguity resolved temporally (consecutive frames only,
     where the heuristic is reliable).
  3. DLT triangulation of the 9 square centers on undistorted/normalized
     coordinates, with one reprojection-outlier rejection pass.
  4. 36 pairwise 3D distances vs the known cm distances → scale factors.

Usage:
  python 04_scale/scale_sfm.py output/sfm/<video_(sfm)>/sparse/1 dataset/frames/<video_(sfm)>
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_calibration"))
from detect_marker import detect_marker, GRID_CM

MAX_REPROJ_PX = 3.0        # inlier threshold for triangulated points
MAX_SEGMENT_SPREAD = 5.0   # % (10th-90th pct) for a segment to be accepted
MIN_SEGMENT_VIEWS = 5


def detect_in_registered_frames(rec: pycolmap.Reconstruction, frames_dir: Path) -> list[dict]:
    """Marker detections in registered frames, chronological order."""
    sightings = []
    for img in sorted(rec.images.values(), key=lambda im: im.name):
        bgr = cv2.imread(str(frames_dir / img.name))
        if bgr is None:
            continue
        px_per_cm, corners, H, _ = detect_marker(bgr)
        if px_per_cm is None:
            continue
        pts = cv2.perspectiveTransform(
            GRID_CM.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
        cam = rec.cameras[img.camera_id]
        P = np.asarray(img.cam_from_world().matrix())      # 3x4 world->cam
        norm_xy = np.asarray(cam.cam_from_img(pts.astype(np.float64)))
        frame_idx = int(img.name.rsplit("_", 1)[1].split(".")[0])
        sightings.append(dict(name=img.name, frame_idx=frame_idx,
                              px_per_cm=px_per_cm, pts=pts, norm_xy=norm_xy,
                              P=P, focal=float(cam.params[0])))
    print(f"Marker detected in {len(sightings)} / {rec.num_reg_images()} registered frames")
    return sightings


def split_segments(sightings: list[dict], max_gap: int = 4) -> list[list[dict]]:
    """Contiguous runs of sightings = one physical marker placement each."""
    segments: list[list[dict]] = []
    for s in sightings:
        if segments and s["frame_idx"] - segments[-1][-1]["frame_idx"] <= max_gap:
            segments[-1].append(s)
        else:
            segments.append([s])
    return segments


def resolve_ordering(segment: list[dict]) -> None:
    """Fix 180° flips in-place, consistent within a contiguous segment."""
    v_prev = None
    for s in segment:
        v = s["pts"][0] - s["pts"][4]
        if v_prev is not None and float(v @ v_prev) < 0:
            s["pts"] = s["pts"][::-1].copy()
            s["norm_xy"] = s["norm_xy"][::-1].copy()
            v = s["pts"][0] - s["pts"][4]
        v_prev = v


def triangulate_dlt(observations: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    """Multi-view linear triangulation from normalized coords + 3x4 poses."""
    if len(observations) < 2:
        return None
    rows = []
    for (x, y), P in observations:
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A = np.stack(rows)
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def reproj_errors(X: np.ndarray, observations) -> np.ndarray:
    errs = []
    for (x, y), P in observations:
        p = P @ np.append(X, 1.0)
        if p[2] <= 1e-9:
            errs.append(np.inf)
            continue
        errs.append(float(np.hypot(p[0] / p[2] - x, p[1] / p[2] - y)))
    return np.array(errs)


def solve_segment(segment: list[dict]) -> dict | None:
    """Triangulate the 9 grid points of one placement; return scale stats."""
    resolve_ordering(segment)
    focal = segment[0]["focal"]
    thresh_norm = MAX_REPROJ_PX / focal

    pts3d = {}
    inlier_counts = []
    for k in range(9):
        observations = [(s["norm_xy"][k], s["P"]) for s in segment]
        X = triangulate_dlt(observations)
        if X is None:
            continue
        errs = reproj_errors(X, observations)
        keep = [o for o, e in zip(observations, errs) if e < thresh_norm]
        if len(keep) >= 2:
            X2 = triangulate_dlt(keep)
            if X2 is not None and np.all(reproj_errors(X2, keep) < thresh_norm):
                X = X2
        final_inliers = int(np.sum(reproj_errors(X, observations) < thresh_norm))
        if final_inliers < 2:
            continue
        pts3d[k] = X
        inlier_counts.append(final_inliers)

    if len(pts3d) < 6:
        return None

    scales = []
    for a, b in itertools.combinations(sorted(pts3d), 2):
        d_model = float(np.linalg.norm(pts3d[a] - pts3d[b]))
        d_cm = float(np.linalg.norm(GRID_CM[a] - GRID_CM[b]))
        if d_model > 1e-9:
            scales.append(d_cm / d_model)
    scales = np.array(scales)
    cm_per_unit = float(np.median(scales))
    spread_pct = float(100 * (np.percentile(scales, 90) - np.percentile(scales, 10))
                       / cm_per_unit)

    P9 = np.stack(list(pts3d.values()))
    _, sv, _ = np.linalg.svd(P9 - P9.mean(axis=0))
    planarity_rms_cm = float(sv[-1] / np.sqrt(len(P9)) * cm_per_unit)

    return dict(
        frames=[s["name"] for s in segment],
        n_views=len(segment),
        mean_inlier_views=round(float(np.mean(inlier_counts)), 1),
        points_triangulated=len(pts3d),
        cm_per_unit=cm_per_unit,
        spread_pct=round(spread_pct, 2),
        planarity_rms_cm=round(planarity_rms_cm, 3),
        marker_points_model={str(k): pts3d[k].tolist() for k in sorted(pts3d)},
    )


INVALID_P3D = 2**63 - 1


def depth_ratio_scale(rec: pycolmap.Reconstruction,
                      sightings: list[dict]) -> dict | None:
    """
    Fallback scale for videos where the marker is small/distant and its own
    triangulation is ill-conditioned (e.g. low-res wide-orbit clips).

    Per frame, the pinhole relation px_per_cm = focal / depth gives the
    marker-plane distance in CM, while the SfM points falling inside the
    marker footprint give the same distance in MODEL UNITS. Their ratio is
    cm_per_unit. This leans on well-triangulated wall points and the large
    camera baseline instead of the marker's tiny parallax.

    A robust "tightest cluster" estimate rejects frames whose near-marker
    points actually lie on distant background.
    """
    name_to_img = {im.name: im for im in rec.images.values()}
    ratios = []
    for s in sightings:
        img = name_to_img[s["name"]]
        cam = rec.cameras[img.camera_id]
        focal = cam.params[0]
        P = np.asarray(img.cam_from_world().matrix())
        mcx, mcy = s["pts"][4]
        radius = 0.6 * float(np.ptp(s["pts"], axis=0).max())   # marker footprint
        depths = []
        for p2d in img.points2D:
            pid = p2d.point3D_id
            if pid == INVALID_P3D or pid not in rec.points3D:
                continue
            if np.hypot(p2d.xy[0] - mcx, p2d.xy[1] - mcy) < radius:
                Xc = P @ np.append(rec.points3D[pid].xyz, 1.0)
                if Xc[2] > 0:
                    depths.append(Xc[2])
        if len(depths) < 4:
            continue
        ratios.append((focal / s["px_per_cm"]) / float(np.median(depths)))

    if len(ratios) < 4:
        return None
    r = np.array(sorted(ratios))
    k = max(int(0.5 * len(r)), 3)                  # tightest 50% window = mode
    _, best = min((r[i + k - 1] - r[i], i) for i in range(len(r) - k + 1))
    core = r[best:best + k]
    cm_per_unit = float(np.median(core))
    cv_pct = float(np.std(core) / np.mean(core) * 100)
    return dict(cm_per_unit=cm_per_unit, frames_used=len(r),
                core_cv_pct=round(cv_pct, 2))


def main():
    parser = argparse.ArgumentParser(description="Solve metric scale of a COLMAP model via the marker")
    parser.add_argument("model_dir", type=Path, help="Sparse model dir (e.g. output/sfm/X/sparse/1)")
    parser.add_argument("frames_dir", type=Path, help="Directory with the registered frames")
    args = parser.parse_args()

    rec = pycolmap.Reconstruction(str(args.model_dir))
    print(f"Model: {rec.num_reg_images()} images, {rec.num_points3D()} points")

    sightings = detect_in_registered_frames(rec, args.frames_dir)
    segments = split_segments(sightings)
    print(f"Sightings split into {len(segments)} temporal segments "
          f"(marker may be moved between placements)\n")

    accepted, rejected = [], []
    for i, seg in enumerate(segments):
        if len(seg) < MIN_SEGMENT_VIEWS:
            print(f"segment {i}: {len(seg)} views — skipped (too few)")
            continue
        result = solve_segment(seg)
        tag = f"segment {i}: {len(seg)} views [{seg[0]['name']} .. {seg[-1]['name']}]"
        if result is None:
            print(f"{tag} — triangulation failed")
            continue
        if result["spread_pct"] > MAX_SEGMENT_SPREAD:
            print(f"{tag} — REJECTED: scale={result['cm_per_unit']:.4f} "
                  f"spread={result['spread_pct']:.1f}%")
            rejected.append(result)
            continue
        print(f"{tag} — scale={result['cm_per_unit']:.4f} cm/unit "
              f"spread={result['spread_pct']:.1f}% "
              f"planarity={result['planarity_rms_cm']:.2f}cm")
        accepted.append(result)

    if not accepted:
        print("\nMarker triangulation gave no reliable segment — "
              "trying depth-ratio fallback (small/distant marker)...")
        fb = depth_ratio_scale(rec, sightings)
        if fb is None:
            print("FAILED: no segment produced a reliable scale")
            return
        print(f"Depth-ratio scale : {fb['cm_per_unit']:.4f} cm/unit  "
              f"(frames={fb['frames_used']}, core CV={fb['core_cv_pct']}%)")
        out_path = args.model_dir.parent.parent / "scale.json"
        out_path.write_text(json.dumps(dict(
            model_dir=str(args.model_dir),
            cm_per_unit=fb["cm_per_unit"],
            method="depth_ratio_fallback",
            depth_ratio_frames=fb["frames_used"],
            depth_ratio_core_cv_pct=fb["core_cv_pct"],
            segments=[],
        ), indent=2))
        print(f"Saved {out_path}")
        return

    final_scale = float(np.median([r["cm_per_unit"] for r in accepted]))
    seg_scales = [round(r["cm_per_unit"], 4) for r in accepted]
    cross_spread = (100 * (max(seg_scales) - min(seg_scales)) / final_scale
                    if len(seg_scales) > 1 else 0.0)

    print(f"\nFinal scale       : {final_scale:.4f} cm per model unit")
    print(f"Accepted segments : {len(accepted)}  scales: {seg_scales}")
    if len(seg_scales) > 1:
        print(f"Cross-segment agreement: {cross_spread:.2f}% spread "
              f"(independent placements — this is the real accuracy check)")

    out_path = args.model_dir.parent.parent / "scale.json"
    out_path.write_text(json.dumps(dict(
        model_dir=str(args.model_dir),
        cm_per_unit=final_scale,
        method="marker_triangulation",
        cross_segment_spread_pct=round(cross_spread, 3),
        segments_accepted=len(accepted),
        segments=accepted,
    ), indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
