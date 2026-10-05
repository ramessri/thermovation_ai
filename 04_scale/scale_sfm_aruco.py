"""
scale_sfm_aruco.py — ArUco-based metric scale for a COLMAP sparse model.

Deliberately mirrors the sibling photogram project's algorithm exactly
(backend/workers/pipeline/{aruco_detector.py, scale_from_aruco.py}) rather
than inventing a thermovation-specific approach:

  1. Per registered frame, detect all ArUco markers (cv2.aruco).
  2. Same-frame marker PAIRS get a physical baseline distance via solvePnP
     (each marker's known real-world size -> a camera-space tvec -> the
     Euclidean distance between two markers' tvecs, in metres). This is a
     per-frame, per-pair physical measurement — it doesn't depend on SfM.
  3. Each marker's centroid is also triangulated in SFM-UNIT space from
     multiple views, via camera-ray least-squares intersection (not corner
     DLT) — same method as photogram's _triangulate_marker.
  4. Scale = physical_baseline_m / sfm_unit_distance, for every marker pair
     that has both a baseline (step 2) and a triangulated SFM distance
     (step 3); final scale is the median of those estimates after a 1.5x
     IQR outlier pass — identical combination logic to derive_scale_factor.

This needs 2+ ArUco markers with at least two co-visible in one frame — a
single marker alone cannot be scaled by this method, same limitation as
photogram (it does not fall back to a known-marker-size / corner-geometry
estimate). Unlike photogram, there's no separate pre-SfM detection stage
here: thermovation_ai's scale scripts run post-SfM only, so baseline and
triangulation are both computed from the already-registered frames using
the SfM-refined camera intrinsics (calibration_matrix()) in one pass —
arguably more accurate than photogram's pre-SfM heuristic-focal-length
baseline measurement, since the intrinsics have already been bundle-adjusted.

Usage:
  python 04_scale/scale_sfm_aruco.py output/sfm/<video>/sparse/0 dataset/frames/<video>
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_calibration"))
from detect_marker_aruco import detect_aruco_markers, ARUCO_MARKER_SIZE_CM, ARUCO_DICT_NAME

MIN_BASELINE_M = 0.30   # matches photogram's scale_from_aruco.py MIN_BASELINE_M


def _marker_obj_points_m(size_cm: float) -> np.ndarray:
    h = (size_cm / 100.0) / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def _solvepnp_tvec(corners_px: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    obj_pts = _marker_obj_points_m(ARUCO_MARKER_SIZE_CM)
    dist = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(obj_pts, corners_px.astype(np.float64), K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return tvec.flatten() if ok else None


def _backproject_centroid_ray(corners_px: np.ndarray, K: np.ndarray,
                              R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t: world->cam (COLMAP convention). Returns (cam_centre_world, ray_direction_world)."""
    cx_px, cy_px = corners_px.mean(axis=0)
    fx, fy = K[0, 0], K[1, 1]
    ppx, ppy = K[0, 2], K[1, 2]
    ray_cam = np.array([(cx_px - ppx) / fx, (cy_px - ppy) / fy, 1.0])
    ray_cam /= np.linalg.norm(ray_cam)
    R_t = R.T
    cam_centre = -R_t @ t
    ray_world = R_t @ ray_cam
    ray_world /= np.linalg.norm(ray_world)
    return cam_centre, ray_world


def _triangulate_ray_intersection(observations: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    """Least-squares closest point to multiple camera rays (photogram's _triangulate_marker)."""
    if len(observations) < 2:
        return None
    A_rows, b_rows = [], []
    for origin, direction in observations:
        M = np.eye(3) - np.outer(direction, direction)
        A_rows.append(M)
        b_rows.append(M @ origin)
    A = np.vstack(A_rows)
    b = np.concatenate(b_rows)
    try:
        pt, *_ = np.linalg.lstsq(A, b, rcond=None)
        return pt
    except Exception:
        return None


def collect_observations(rec: pycolmap.Reconstruction, frames_dir: Path):
    """One pass over registered frames: same-frame physical baselines (solvePnP)
    and per-marker world-ray observations (for SFM-unit triangulation)."""
    marker_observations: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    baselines: list[dict] = []
    n_frames_with_any = 0

    for img in sorted(rec.images.values(), key=lambda im: im.name):
        bgr = cv2.imread(str(frames_dir / img.name))
        if bgr is None:
            continue
        dets = detect_aruco_markers(bgr)
        if not dets:
            continue
        n_frames_with_any += 1

        cam = rec.cameras[img.camera_id]
        K = np.asarray(cam.calibration_matrix())
        m = np.asarray(img.cam_from_world().matrix())
        R, t = m[:, :3], m[:, 3]

        # Same-frame physical baselines (metres, via solvePnP) — mirrors
        # aruco_detector.py's pre-SfM baseline measurement, but computed
        # here with the SfM-refined intrinsics instead of a pre-SfM guess.
        tvecs = {mid: _solvepnp_tvec(corners, K) for mid, corners in dets.items()}
        tvecs = {mid: tv for mid, tv in tvecs.items() if tv is not None}
        ids = sorted(tvecs)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                dist_m = float(np.linalg.norm(tvecs[id_a] - tvecs[id_b]))
                baselines.append({"id_a": id_a, "id_b": id_b, "dist_m": dist_m, "frame": img.name})

        # World-space ray per marker, for post-SfM centroid triangulation.
        for mid, corners in dets.items():
            origin, ray = _backproject_centroid_ray(corners, K, R, t)
            marker_observations.setdefault(mid, []).append((origin, ray))

    print(f"ArUco (dict={ARUCO_DICT_NAME}) detected in {n_frames_with_any} / "
          f"{rec.num_reg_images()} registered frames — "
          f"{len(marker_observations)} distinct marker ID(s): {sorted(marker_observations)}, "
          f"{len(baselines)} same-frame pair baseline(s)")
    return marker_observations, baselines


def main():
    parser = argparse.ArgumentParser(description="Solve metric scale of a COLMAP model via ArUco markers")
    parser.add_argument("model_dir", type=Path, help="Sparse model dir (e.g. output/sfm/X/sparse/0)")
    parser.add_argument("frames_dir", type=Path, help="Directory with the registered frames")
    args = parser.parse_args()

    rec = pycolmap.Reconstruction(str(args.model_dir))
    print(f"Model: {rec.num_reg_images()} images, {rec.num_points3D()} points")

    marker_observations, baselines = collect_observations(rec, args.frames_dir)

    if not marker_observations:
        print(f"FAILED: no ArUco markers (dict={ARUCO_DICT_NAME}) detected in any registered frame. "
              f"Print markers from this dictionary and place them in the scene, or use "
              f"--marker-type custom for the boiler-room board instead.")
        return

    # Triangulate each marker's centroid in SFM-UNIT space (2+ views required).
    marker_world_pos: dict[int, np.ndarray] = {}
    for mid, obs in marker_observations.items():
        pt = _triangulate_ray_intersection(obs)
        if pt is not None:
            marker_world_pos[mid] = pt
    print(f"Triangulated {len(marker_world_pos)} / {len(marker_observations)} marker(s) "
         f"(need 2+ views each): {sorted(marker_world_pos)}")

    if not marker_world_pos:
        print("FAILED: could not triangulate any marker position (need 2+ views per marker ID).")
        return
    if len(marker_world_pos) < 2:
        print(f"FAILED: only 1 marker ({list(marker_world_pos)[0]}) triangulated — this method "
              f"needs 2+ markers co-visible in at least one frame to establish a physical baseline. "
              f"Add a second printed marker to the scene.")
        return

    baseline_lookup: dict[tuple, float] = {}
    for b in baselines:
        key = (min(b["id_a"], b["id_b"]), max(b["id_a"], b["id_b"]))
        baseline_lookup.setdefault(key, []).append(b["dist_m"])
    baseline_lookup = {k: float(np.median(v)) for k, v in baseline_lookup.items()}

    scale_estimates = []
    marker_ids = list(marker_world_pos.keys())
    for i in range(len(marker_ids)):
        for j in range(i + 1, len(marker_ids)):
            id_a, id_b = marker_ids[i], marker_ids[j]
            physical_m = baseline_lookup.get((min(id_a, id_b), max(id_a, id_b)))
            if physical_m is None:
                continue
            if physical_m < MIN_BASELINE_M:
                print(f"  skip ({id_a},{id_b}): baseline {physical_m:.3f}m < {MIN_BASELINE_M}m min")
                continue
            sfm_dist = float(np.linalg.norm(marker_world_pos[id_a] - marker_world_pos[id_b]))
            if sfm_dist < 1e-9:
                continue
            est = physical_m / sfm_dist
            scale_estimates.append(est)
            print(f"  markers ({id_a},{id_b}): sfm={sfm_dist:.4f}  phys={physical_m:.4f}m  scale={est:.5f}")

    if not scale_estimates:
        print("FAILED: triangulated markers but no matching physical baseline "
              "(need 2+ markers visible together in at least one frame).")
        return

    estimates = np.array(scale_estimates)
    n_outliers = 0
    if len(estimates) >= 4:
        q1, q3 = np.percentile(estimates, [25, 75])
        iqr = q3 - q1
        mask = (estimates >= q1 - 1.5 * iqr) & (estimates <= q3 + 1.5 * iqr)
        n_outliers = int((~mask).sum())
        estimates = estimates[mask]

    scale_factor = float(np.median(estimates))   # metres per SfM unit
    cm_per_unit = scale_factor * 100.0
    scale_std = float(np.std(estimates))

    print(f"\nFinal scale : {cm_per_unit:.4f} cm per model unit "
         f"({scale_factor:.6f} m/unit, from {len(estimates)} estimate(s), "
         f"{n_outliers} outlier(s) removed, std={scale_std:.6f})")

    out_path = args.model_dir.parent.parent / "scale.json"
    out_path.write_text(json.dumps(dict(
        model_dir=str(args.model_dir),
        cm_per_unit=cm_per_unit,
        method="aruco_baseline_triangulation",
        aruco_dict=ARUCO_DICT_NAME,
        aruco_marker_size_cm=ARUCO_MARKER_SIZE_CM,
        scale_n_inliers=int(len(estimates)),
        scale_outliers_removed=n_outliers,
        scale_std_m_per_unit=round(scale_std, 6),
        triangulated_marker_ids=sorted(marker_world_pos),
        segments=[],   # room_dims.py reads scale_info.get("segments") for the topdown
                       # render's marker-star overlay; empty here (no per-segment shape
                       # to plot — see marker_points_model note below if needed later).
    ), indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
