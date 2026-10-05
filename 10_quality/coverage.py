"""
coverage.py — scan-completeness scoring + re-shoot suggestions.

Ported from the sibling photogram project's coverage.py: score every point
by (a) how many registered cameras can actually see it (occlusion-aware,
via open3d's Hidden Point Removal) and (b) how face-on those views were
(cosine between the surface normal and the camera ray). Colors the cloud
red->yellow->green, and DBSCANs the low-scoring points into "go re-film
here" suggestions with a camera position + direction + wall label.

Room-only (this repo doesn't have photogram's object/outdoor scan modes,
so those branches — orbit-hull clipping, object surface labels, angular
dedup — aren't ported).

Usage:
  python 10_quality/coverage.py output/sfm/IMG_3126 --model 0
  python 10_quality/coverage.py output/sfm/IMG_3126 --model 0 --dense-ply output/sfm/IMG_3126/dense/fused.ply
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap

_SUBSAMPLE_TARGET = 500_000       # voxel-downsample above this before scoring
_SUGGESTION_STANDOFF_M = 1.5      # suggested camera distance from a low-coverage surface
MIN_CLUSTER_FRACTION = 0.005      # keep clusters with >= 0.5% of low-coverage points
MAX_SUGGESTIONS = 6


def _hpr_camera_count(n_cameras: int) -> int:
    return max(25, min(150, int(n_cameras * 0.35)))


def _score_to_rgb(s: np.ndarray) -> np.ndarray:
    r = np.clip(2 * (1 - s), 0, 1)
    g = np.clip(2 * s, 0, 1)
    b = np.zeros_like(s)
    return np.stack([r, g, b], axis=-1)


def _camera_params(rec: pycolmap.Reconstruction, img) -> dict | None:
    cam = rec.cameras[img.camera_id]
    params = cam.params
    model = cam.model.name
    if model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE", "RADIAL") and len(params) >= 3:
        f, cx, cy = float(params[0]), float(params[1]), float(params[2])
        fx = fy = f
    elif model in ("PINHOLE", "OPENCV") and len(params) >= 4:
        fx, fy, cx, cy = float(params[0]), float(params[1]), float(params[2]), float(params[3])
    elif len(params) >= 3:
        f, cx, cy = float(params[0]), float(params[1]), float(params[2])
        fx = fy = f
    else:
        return None
    Rt = np.asarray(img.cam_from_world().matrix())   # 3x4 world->cam
    R, t = Rt[:, :3], Rt[:, 3]
    cam_loc = (-R.T @ t).astype(np.float64)
    return dict(fx=fx, fy=fy, cx=cx, cy=cy, width=int(cam.width), height=int(cam.height),
               Rt=Rt, cam_loc=cam_loc)


def run_coverage(pcd_in: "o3d.geometry.PointCloud", rec: pycolmap.Reconstruction,
                 gravity_up: np.ndarray | None) -> tuple["o3d.geometry.PointCloud", float, list[dict], dict]:
    """Returns (colored_pcd, coverage_score, suggestions, diagnostics)."""
    n_full = len(pcd_in.points)
    if n_full > _SUBSAMPLE_TARGET:
        bbox_diag = float(np.linalg.norm(
            np.asarray(pcd_in.get_max_bound()) - np.asarray(pcd_in.get_min_bound())
        ))
        voxel = max(bbox_diag / (_SUBSAMPLE_TARGET ** 0.5), 0.001)
        pcd = pcd_in.voxel_down_sample(voxel)
        print(f"Voxel-downsampled for scoring: {n_full:,} -> {len(pcd.points):,} points")
    else:
        pcd = pcd_in

    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    n_pts = len(pts)

    scene_scale = float(np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
    ))
    hpr_radius = scene_scale * 100.0
    print(f"Scene scale={scene_scale:.4f}, HPR radius={hpr_radius:.2f}")

    view_counts = np.zeros(n_pts, dtype=np.float64)
    angle_cos_sum = np.zeros(n_pts, dtype=np.float64)

    all_images = list(rec.images.values())
    max_hpr = _hpr_camera_count(len(all_images))
    if len(all_images) > max_hpr:
        step = len(all_images) / max_hpr
        all_images = [all_images[int(round(step * i))] for i in range(max_hpr)]
    n_cameras = len(all_images)

    for cam_idx, img in enumerate(all_images):
        cam = _camera_params(rec, img)
        if cam is None:
            continue
        cam_loc, Rt = cam["cam_loc"], cam["Rt"]
        fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        width, height = cam["width"], cam["height"]

        pts_h = np.hstack([pts, np.ones((n_pts, 1))])
        Xc = (Rt @ pts_h.T).T
        depth = Xc[:, 2]
        u = fx * Xc[:, 0] / np.maximum(depth, 1e-9) + cx
        v = fy * Xc[:, 1] / np.maximum(depth, 1e-9) + cy
        frustum_mask = (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        frustum_idx = np.where(frustum_mask)[0]
        if len(frustum_idx) == 0:
            continue

        pcd_frustum = pcd.select_by_index(frustum_idx.tolist())
        try:
            _, hpr_pt_map = pcd_frustum.hidden_point_removal(cam_loc, hpr_radius)
            visible_local = np.array(hpr_pt_map, dtype=np.int64)
        except Exception as e:
            print(f"  HPR failed for camera {cam_idx} ({e}); using frustum only")
            visible_local = np.arange(len(frustum_idx))
        visible_global = frustum_idx[visible_local]

        view_counts[visible_global] += 1
        rays = pts[visible_global] - cam_loc
        ray_norms = np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-9)
        rays_unit = rays / ray_norms
        nrms = normals[visible_global]
        nrm_norms = np.maximum(np.linalg.norm(nrms, axis=1, keepdims=True), 1e-9)
        nrms_unit = nrms / nrm_norms
        cos_angles = np.clip(np.einsum("ij,ij->i", -rays_unit, nrms_unit), 0.0, 1.0)
        angle_cos_sum[visible_global] += cos_angles

        if (cam_idx + 1) % max(1, n_cameras // 10) == 0 or cam_idx + 1 == n_cameras:
            print(f"  camera {cam_idx+1}/{n_cameras}")

    p95 = np.percentile(view_counts, 95)
    if p95 < 1e-9:
        p95 = max(float(view_counts.max()), 1.0)
    view_count_score = np.clip(view_counts / p95, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        angle_score = np.where(view_counts > 0, angle_cos_sum / view_counts, 0.0)
    score = 0.6 * view_count_score + 0.4 * angle_score
    coverage_score = float(np.mean(score))
    print(f"Coverage score: {coverage_score:.3f}")

    pcd.colors = o3d.utility.Vector3dVector(_score_to_rgb(score))

    # ── DBSCAN shot suggestions ────────────────────────────────────────────
    suggestions: list[dict] = []
    low_idx = np.where(score < 0.4)[0]
    if len(low_idx) >= 10:
        from sklearn.cluster import DBSCAN

        scene_diag = float(np.linalg.norm(
            np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
        ))
        eps = scene_diag * 0.02
        for _ in range(3):
            n_test = len(set(DBSCAN(eps=eps, min_samples=5).fit(pts[low_idx]).labels_) - {-1})
            if n_test >= 3 or eps < scene_diag * 0.005:
                break
            eps *= 0.6
        db = DBSCAN(eps=eps, min_samples=5).fit(pts[low_idx])
        labels = db.labels_
        n_clusters = len(set(labels) - {-1})
        print(f"DBSCAN eps={eps:.4f} -> {n_clusters} cluster(s)")

        unique_labels = [lb for lb in set(labels) if lb != -1]
        cluster_sizes = [(lb, int(np.sum(labels == lb))) for lb in unique_labels]
        cluster_sizes.sort(key=lambda x: -x[1])
        min_cluster_pts = max(5, int(len(low_idx) * MIN_CLUSTER_FRACTION))
        cluster_sizes = [(lb, sz) for lb, sz in cluster_sizes if sz >= min_cluster_pts]

        raw_suggestions = []
        if len(cluster_sizes) <= 1 and len(low_idx) > 100:
            # One big blob — bucket by dominant surface-normal direction instead
            # (floor/ceiling/walls), same fallback photogram uses.
            low_normals = normals[low_idx]
            low_pts_arr = pts[low_idx]
            principal = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float)
            dots = low_normals @ principal.T
            bucket = dots.argmax(axis=1)
            if gravity_up is not None:
                gup = gravity_up / (np.linalg.norm(gravity_up) + 1e-9)
                gravity_dots = [float(np.dot(gup, p)) for p in principal]
                up_idx, down_idx = int(np.argmax(gravity_dots)), int(np.argmin(gravity_dots))
                names = ["+X wall", "-X wall", "+Y surface", "-Y surface", "+Z wall", "-Z wall"]
                names[up_idx], names[down_idx] = "ceiling", "floor"
            else:
                names = ["+X wall", "-X wall", "horizontal surface", "horizontal surface", "+Z wall", "-Z wall"]
            for b_idx, s_name in enumerate(names):
                mask = bucket == b_idx
                if mask.sum() < min_cluster_pts:
                    continue
                b_pts, b_normals = low_pts_arr[mask], low_normals[mask]
                centroid = b_pts.mean(axis=0)
                avg_normal = b_normals.mean(axis=0)
                n_norm = np.linalg.norm(avg_normal)
                avg_normal = avg_normal / n_norm if n_norm > 1e-9 else principal[b_idx]
                raw_suggestions.append(dict(
                    centroid=centroid, cam_pos=centroid + _SUGGESTION_STANDOFF_M * avg_normal,
                    direction=avg_normal, cluster_size=int(mask.sum()),
                    pct_low=mask.sum() / max(1, len(low_idx)), surface=s_name,
                    cluster_pts_idx=low_idx[mask],
                ))
        else:
            for lb, cluster_size in cluster_sizes:
                cluster_mask = labels == lb
                cluster_pts_idx = low_idx[cluster_mask]
                cluster_pts = pts[cluster_pts_idx]
                centroid = cluster_pts.mean(axis=0)
                cluster_normals = normals[cluster_pts_idx]
                avg_normal = cluster_normals.mean(axis=0)
                n_norm = np.linalg.norm(avg_normal)
                avg_normal = avg_normal / n_norm if n_norm > 1e-9 else np.array([0.0, 0.0, 1.0])
                raw_suggestions.append(dict(
                    centroid=centroid, cam_pos=centroid + _SUGGESTION_STANDOFF_M * avg_normal,
                    direction=avg_normal, cluster_size=cluster_size,
                    pct_low=cluster_size / max(1, len(low_idx)), surface=None,
                    cluster_pts_idx=cluster_pts_idx,
                ))

        raw_suggestions = raw_suggestions[:MAX_SUGGESTIONS]

        # Greedy nearest-neighbor ordering — one continuous walking path.
        if len(raw_suggestions) > 1:
            remaining = list(range(1, len(raw_suggestions)))
            ordered = [raw_suggestions[0]]
            while remaining:
                last = ordered[-1]["cam_pos"]
                best_i = min(remaining, key=lambda i: np.linalg.norm(raw_suggestions[i]["cam_pos"] - last))
                ordered.append(raw_suggestions[best_i])
                remaining.remove(best_i)
            raw_suggestions = ordered

        for s in raw_suggestions:
            cluster_pts_idx = s["cluster_pts_idx"]
            mean_views = float(view_counts[cluster_pts_idx].mean()) if len(cluster_pts_idx) else 0.0
            quality_issue = mean_views >= 1.0
            pct = s["pct_low"] * 100
            where = f"the {s['surface']}" if s["surface"] else "this area"
            if quality_issue:
                msg = (f"{where.capitalize()}: {s['cluster_size']:,} pts ({pct:.0f}% of under-covered) "
                      f"— filmed but coverage is poor. Shoot closer, slower, or from more angles.")
            else:
                msg = (f"{where.capitalize()}: {s['cluster_size']:,} pts ({pct:.0f}% of under-covered) "
                      f"— appears unfilmed. Aim at this spot from the suggested position.")
            suggestions.append({
                "position": [float(v) for v in s["cam_pos"]],
                "direction": [float(v) for v in s["direction"]],
                "surface": s["surface"],
                "cluster_size": s["cluster_size"],
                "message": msg,
            })

    diagnostics = dict(
        n_points=n_pts, n_points_full=n_full,
        n_cameras_used=n_cameras, n_low_coverage=int(len(low_idx)), n_suggestions=len(suggestions),
    )
    return pcd, coverage_score, suggestions, diagnostics


def main():
    parser = argparse.ArgumentParser(description="Scan-completeness scoring + re-shoot suggestions")
    parser.add_argument("sfm_dir", type=Path)
    parser.add_argument("--model", default="0", help="Sparse model index (default 0)")
    parser.add_argument("--dense-ply", type=Path, default=None,
                        help="Score a dense MVS cloud (sfm_dir/dense/fused.ply) instead of the "
                             "sparse SfM cloud — more points, more representative coverage score.")
    parser.add_argument("--scale-json", type=Path, default=None)
    args = parser.parse_args()

    scale_path = args.scale_json or (args.sfm_dir / "scale.json")
    scale_info = json.loads(scale_path.read_text())
    cm_per_unit = scale_info["cm_per_unit"]

    rec = pycolmap.Reconstruction(str(args.sfm_dir / "sparse" / args.model))
    print(f"Model: {rec.num_reg_images()} images, {rec.num_points3D()} points")

    if args.dense_ply and args.dense_ply.exists():
        pcd = o3d.io.read_point_cloud(str(args.dense_ply))
        print(f"Dense cloud: {len(pcd.points):,} points")
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "05_geometry"))
        from room_dims import load_filtered_points
        pts = load_filtered_points(rec)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        print(f"Sparse cloud (dense unavailable): {len(pts):,} points")

    # Points must be metric-scaled for the 1.5m suggestion standoff and HPR
    # radius heuristic (both physically-meaningful distances) to make sense.
    pts_m = np.asarray(pcd.points) * (cm_per_unit / 100.0)
    pcd.points = o3d.utility.Vector3dVector(pts_m)

    gravity_up = None
    dims_path = args.sfm_dir / "room_dims.json"
    if dims_path.exists():
        # room_dims.py doesn't persist the gravity rotation directly, but we can
        # recompute the same camera-based prior cheaply for suggestion labeling.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "05_geometry"))
        from room_dims import up_from_cameras
        gravity_up = up_from_cameras(rec)

    colored_pcd, score, suggestions, diagnostics = run_coverage(pcd, rec, gravity_up)

    export_dir = args.sfm_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    colored_path = export_dir / "coverage_colored.ply"
    o3d.io.write_point_cloud(str(colored_path), colored_pcd)
    print(f"Saved {colored_path}")

    for s in suggestions:
        print(f"  - {s['message']}")

    out = dict(coverage_score=round(score, 4), suggestions=suggestions,
              colored_cloud=str(colored_path), **diagnostics)
    out_path = args.sfm_dir / "coverage.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
