"""
fix_trajectory_jumps.py — detect and correct mis-registered "teleport" blocks
caused by a tracking break during SfM.

This is the failure mode the project's own results already flagged (README,
IMG_3126 SfM baseline): a camera that briefly enters a small, mostly
featureless space (a doorway alcove, a gap behind equipment) and loses
tracking can have the REST of its frames re-anchored to a near-duplicate of
nearby geometry, offset by a rigid transform — e.g. the same pipe run
reconstructed twice, a few units apart. That inflates room_dims' footprint
and can put a placement candidate on a wall that doesn't really extend that
far.

Detection: two temporally-consecutive registered frames (no gap in the
source video) whose camera centres are implausibly far apart, yet whose
viewing directions are nearly identical — a spatial "teleport" with no
corresponding change in where the camera was looking. A real turn/pan
changes viewing direction; a tracking-break re-anchor usually doesn't.

Correction, per confirmed jump:
  1. Estimate a correction transform G from trajectory continuity — assume
     the camera's motion from the frame before the jump to the jump frame
     resembles its motion in the preceding step.
  2. Refine G with ICP between the sparse points seen ONLY by the "after"
     block and the sparse points seen ONLY by the "before" block (track-
     based visibility — exact, no distance heuristics needed at the sparse
     stage). A high ICP fitness confirms "after" is a duplicate of existing
     geometry, not new content — low fitness means leave it alone.
  3. Apply the refined transform to those points3D and to the poses of the
     "after" block's registered frames, then write a corrected sparse model
     to <sfm_dir>/sparse/<model>_fixed/. If a dense cloud exists (see
     run_mvs.py) at <sfm_dir>/dense/fused.ply, also reposition the dense
     points that fall nearer the "after" sparse cluster than the "before"
     one (same technique, now needed since dense points carry no visibility
     track).

This mirrors the equivalent stage in the sibling `photogram` project
(backend/workers/pipeline/{sfm.py's _detect_trajectory_jumps,
trajectory_correction.py}) — same algorithm, ported here to mutate a
pycolmap.Reconstruction in place (via Frame.rig_from_world / Point3D.xyz)
rather than a serialized dense-cloud-only representation, since this
project's downstream stages (room_dims.py, placement_3d.py) consume the
sparse model directly.

Usage:
  python 03_reconstruction/fix_trajectory_jumps.py output/sfm/IMG_3126 dataset/frames/IMG_3126 --model 0
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

_ICP_FITNESS_THRESHOLD = 0.5     # below this, treat "after" as real content, not a duplicate
_ICP_DIST_THRESHOLD_M = 0.5      # ICP correspondence search radius, in the model's own units
_CLASSIFY_RADIUS_M = 0.3         # dense points within this of an "after" sparse point are candidates


def _world_from_cam(img) -> np.ndarray:
    m = img.cam_from_world().matrix()   # 3x4, world -> cam
    R, t = m[:, :3], m[:, 3]
    W = np.eye(4)
    W[:3, :3] = R.T
    W[:3, 3] = -R.T @ t
    return W


def detect_trajectory_jumps(rec, all_names: list[str]) -> list[dict]:
    """Detect mis-registered frame blocks. `all_names` must be every sampled
    frame in true temporal order (registered or not) — adjacency in this
    list is what "consecutive in the source video" means."""
    name_to_img = {img.name: img for img in rec.images.values()}

    centers: dict[str, np.ndarray] = {}
    forwards: dict[str, np.ndarray] = {}
    for name, img in name_to_img.items():
        m = img.cam_from_world().matrix()
        R, t = m[:, :3], m[:, 3]
        centers[name] = -R.T @ t
        forwards[name] = R.T @ np.array([0.0, 0.0, 1.0])   # COLMAP: camera looks down +Z in cam space

    steps = []
    for prev_name, next_name in zip(all_names, all_names[1:]):
        if prev_name in centers and next_name in centers:
            dist = float(np.linalg.norm(centers[next_name] - centers[prev_name]))
            steps.append((prev_name, next_name, dist))

    if len(steps) < 5:
        return []

    dists = np.array([s[2] for s in steps])
    median_step = float(np.median(dists))
    if median_step <= 1e-6:
        return []

    jumps: list[dict] = []
    for prev_name, next_name, dist in steps:
        if dist < max(2.0, 15 * median_step):
            continue
        f_prev = forwards[prev_name] / (np.linalg.norm(forwards[prev_name]) + 1e-9)
        f_next = forwards[next_name] / (np.linalg.norm(forwards[next_name]) + 1e-9)
        view_similarity = float(np.dot(f_prev, f_next))
        if view_similarity < 0.97:
            continue   # a real turn, not a teleport
        jumps.append({
            "before_frame": prev_name,
            "after_frame": next_name,
            "jump_distance": round(dist, 3),
            "median_step": round(median_step, 3),
            "view_similarity": round(view_similarity, 3),
        })
    return jumps


def correct_jumps(rec, all_names: list[str], jumps: list[dict],
                  dense_ply_path: Path | None):
    """Mutates `rec` in place for confirmed jumps. Returns (applied, skipped, dense_pcd_or_None)."""
    import open3d as o3d
    import pycolmap

    name_to_img = {img.name: img for img in rec.images.values()}
    name_to_idx = {n: i for i, n in enumerate(all_names)}

    dense_pcd = None
    dense_pts = dense_normals = dense_colors = None
    if dense_ply_path and dense_ply_path.exists():
        dense_pcd = o3d.io.read_point_cloud(str(dense_ply_path))
        dense_pts = np.asarray(dense_pcd.points)
        dense_normals = np.asarray(dense_pcd.normals) if dense_pcd.has_normals() else None
        dense_colors = np.asarray(dense_pcd.colors) if dense_pcd.has_colors() else None

    applied, skipped = [], []

    for ji, jump in enumerate(jumps):
        before_frame, after_frame = jump["before_frame"], jump["after_frame"]

        if before_frame not in name_to_img or after_frame not in name_to_img:
            skipped.append({**jump, "reason": "frame not registered"})
            continue

        before_idx, after_idx = name_to_idx[before_frame], name_to_idx[after_frame]

        block_end_idx = len(all_names) - 1
        for other in jumps[ji + 1:]:
            if other["before_frame"] in name_to_idx:
                block_end_idx = name_to_idx[other["before_frame"]]
                break

        if before_idx - 1 < 0:
            skipped.append({**jump, "reason": "no preceding frame for continuity estimate"})
            continue
        prev_name = all_names[before_idx - 1]
        if prev_name not in name_to_img:
            skipped.append({**jump, "reason": "preceding frame not registered"})
            continue

        W_prev = _world_from_cam(name_to_img[prev_name])
        W_before = _world_from_cam(name_to_img[before_frame])
        W_after = _world_from_cam(name_to_img[after_frame])
        M = np.linalg.inv(W_prev) @ W_before
        W_after_expected = W_before @ M
        G = W_after_expected @ np.linalg.inv(W_after)

        after_ids = {
            name_to_img[all_names[i]].image_id
            for i in range(after_idx, block_end_idx + 1)
            if all_names[i] in name_to_img
        }
        before_ids = {
            img.image_id for name, img in name_to_img.items()
            if name_to_idx.get(name, -1) not in range(after_idx, block_end_idx + 1)
        }

        after_pt_ids, before_pt_ids = [], []
        after_pts_list, before_pts_list = [], []
        for pid, p3d in rec.points3D.items():
            track_ids = {el.image_id for el in p3d.track.elements}
            in_after = bool(track_ids & after_ids)
            in_before = bool(track_ids & before_ids)
            if in_after and not in_before:
                after_pt_ids.append(pid)
                after_pts_list.append(p3d.xyz)
            elif in_before and not in_after:
                before_pt_ids.append(pid)
                before_pts_list.append(p3d.xyz)

        if len(after_pts_list) < 50 or len(before_pts_list) < 50:
            skipped.append({**jump, "reason": "not enough exclusive sparse points to validate"})
            continue

        before_pts_sparse = np.asarray(before_pts_list)
        after_pts_sparse = np.asarray(after_pts_list)

        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(after_pts_sparse)
        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(before_pts_sparse)
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, _ICP_DIST_THRESHOLD_M, G,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )

        if reg.fitness < _ICP_FITNESS_THRESHOLD:
            skipped.append({
                **jump,
                "reason": f"ICP fitness {reg.fitness:.2f} below threshold "
                          f"{_ICP_FITNESS_THRESHOLD} — likely real new content, not a duplicate",
                "icp_fitness": round(reg.fitness, 3),
            })
            continue

        G_refined = np.asarray(reg.transformation)
        R_g, t_g = G_refined[:3, :3], G_refined[:3, 3]

        # ── Apply to sparse points3D (exact — track visibility already told
        # us these belong to the after-block, no distance heuristic needed) ──
        for pid in after_pt_ids:
            xyz = np.asarray(rec.points3D[pid].xyz)
            rec.points3D[pid].xyz = R_g @ xyz + t_g

        # ── Apply to the after-block's camera poses ───────────────────────
        n_cameras_corrected = 0
        for i in range(after_idx, block_end_idx + 1):
            name = all_names[i]
            if name not in name_to_img:
                continue
            img = name_to_img[name]
            W = _world_from_cam(img)
            W_corrected = G_refined @ W
            cam_from_world_corrected = np.linalg.inv(W_corrected)[:3, :]
            img.frame.rig_from_world = pycolmap.Rigid3d(cam_from_world_corrected)
            n_cameras_corrected += 1

        n_dense_corrected = 0
        if dense_pts is not None:
            margin = _CLASSIFY_RADIUS_M * 2
            bbox_min = after_pts_sparse.min(axis=0) - margin
            bbox_max = after_pts_sparse.max(axis=0) + margin
            in_bbox = np.all((dense_pts >= bbox_min) & (dense_pts <= bbox_max), axis=1)
            candidate_idx = np.where(in_bbox)[0]
            if len(candidate_idx):
                after_tree = o3d.geometry.KDTreeFlann(src)
                before_tree = o3d.geometry.KDTreeFlann(tgt)
                to_transform = []
                for idx in candidate_idx:
                    p = dense_pts[idx]
                    _, _, d_after = after_tree.search_knn_vector_3d(p, 1)
                    _, _, d_before = before_tree.search_knn_vector_3d(p, 1)
                    if np.sqrt(d_after[0]) < _CLASSIFY_RADIUS_M and d_after[0] < d_before[0]:
                        to_transform.append(idx)
                if to_transform:
                    to_transform = np.array(to_transform)
                    dense_pts[to_transform] = (R_g @ dense_pts[to_transform].T).T + t_g
                    if dense_normals is not None:
                        dense_normals[to_transform] = (R_g @ dense_normals[to_transform].T).T
                    n_dense_corrected = len(to_transform)

        applied.append({
            **jump,
            "icp_fitness": round(reg.fitness, 3),
            "icp_inlier_rmse": round(reg.inlier_rmse, 4),
            "n_sparse_points_corrected": len(after_pt_ids),
            "n_dense_points_corrected": n_dense_corrected,
            "n_cameras_corrected": n_cameras_corrected,
            "block_start": after_frame,
            "block_end": all_names[block_end_idx],
        })

    if dense_pcd is not None and dense_pts is not None:
        dense_pcd.points = o3d.utility.Vector3dVector(dense_pts)
        if dense_normals is not None:
            dense_pcd.normals = o3d.utility.Vector3dVector(dense_normals)
        if dense_colors is not None:
            dense_pcd.colors = o3d.utility.Vector3dVector(dense_colors)

    return applied, skipped, dense_pcd


def main():
    parser = argparse.ArgumentParser(description="Detect + correct SfM trajectory jumps")
    parser.add_argument("sfm_dir", type=Path)
    parser.add_argument("frames_dir", type=Path, help="Frame images dir used for SfM "
                                                       "(needed for true temporal adjacency)")
    parser.add_argument("--model", default="0", help="Sparse model index (default 0)")
    args = parser.parse_args()

    import pycolmap

    model_dir = args.sfm_dir / "sparse" / args.model
    if not (model_dir / "images.bin").exists():
        raise SystemExit(f"fix_trajectory_jumps: no reconstruction at {model_dir}")

    rec = pycolmap.Reconstruction(str(model_dir))
    all_names = sorted(p.name for p in args.frames_dir.glob("*.jpg"))
    print(f"Model: {rec.num_reg_images()} registered / {len(all_names)} sampled frames, "
          f"{rec.num_points3D()} points")

    jumps = detect_trajectory_jumps(rec, all_names)
    out = {"jumps_detected": jumps, "jumps_applied": [], "jumps_skipped": []}

    if not jumps:
        print("No trajectory jumps detected.")
        (args.sfm_dir / "trajectory_jumps.json").write_text(json.dumps(out, indent=2))
        return

    print(f"Found {len(jumps)} candidate jump(s):")
    for j in jumps:
        print(f"  {j['before_frame']} -> {j['after_frame']}: "
              f"jump={j['jump_distance']:.2f} (median step {j['median_step']:.2f}), "
              f"view_similarity={j['view_similarity']:.3f}")

    dense_ply = args.sfm_dir / "dense" / "fused.ply"
    applied, skipped, dense_pcd = correct_jumps(
        rec, all_names, jumps, dense_ply if dense_ply.exists() else None,
    )
    out["jumps_applied"] = applied
    out["jumps_skipped"] = skipped
    (args.sfm_dir / "trajectory_jumps.json").write_text(json.dumps(out, indent=2))

    if not applied:
        print(f"No corrections applied ({len(skipped)} candidate(s) rejected — see reasons in trajectory_jumps.json).")
        return

    fixed_dir = args.sfm_dir / "sparse" / f"{args.model}_fixed"
    shutil.rmtree(fixed_dir, ignore_errors=True)
    fixed_dir.mkdir(parents=True, exist_ok=True)
    rec.write_binary(str(fixed_dir))
    print(f"Corrected {len(applied)} jump(s) — wrote fixed sparse model to {fixed_dir}")

    if dense_pcd is not None:
        import open3d as o3d
        o3d.io.write_point_cloud(str(dense_ply), dense_pcd)
        print(f"Also corrected dense cloud in place: {dense_ply}")


if __name__ == "__main__":
    main()
