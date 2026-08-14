"""
pipe_paths.py — pipe path & length extraction, plus Vor-/Rücklauf pairing.

Reuses placement_3d.py's co-visible-point-lifting mechanism, generalized
from a single pixel (radius-based) to full mask membership, to fuse 2D pipe
segmentation masks into a 3D point cloud across all registered frames.

Usage:
  python 07_pipes/pipe_paths.py output/sfm/IMG_3126 dataset/frames/IMG_3126_sfm
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "06_placement"))
sys.path.insert(0, str(_ROOT / "experiments"))
from placement_3d import BLUE_HSV_LOW, BLUE_HSV_HIGH

INVALID_P3D = 2**63 - 1


def lift_mask_to_3d(rec: pycolmap.Reconstruction, img, mask: np.ndarray) -> list[int]:
    """point3D_ids whose 2D observation in this image falls inside the mask."""
    ids = []
    h, w = mask.shape[:2]
    for p2d in img.points2D:
        pid = p2d.point3D_id
        if pid == INVALID_P3D or pid not in rec.points3D:
            continue
        x, y = int(round(p2d.xy[0])), int(round(p2d.xy[1]))
        if 0 <= y < h and 0 <= x < w and mask[y, x] > 0:
            ids.append(pid)
    return ids


def fuse_masks_to_point_ids(rec: pycolmap.Reconstruction,
                            frame_masks: dict[str, np.ndarray]) -> set[int]:
    """Union of lift_mask_to_3d across every frame that has a mask."""
    name_to_img = {im.name: im for im in rec.images.values()}
    ids: set[int] = set()
    for name, mask in frame_masks.items():
        img = name_to_img.get(name)
        if img is None:
            continue
        ids.update(lift_mask_to_3d(rec, img, mask))
    return ids


def points_xyz_and_color(rec: pycolmap.Reconstruction,
                         point_ids: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """(xyz Nx3 float64, rgb Nx3 uint8) for the given point3D ids."""
    ids = list(point_ids)
    if not ids:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    xyz = np.array([rec.points3D[i].xyz for i in ids])
    rgb = np.array([rec.points3D[i].color for i in ids], dtype=np.uint8)
    return xyz, rgb


def cluster_points_voxel(pts: np.ndarray, voxel_cm: float, cm_per_unit: float,
                         min_cluster_points: int = 15) -> list[np.ndarray]:
    """3D connected-component clustering via voxel occupancy + union-find over
    26-connected occupied voxels (no scipy/sklearn dependency — same spirit as
    room_dims.py's existing voxel/connected-component footprint filter, just
    generalized from 2D to 3D). Returns index arrays into pts, largest first,
    dropping clusters smaller than min_cluster_points."""
    if len(pts) == 0:
        return []
    voxel = voxel_cm / cm_per_unit
    ijk = np.floor(pts / voxel).astype(np.int64)

    voxel_to_indices: dict[tuple, list[int]] = {}
    for idx, key in enumerate(map(tuple, ijk)):
        voxel_to_indices.setdefault(key, []).append(idx)

    parent = {v: v for v in voxel_to_indices}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
              if (dx, dy, dz) != (0, 0, 0)]
    for v in list(voxel_to_indices):
        for off in offsets:
            n = (v[0] + off[0], v[1] + off[1], v[2] + off[2])
            if n in voxel_to_indices:
                union(v, n)

    groups: dict[tuple, list[int]] = {}
    for v, idxs in voxel_to_indices.items():
        root = find(v)
        groups.setdefault(root, []).extend(idxs)

    clusters = [np.array(idxs) for idxs in groups.values() if len(idxs) >= min_cluster_points]
    clusters.sort(key=len, reverse=True)
    return clusters


def dominant_axis(cluster_pts: np.ndarray) -> np.ndarray:
    """Unit vector of the cluster's dominant PCA axis (sign arbitrary)."""
    centered = cluster_pts - cluster_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    return axis / np.linalg.norm(axis)


def cluster_centerline(cluster_pts: np.ndarray, n_bins: int = 12) -> np.ndarray:
    """PCA dominant axis, bin points along it, centroid per non-empty bin.
    The resulting ordered points ARE the piecewise-linear polyline/spline fit
    through the cluster — no separate smoothing library needed at this scale."""
    centroid = cluster_pts.mean(axis=0)
    axis = dominant_axis(cluster_pts)
    t = (cluster_pts - centroid) @ axis
    lo, hi = t.min(), t.max()
    if hi - lo < 1e-9:
        return centroid[None, :]
    bin_idx = np.clip(((t - lo) / (hi - lo) * n_bins).astype(int), 0, n_bins - 1)
    pts_out = []
    for b in range(n_bins):
        m = bin_idx == b
        if m.any():
            pts_out.append(cluster_pts[m].mean(axis=0))
    return np.array(pts_out)


def polyline_length_cm(centerline_pts: np.ndarray, cm_per_unit: float) -> float:
    """Sum of consecutive segment lengths along the ordered centerline."""
    if len(centerline_pts) < 2:
        return 0.0
    diffs = np.diff(centerline_pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1) * cm_per_unit
    return float(seg_lengths.sum())


# Blue range reused from placement_3d.blue_circle_candidates; red needs two
# sub-ranges since OpenCV hue wraps at 0/180.
RED_HSV_RANGES = [((0, 90, 60), (10, 255, 255)), ((170, 90, 60), (180, 255, 255))]


def classify_cluster_color(rgb: np.ndarray, min_fraction: float = 0.25) -> str:
    """rgb: Nx3 uint8 array of a cluster's point colors (from points_xyz_and_color)
    -> 'blue' | 'red' | 'neither', by which HSV range covers the larger
    fraction of points, gated by min_fraction so a mostly-neutral cluster
    (bare metal pipe, insulation) isn't force-classified."""
    if len(rgb) == 0:
        return "neither"
    bgr = rgb[:, ::-1].astype(np.uint8).reshape(1, -1, 3)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).reshape(1, -1, 3)

    blue = cv2.inRange(hsv, BLUE_HSV_LOW, BLUE_HSV_HIGH).reshape(-1) > 0
    red = np.zeros(rgb.shape[0], dtype=bool)
    for lo, hi in RED_HSV_RANGES:
        red |= cv2.inRange(hsv, lo, hi).reshape(-1) > 0

    blue_frac = float(blue.mean())
    red_frac = float(red.mean())
    if blue_frac >= min_fraction and blue_frac > red_frac:
        return "blue"
    if red_frac >= min_fraction and red_frac > blue_frac:
        return "red"
    return "neither"


def pairing_score(blue_centroid: np.ndarray, red_centroid: np.ndarray,
                  blue_axis: np.ndarray, red_axis: np.ndarray,
                  boiler_xyz: np.ndarray | None, cm_per_unit: float,
                  w_dist: float = 1.0, w_parallel: float = 50.0,
                  w_boiler: float = 1.0) -> float:
    """Lower is better (matches placement_3d.py's scoring convention). Combines
    proximity, parallelism (soft — no hard rejection, room layouts vary), and
    boiler-adjacency into one cm-scale score."""
    dist_cm = float(np.linalg.norm(blue_centroid - red_centroid)) * cm_per_unit
    cosang = abs(float(np.dot(blue_axis, red_axis)))
    parallel_penalty = (1.0 - cosang) * w_parallel
    if boiler_xyz is not None:
        boiler_dist_cm = (float(np.linalg.norm(blue_centroid - boiler_xyz)) +
                          float(np.linalg.norm(red_centroid - boiler_xyz))) * cm_per_unit / 2
    else:
        boiler_dist_cm = 0.0
    return w_dist * dist_cm + parallel_penalty + w_boiler * boiler_dist_cm


def find_best_rucklauf_pair(clusters: list[dict], boiler_xyz: np.ndarray | None,
                            cm_per_unit: float) -> dict | None:
    """clusters: [{"id", "centroid", "axis", "color"}, ...]. Returns the
    lowest-scoring blue/red pair, or None if no blue or no red cluster exists."""
    blues = [c for c in clusters if c["color"] == "blue"]
    reds = [c for c in clusters if c["color"] == "red"]
    if not blues or not reds:
        return None
    best = None
    for b in blues:
        for r in reds:
            score = pairing_score(b["centroid"], r["centroid"], b["axis"], r["axis"],
                                  boiler_xyz, cm_per_unit)
            if best is None or score < best["score"]:
                best = dict(rucklauf_id=b["id"], vorlauf_id=r["id"],
                           score=round(score, 1),
                           rucklauf_xyz_model=b["centroid"].tolist())
    return best


def get_masks_for_prompt(detector: str, prompt: str, image_rgb: np.ndarray,
                         models: dict, threshold: float, device: str) -> list[np.ndarray]:
    """Dispatches to whichever detector arm is selected; GDINO/YOLO-World
    route through SAM2, YOLOE returns its own native masks directly."""
    if detector == "gdino":
        from experiment_pipeline import detect_gdino, sam2_from_boxes
        boxes, labels, scores = detect_gdino(image_rgb, models["gdino_proc"], models["gdino_model"],
                                             prompt, threshold, device)
        return sam2_from_boxes(models["sam2"], image_rgb, boxes)
    if detector == "yoloworld":
        from experiment_pipeline import detect_yolo_world, sam2_from_boxes
        boxes, labels, scores = detect_yolo_world(image_rgb, models["yoloworld"], [prompt], threshold)
        return sam2_from_boxes(models["sam2"], image_rgb, boxes)
    if detector == "yoloe":
        from experiment_pipeline import detect_yoloe
        boxes, labels, scores, masks = detect_yoloe(image_rgb, models["yoloe"], [prompt], threshold)
        return masks
    raise ValueError(f"Unknown detector: {detector}")


def load_detector_models(detector: str, device: str) -> dict:
    from experiment_pipeline import load_sam2, load_gdino, load_yolo_world, load_yoloe
    models = {}
    if detector in ("gdino", "yoloworld"):
        models["sam2"] = load_sam2(device)
    if detector == "gdino":
        models["gdino_proc"], models["gdino_model"] = load_gdino(device)
    elif detector == "yoloworld":
        models["yoloworld"] = load_yolo_world(device)
    elif detector == "yoloe":
        models["yoloe"] = load_yoloe(device)
    return models


def main():
    parser = argparse.ArgumentParser(description="Pipe path/length extraction + Vor-/Ruecklauf pairing")
    parser.add_argument("sfm_dir", type=Path)
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("--model", default="1")
    parser.add_argument("--detector", choices=["gdino", "yoloworld", "yoloe"], default="gdino",
                        help="Placeholder default until Foundation's compare_arms picks a winner")
    parser.add_argument("--voxel-cm", type=float, default=8.0)
    parser.add_argument("--min-cluster-points", type=int, default=15)
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scale_info = json.loads((args.sfm_dir / "scale.json").read_text())
    cm_per_unit = scale_info["cm_per_unit"]
    rec = pycolmap.Reconstruction(str(args.sfm_dir / "sparse" / args.model))
    print(f"Model: {rec.num_reg_images()} images | scale {cm_per_unit:.3f} cm/unit")

    models = load_detector_models(args.detector, device)

    # 2D pipe masks over every registered frame
    pipe_frame_masks: dict[str, np.ndarray] = {}
    boiler_frame_masks: dict[str, np.ndarray] = {}
    for img in sorted(rec.images.values(), key=lambda im: im.name):
        bgr = cv2.imread(str(args.frames_dir / img.name))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pipe_masks = get_masks_for_prompt(args.detector, "pipe", rgb, models, args.threshold, device)
        if pipe_masks:
            combined = np.clip(sum(m.squeeze().astype(np.uint8) for m in pipe_masks), 0, 1)
            pipe_frame_masks[img.name] = combined
        boiler_masks = get_masks_for_prompt(args.detector, "boiler", rgb, models, args.threshold, device)
        if boiler_masks:
            boiler_frame_masks[img.name] = boiler_masks[0].squeeze().astype(np.uint8)
    print(f"Pipe masks found in {len(pipe_frame_masks)}/{rec.num_reg_images()} frames; "
          f"boiler masks in {len(boiler_frame_masks)} frames")

    # fuse to 3D
    pipe_ids = fuse_masks_to_point_ids(rec, pipe_frame_masks)
    xyz, rgb_colors = points_xyz_and_color(rec, pipe_ids)
    print(f"Fused pipe cloud: {len(xyz)} points")

    boiler_xyz = None
    if boiler_frame_masks:
        boiler_ids = fuse_masks_to_point_ids(rec, boiler_frame_masks)
        b_xyz, _ = points_xyz_and_color(rec, boiler_ids)
        if len(b_xyz):
            boiler_xyz = b_xyz.mean(axis=0)

    # cluster into pipe instances
    cluster_idx_lists = cluster_points_voxel(xyz, args.voxel_cm, cm_per_unit, args.min_cluster_points)
    print(f"{len(cluster_idx_lists)} pipe instance(s) after clustering")

    pipes_out = []
    pairing_clusters = []
    for cid, idxs in enumerate(cluster_idx_lists):
        cpts = xyz[idxs]
        ccolors = rgb_colors[idxs]
        centerline = cluster_centerline(cpts, args.n_bins)
        length_cm = polyline_length_cm(centerline, cm_per_unit)
        color = classify_cluster_color(ccolors)
        confidence = "reliable" if len(cpts) >= 50 else "low_confidence"
        pipes_out.append(dict(id=cid, length_cm=round(length_cm, 1),
                              n_points=int(len(cpts)), confidence=confidence, color=color))
        pairing_clusters.append(dict(id=cid, centroid=cpts.mean(axis=0),
                                     axis=dominant_axis(cpts), color=color))
        print(f"  pipe {cid}: {length_cm:.1f}cm, {len(cpts)} points, color={color}, {confidence}")

    pair = find_best_rucklauf_pair(pairing_clusters, boiler_xyz, cm_per_unit)
    rucklauf_pipe_pairing = pair if pair else dict(pair_found=False)
    if pair:
        rucklauf_pipe_pairing["pair_found"] = True
        print(f"Vor-/Ruecklauf pair: pipe {pair['rucklauf_id']} (blue) <-> "
              f"pipe {pair['vorlauf_id']} (red), score={pair['score']}")
    else:
        print("No Vor-/Ruecklauf pair found (need at least one blue and one red pipe cluster)")

    out = dict(pipes=pipes_out, rucklauf_pipe_pairing=rucklauf_pipe_pairing)
    out_path = args.sfm_dir / "pipe_lengths.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_path}")

    # visualization: overlay centerlines on the first registered frame that
    # has 2D pipe detections
    if pipe_frame_masks:
        view_name = next(iter(pipe_frame_masks))
        view_img = next(im for im in rec.images.values() if im.name == view_name)
        cam = rec.cameras[view_img.camera_id]
        P = np.asarray(view_img.cam_from_world().matrix())
        bgr = cv2.imread(str(args.frames_dir / view_name))

        def project(X_model: np.ndarray):
            Xc = P @ np.append(X_model, 1.0)
            if Xc[2] <= 0:
                return None
            xy = np.asarray(cam.img_from_cam(Xc[None, :3] / Xc[2]))
            return tuple(np.round(xy.ravel()[:2]).astype(int))

        colors_bgr = {"blue": (255, 100, 0), "red": (0, 0, 255), "neither": (0, 220, 0)}
        for cid, idxs in enumerate(cluster_idx_lists):
            centerline = cluster_centerline(xyz[idxs], args.n_bins)
            color = pipes_out[cid]["color"]
            pts2d = [p for p in (project(x) for x in centerline) if p is not None]
            if len(pts2d) >= 2:
                cv2.polylines(bgr, [np.array(pts2d)], False, colors_bgr[color], 3)
        out_img = args.sfm_dir / "pipe_paths.jpg"
        cv2.imwrite(str(out_img), bgr)
        print(f"Saved {out_img}")


if __name__ == "__main__":
    main()
