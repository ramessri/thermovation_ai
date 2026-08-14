"""
mast3r_reconstruct.py — feed-forward multi-view reconstruction via MASt3R
(no training, no seeding from the existing SfM run — this arm gets nothing
from the rest of the pipeline except the raw video frames), for 9.3.

Three questions this answers directly:
  1. Runtime vs classical SfM, on the same frames (printed + saved).
  2. Does it reconstruct the section of a video where classical SfM
     fragmented? Pass --frame-range to target that section specifically
     (IMG_3126's own SfM run fragments into a 194-image main model + a
     41-image fragment — see the main README's SfM baseline section for
     which frame indices that corresponds to).
  3. **The headline question**: does MASt3R's own metric-scale training
     make its point cloud usably metric WITHOUT the marker at all? Every
     other arm in this project (SfM, 3DGS, NeRF, the 08_depth/ scale
     methods) still needs cm_per_unit from marker/Zhang calibration to turn
     its point cloud into centimeters. This arm never touches scale.json —
     the metric checkpoint's output is used AS METRIC directly (just a
     unit conversion, meters -> cm), and that's the actual thing being
     tested: is that number trustworthy?

Requires the naver/mast3r repo installed (not just the model weights) — see
3d_recons_alternatives/README.md section D0 for setup. MASt3R's Python API
for the sparse global aligner was verified against the repo's public source
as of 2026-08 (sparse_global_alignment() in mast3r/cloud_opt/sparse_ga.py,
returning a SparseGA object with get_dense_pts3d/get_im_poses/get_focals),
but exact tensor shapes weren't independently confirmed by running it —
check the printed point/pose counts make sense the first time you run this.

Usage:
  python 3d_recons_alternatives/mast3r_reconstruct.py \
      --frames-dir dataset/frames/IMG_3126_sfm \
      --out 3d_recons_alternatives/results/IMG_3126_mast3r
  # target the specific frame range where classical SfM fragmented:
  python 3d_recons_alternatives/mast3r_reconstruct.py \
      --frames-dir dataset/frames/IMG_3126_sfm --frame-range 180:226 \
      --out 3d_recons_alternatives/results/IMG_3126_mast3r_fragment
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "05_geometry"))
from room_dims import rotation_to_z, density_peak, compute_room_dims, render_topdown


def up_from_poses(im_poses: np.ndarray) -> np.ndarray:
    """Gravity prior from MASt3R's own camera-to-world poses.

    MASt3R/dust3r's get_im_poses() returns camera-TO-world (world_from_cam)
    4x4 matrices — the OPPOSITE convention from pycolmap's cam_from_world()
    used everywhere else in this project. A camera's local Y-axis ("down"
    in typical image conventions) expressed in world coordinates is the
    SECOND COLUMN of the rotation block here (not a row, as it would be for
    a world->cam matrix in room_dims.py's up_from_cameras) — getting this
    backwards silently produces a plausible-looking but wrong gravity
    direction, so it's called out explicitly rather than copy-pasted.
    """
    downs = [T[:3, 1] for T in im_poses]
    down = np.mean(downs, axis=0)
    return -down / np.linalg.norm(down)


def select_frames(frames_dir: Path, frame_range: str | None, every_n: int) -> list[Path]:
    frames = sorted(frames_dir.glob("*.jpg"))
    if frame_range:
        lo, hi = (int(x) for x in frame_range.split(":"))
        frames = frames[lo:hi]
    return frames[::every_n]


def main():
    parser = argparse.ArgumentParser(description="Feed-forward reconstruction via MASt3R")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--frame-range", default=None,
                        help="start:end frame INDEX slice (post-sort, pre-subsample) — e.g. "
                             "180:226 to target the specific section where classical SfM fragmented")
    parser.add_argument("--every-n-frames", type=int, default=3,
                        help="Pairwise inference is heavier than SfM's feature matching even with "
                             "windowed pairing — subsample")
    parser.add_argument("--win-size", type=int, default=3,
                        help="Sliding-window pair radius for scene_graph='swin-N' (video-appropriate, "
                             "not all-pairs) — each frame pairs with its N nearest neighbors")
    parser.add_argument("--checkpoint", default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from mast3r.model import AsymmetricMASt3R
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images

    all_frames = sorted(args.frames_dir.glob("*.jpg"))
    frame_paths = select_frames(args.frames_dir, args.frame_range, args.every_n_frames)
    print(f"Using {len(frame_paths)} frames (of {len(all_frames)} available in {args.frames_dir})")
    if len(frame_paths) < 3:
        print("FAILED: need at least 3 frames")
        return

    print(f"Loading {args.checkpoint}...")
    model = AsymmetricMASt3R.from_pretrained(args.checkpoint).to(device)

    t0 = time.perf_counter()
    imgs = load_images([str(p) for p in frame_paths], size=512)
    pairs = make_pairs(imgs, scene_graph=f"swin-{args.win_size}", prefilter=None, symmetrize=True)
    print(f"{len(pairs)} pairs from {len(imgs)} images (sliding-window, not all-pairs)")

    args.out.mkdir(parents=True, exist_ok=True)
    scene = sparse_global_alignment(
        [str(p) for p in frame_paths], pairs, str(args.out / "cache"), model, device=device,
    )
    runtime_s = time.perf_counter() - t0
    print(f"Reconstruction done in {runtime_s:.1f}s")

    pts3d_list, _confs = scene.get_dense_pts3d(clean_depth=True)
    pts_m = np.concatenate([np.asarray(p).reshape(-1, 3) for p in pts3d_list], axis=0)
    im_poses = np.asarray(scene.get_im_poses().detach().cpu())
    focals = np.asarray(scene.get_focals().detach().cpu())
    print(f"Dense point cloud: {len(pts_m)} points, {len(im_poses)} camera poses")
    print(f"Focal lengths (px, MASt3R's own estimate): {focals.round(0).tolist()}")

    # MASt3R's metric-scale checkpoint outputs points already in METERS —
    # this is the actual test of "does metric-scale training remove marker
    # dependency": no scale.json, no cm_per_unit ratio-fitting, just x100.
    pts_cm = pts_m * 100.0
    cam_centers_cm = im_poses[:, :3, 3] * 100.0

    up = up_from_poses(im_poses)
    R = rotation_to_z(up)
    z_all = (pts_cm @ R.T)[:, 2]
    z_floor = density_peak(z_all, 0.5, 35.0)
    if z_floor is None:
        print("FAILED: could not localize the floor density peak")
        return

    metrics, geometry = compute_room_dims(pts_cm, R, z_floor, cam_centers_cm)
    bound = ">= " if metrics["height_is_lower_bound"] else ""
    print(f"\nRoom height : {bound}{metrics['height_cm']:.1f} cm"
          f"{'  [LOW CONFIDENCE]' if not metrics['height_reliable'] else ''}")
    print(f"Room length : {metrics['length_cm']:.1f} cm")
    print(f"Room width  : {metrics['width_cm']:.1f} cm")

    viz_path = args.out / "mast3r_room_topdown.png"
    render_topdown(geometry, R, z_floor, 1.0, cam_centers_cm, None, viz_path)

    out = dict(source="mast3r", checkpoint=args.checkpoint,
              frames_used=len(frame_paths), frames_available=len(all_frames),
              frame_range=args.frame_range, runtime_s=round(runtime_s, 1),
              n_points=len(pts_m),
              scale_source="mast3r_native_metric (no marker, no scale.json)",
              **metrics)
    out_path = args.out / "mast3r_room_dims.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")
    print(f"Saved {viz_path}")


if __name__ == "__main__":
    main()
