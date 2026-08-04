"""
visualize_sfm.py — presentation-quality visualizations of a (scaled) SfM model.

Outputs into <sfm_dir>/viz/:
  - pointcloud.ply       colored point cloud, metric (cm) if scale.json exists
                         → open in CloudCompare / MeshLab for interactive orbits
  - view_*.png           rendered views (dark theme, camera trajectory, marker)

Usage:
  python experiments/visualize_sfm.py output/sfm/IMG_3126 --model 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "05_geometry"))
from room_dims import up_from_cameras, rotation_to_z


def load_model(sfm_dir: Path, model: str):
    rec = pycolmap.Reconstruction(str(sfm_dir / "sparse" / model))
    scale_path = sfm_dir / "scale.json"
    cm_per_unit = 1.0
    scale_info = None
    if scale_path.exists():
        scale_info = json.loads(scale_path.read_text())
        cm_per_unit = scale_info["cm_per_unit"]
    return rec, cm_per_unit, scale_info


def export_ply(rec: pycolmap.Reconstruction, cm_per_unit: float,
               R: np.ndarray, out_path: Path,
               min_track: int = 3, max_err: float = 2.0) -> int:
    """Colored PLY, gravity-aligned, in centimeters."""
    pts, cols = [], []
    for p in rec.points3D.values():
        if p.track.length() < min_track or p.error > max_err:
            continue
        pts.append(p.xyz)
        cols.append(p.color)
    pts = (np.asarray(pts) * cm_per_unit) @ R.T
    cols = np.asarray(cols, dtype=np.uint8)

    # camera trajectory as red points so it shows up in external viewers
    cams = np.array([im.projection_center() for im in rec.images.values()])
    cams = (cams * cm_per_unit) @ R.T
    cam_cols = np.tile([255, 40, 40], (len(cams), 1)).astype(np.uint8)

    allp = np.vstack([pts, cams])
    allc = np.vstack([cols, cam_cols])

    with open(out_path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(allp)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(allp, allc):
            f.write(f"{x:.2f} {y:.2f} {z:.2f} {r} {g} {b}\n")
    return len(pts)


def render_views(rec, cm_per_unit: float, R: np.ndarray, scale_info,
                 out_dir: Path, max_points: int = 15000):
    pts, cols = [], []
    for p in rec.points3D.values():
        if p.track.length() < 3 or p.error > 2.0:
            continue
        pts.append(p.xyz)
        cols.append(p.color)
    pts = (np.asarray(pts) * cm_per_unit) @ R.T
    cols = np.asarray(cols) / 255.0

    # percentile crop for a clean view
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts, cols = pts[keep], cols[keep]
    cols = np.clip(cols * 1.7, 0, 1)          # brighten dim indoor colors

    cams = np.array([im.projection_center() for im in rec.images.values()])
    cams = (cams * cm_per_unit) @ R.T
    order = np.argsort([im.name for im in rec.images.values()])
    cams = cams[order]

    markers = []
    if scale_info:
        for seg in scale_info.get("segments", []):
            mp = np.array(list(seg["marker_points_model"].values()))
            markers.append((mp.mean(axis=0) * cm_per_unit) @ R.T)

    def render(fname, p, c, cam, elev, azim, title, psize):
        fig = plt.figure(figsize=(13, 9), facecolor="#101014")
        ax = fig.add_subplot(111, projection="3d", facecolor="#101014")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=c, s=psize, linewidths=0,
                   depthshade=False)
        if cam is not None and len(cam):
            ax.plot(cam[:, 0], cam[:, 1], cam[:, 2],
                    color="#ff5050", linewidth=1.8, alpha=0.95, label="camera path")
        for m in markers:
            if (p[:, 0].min() <= m[0] <= p[:, 0].max()
                    and p[:, 1].min() <= m[1] <= p[:, 1].max()):
                ax.scatter(*m, color="#ff40ff", s=180, marker="*",
                           edgecolors="white", linewidths=0.6, zorder=5,
                           label="scale marker")
        span = p.max(axis=0) - p.min(axis=0)
        ax.set_box_aspect(span)               # true proportions, no stretching
        ax.set_xlim(p[:, 0].min(), p[:, 0].max())
        ax.set_ylim(p[:, 1].min(), p[:, 1].max())
        ax.set_zlim(p[:, 2].min(), p[:, 2].max())
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title, color="white", fontsize=14, pad=0)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(dict(zip(labels, handles)).values(),
                            dict(zip(labels, handles)).keys(),
                            loc="lower left", frameon=False)
            for t in leg.get_texts():
                t.set_color("white")
        fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        fig.savefig(out_dir / fname, dpi=200, facecolor="#101014",
                    bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"  saved {out_dir / fname}")

    unit = "cm" if cm_per_unit != 1.0 else "model units"
    title_full = f"Boiler room — SfM sparse reconstruction ({unit}, metric via marker)"
    for fname, elev, azim in [("view_orbit1.png", 18, -60),
                              ("view_orbit2.png", 25, 30),
                              ("view_top.png", 88, -90)]:
        render(fname, pts, cols, cams, elev, azim, title_full, psize=2.5)

    # zoomed views around the (first) marker placement = the boiler-room wall
    if markers:
        m = markers[0]
        near = np.linalg.norm(pts[:, :2] - m[:2], axis=1) < 220
        p_room, c_room = pts[near], cols[near]
        cam_near = cams[np.linalg.norm(cams[:, :2] - m[:2], axis=1) < 220]
        if len(p_room) > 500:
            for fname, elev, azim in [("room_orbit1.png", 15, -55),
                                      ("room_orbit2.png", 22, 35)]:
                render(fname, p_room, c_room, cam_near, elev, azim,
                       f"Boiler room detail — metric SfM ({unit})", psize=5.0)


def main():
    parser = argparse.ArgumentParser(description="Visualize a scaled SfM model")
    parser.add_argument("sfm_dir", type=Path)
    parser.add_argument("--model", default="1")
    args = parser.parse_args()

    rec, cm_per_unit, scale_info = load_model(args.sfm_dir, args.model)
    print(f"Model: {rec.num_reg_images()} images, {rec.num_points3D()} points, "
          f"scale {cm_per_unit:.3f} cm/unit")

    up = up_from_cameras(rec)
    R = rotation_to_z(up)      # gravity-aligned: floor is horizontal in exports

    out_dir = args.sfm_dir / "viz"
    out_dir.mkdir(exist_ok=True)

    n = export_ply(rec, cm_per_unit, R, out_dir / "pointcloud.ply")
    print(f"  saved {out_dir / 'pointcloud.ply'}  ({n} points + camera path, metric cm)")

    render_views(rec, cm_per_unit, R, scale_info, out_dir)


if __name__ == "__main__":
    main()
