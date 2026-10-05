"""
export_mesh.py — export a textured/surfaced mesh (OBJ) and a colored point
cloud (LAS + PLY) from the reconstruction, matching the sibling photogram
project's deliverables. Room dimensions/scale/placement are numbers and
JSON; this is the first stage in this repo that produces an actual 3D file
you can open in Blender/MeshLab/CAD.

Mesh method: Ball Pivoting (BPA), not Poisson. Boiler rooms are full of
thin structures (pipes, conduit, valve stems) that Poisson's implicit
surface fitting fills in as hallucinated blobs across open space — the
same reason photogram uses BPA for its outdoor/object scan modes instead
of Poisson (reserved there for empty, wall-bounded indoor rooms). A boiler
room is architecturally "indoor" but geometrically closer to those
cluttered cases, so this always uses BPA.

Point source: dense MVS cloud if available (--dense-ply, from
03_reconstruction/run_mvs.py --dense) — far better mesh quality, since BPA
needs real surface density to bridge gaps. Falls back to the sparse SfM
cloud otherwise (same fallback room_dims.py/placement_3d.py use). Output
is in METRES (not cm or SfM units) — the common CAD/Blender convention,
and consistent with compute_dimensions-style output elsewhere.

Usage:
  python 09_export/export_mesh.py output/sfm/IMG_3126 --model 0
  python 09_export/export_mesh.py output/sfm/IMG_3126 --model 0 --dense-ply output/sfm/IMG_3126/dense/fused.ply
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap

TARGET_POINTS = 400_000   # downsample above this before BPA — bounds runtime/memory


def load_cloud_m(sfm_dir: Path, model: str, dense_ply: Path | None, cm_per_unit: float,
                 rec: pycolmap.Reconstruction) -> "o3d.geometry.PointCloud":
    """Load the best-available cloud, scaled to metres, WITH color when available."""
    if dense_ply and dense_ply.exists():
        pcd = o3d.io.read_point_cloud(str(dense_ply))
        print(f"Dense cloud: {len(pcd.points):,} points ({dense_ply.name}) "
             f"— colors: {pcd.has_colors()}")
    else:
        # Same track-length/reprojection-error filter as room_dims.py's
        # load_filtered_points, but also keeping each point's color — COLMAP
        # populates Point3D.color from the source images during mapping, and
        # load_filtered_points (xyz-only, built for room_dims's own needs)
        # would silently drop it.
        pts, colors = [], []
        for p in rec.points3D.values():
            if p.track.length() >= 3 and p.error < 1.5:
                pts.append(p.xyz)
                colors.append(p.color)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pts))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64) / 255.0)
        print(f"Sparse cloud (dense unavailable): {len(pts):,} points")

    pts_m = np.asarray(pcd.points) * (cm_per_unit / 100.0)
    pcd.points = o3d.utility.Vector3dVector(pts_m)
    return pcd


def _estimate_bpa_radii(pcd: "o3d.geometry.PointCloud", sample_n: int = 1000) -> tuple[float, list[float]]:
    pts = np.asarray(pcd.points)
    sample_idx = np.random.choice(len(pts), min(sample_n, len(pts)), replace=False)
    tree = o3d.geometry.KDTreeFlann(pcd)
    nn_dists = []
    for i in sample_idx:
        _, _, d2 = tree.search_knn_vector_3d(pts[i], 2)
        if len(d2) > 1:
            nn_dists.append(float(d2[1]) ** 0.5)
    avg_nn = float(np.median(nn_dists)) if nn_dists else 0.02
    return avg_nn, [avg_nn * 2, avg_nn * 4, avg_nn * 8]


def build_mesh(pcd_in: "o3d.geometry.PointCloud", cam_centers_m: np.ndarray) -> "o3d.geometry.TriangleMesh":
    """BPA reconstruction + the same cleanup chain photogram uses (component
    filter, long-edge filter, hole filling, Taubin smoothing)."""
    pcd = pcd_in
    n_raw = len(pcd.points)
    if n_raw > TARGET_POINTS:
        avg_nn, _ = _estimate_bpa_radii(pcd, sample_n=500)
        ratio = n_raw / TARGET_POINTS
        voxel = avg_nn * (ratio ** (1 / 3))
        pcd = pcd.voxel_down_sample(max(voxel, avg_nn * 1.1))
        print(f"Downsampled for meshing: {n_raw:,} -> {len(pcd.points):,} points")

    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    if len(cam_centers_m):
        pcd.orient_normals_towards_camera_location(cam_centers_m.mean(axis=0))
    else:
        pcd.orient_normals_consistent_tangent_plane(30)

    avg_nn, radii = _estimate_bpa_radii(pcd)
    print(f"Ball Pivoting: {len(pcd.points):,} points, radii~{[round(r*100,1) for r in radii]}cm")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    n_before = len(mesh.triangles)

    # Keep components with >= 1% of the largest component's triangle count —
    # drops floating noise, keeps thin real structures (pipes, handles).
    if len(mesh.triangles) > 0:
        tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        cluster_n_tris = np.asarray(cluster_n_tris)
        if len(cluster_n_tris) > 1:
            max_component = int(cluster_n_tris.max())
            min_keep = max(10, int(max_component * 0.01))
            small_mask = np.array([cluster_n_tris[c] < min_keep for c in tri_clusters])
            if small_mask.any():
                mesh.remove_triangles_by_mask(small_mask)
                mesh.remove_unreferenced_vertices()

    # Long-edge filter — drop "tent" triangles bridging real surface to distant noise.
    if len(mesh.triangles) > 0:
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        edge_a = np.linalg.norm(verts[tris[:, 1]] - verts[tris[:, 0]], axis=1)
        edge_b = np.linalg.norm(verts[tris[:, 2]] - verts[tris[:, 1]], axis=1)
        edge_c = np.linalg.norm(verts[tris[:, 0]] - verts[tris[:, 2]], axis=1)
        max_edge = np.maximum(np.maximum(edge_a, edge_b), edge_c)
        edge_threshold = float(np.median(max_edge)) * 10
        long_mask = max_edge > edge_threshold
        if long_mask.any():
            mesh.remove_triangles_by_mask(long_mask)
            mesh.remove_unreferenced_vertices()

    print(f"Cleanup: {n_before} -> {len(mesh.triangles)} triangles")

    if len(mesh.triangles) > 0:
        import trimesh as tm
        tm_mesh = tm.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles), process=False)
        n_faces_before = len(tm_mesh.faces)
        tm.repair.fill_holes(tm_mesh)
        if len(tm_mesh.faces) > n_faces_before:
            print(f"Hole filling: +{len(tm_mesh.faces) - n_faces_before} triangles")
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(tm_mesh.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(tm_mesh.faces)

    if len(mesh.triangles) > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=30)
        mesh.compute_vertex_normals()
    return mesh


def transfer_colors(mesh: "o3d.geometry.TriangleMesh", source_pcd: "o3d.geometry.PointCloud") -> None:
    """Nearest-neighbor color transfer from the source cloud to mesh vertices (in place)."""
    if not source_pcd.has_colors() or len(mesh.vertices) == 0:
        return
    tree = o3d.geometry.KDTreeFlann(source_pcd)
    src_colors = np.asarray(source_pcd.colors)
    verts = np.asarray(mesh.vertices)
    out_colors = np.zeros((len(verts), 3))
    for i, v in enumerate(verts):
        _, idx, _ = tree.search_knn_vector_3d(v, 1)
        out_colors[i] = src_colors[idx[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(out_colors)


def write_las(pcd: "o3d.geometry.PointCloud", out_path: Path) -> int:
    import laspy
    pts = np.asarray(pcd.points)
    n = len(pts)
    header = laspy.LasHeader(point_format=2, version="1.4")
    las = laspy.LasData(header=header)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        las.red   = (np.clip(colors[:, 0], 0, 1) * 65535).astype(np.uint16)
        las.green = (np.clip(colors[:, 1], 0, 1) * 65535).astype(np.uint16)
        las.blue  = (np.clip(colors[:, 2], 0, 1) * 65535).astype(np.uint16)
    else:
        las.red = las.green = las.blue = np.zeros(n, dtype=np.uint16)
    las.write(str(out_path))
    return n


def main():
    parser = argparse.ArgumentParser(description="Export mesh (OBJ) + point cloud (LAS/PLY)")
    parser.add_argument("sfm_dir", type=Path)
    parser.add_argument("--model", default="0", help="Sparse model index (default 0)")
    parser.add_argument("--dense-ply", type=Path, default=None,
                        help="Dense MVS cloud (sfm_dir/dense/fused.ply) — used for meshing/LAS "
                             "if given; falls back to the sparse SfM cloud otherwise.")
    parser.add_argument("--scale-json", type=Path, default=None,
                        help="Override which scale file to use (default: <sfm_dir>/scale.json)")
    args = parser.parse_args()

    scale_path = args.scale_json or (args.sfm_dir / "scale.json")
    scale_info = json.loads(scale_path.read_text())
    cm_per_unit = scale_info["cm_per_unit"]

    rec = pycolmap.Reconstruction(str(args.sfm_dir / "sparse" / args.model))
    print(f"Model: {rec.num_reg_images()} images, {rec.num_points3D()} points")
    print(f"Scale: {cm_per_unit:.4f} cm/unit")

    pcd = load_cloud_m(args.sfm_dir, args.model, args.dense_ply, cm_per_unit, rec)
    if len(pcd.points) < 100:
        print(f"FAILED: too few points for export ({len(pcd.points)})")
        return

    cam_centers_m = np.array([im.projection_center() for im in rec.images.values()]) * (cm_per_unit / 100.0)

    export_dir = args.sfm_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # ── Colored point cloud (PLY) ──────────────────────────────────────────
    ply_path = export_dir / "output.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"Saved {ply_path} ({len(pcd.points):,} points)")

    # ── LAS ─────────────────────────────────────────────────────────────────
    las_path = export_dir / "output.las"
    n_las = write_las(pcd, las_path)
    print(f"Saved {las_path} ({n_las:,} points)")

    # ── Mesh (OBJ) ──────────────────────────────────────────────────────────
    mesh = build_mesh(pcd, cam_centers_m)
    if len(mesh.triangles) == 0:
        print("FAILED: Ball Pivoting produced no triangles (point cloud too sparse/noisy for meshing)")
        out = dict(ply=str(ply_path), las=str(las_path), obj=None,
                  n_points=len(pcd.points), source="dense" if args.dense_ply and args.dense_ply.exists() else "sparse")
        (export_dir / "export.json").write_text(json.dumps(out, indent=2))
        return

    transfer_colors(mesh, pcd)
    obj_path = export_dir / "output.obj"
    o3d.io.write_triangle_mesh(str(obj_path), mesh, write_vertex_colors=True)
    print(f"Saved {obj_path} ({len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris)")

    # Mesh also as PLY — OBJ's per-vertex color extension isn't reliably
    # parsed by common web viewers (Three.js's OBJLoader doesn't read it at
    # all), but PLY vertex colors are a first-class, universally-supported
    # field. This is what the wrapper's browser viewer loads.
    mesh_ply_path = export_dir / "output_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_ply_path), mesh, write_vertex_colors=True)
    print(f"Saved {mesh_ply_path} (for browser viewing)")

    out = dict(
        ply=str(ply_path), las=str(las_path), obj=str(obj_path), mesh_ply=str(mesh_ply_path),
        n_points=len(pcd.points), n_mesh_vertices=len(mesh.vertices), n_mesh_triangles=len(mesh.triangles),
        source="dense" if args.dense_ply and args.dense_ply.exists() else "sparse",
        cm_per_unit=cm_per_unit,
    )
    (export_dir / "export.json").write_text(json.dumps(out, indent=2))
    print(f"Saved {export_dir / 'export.json'}")


if __name__ == "__main__":
    main()
