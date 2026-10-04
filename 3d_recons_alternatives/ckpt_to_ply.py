"""
ckpt_to_ply.py — converts a gsplat v1.4.0 training checkpoint (.pt, saved
by examples/simple_trainer.py's own torch.save) into the standard 3DGS PLY
format gaussians_to_room_dims.py expects.

This gsplat version (matching the installed pip package, checked out via
git tag v1.4.0 in C:\\thermovation-repos\\gsplat) doesn't have the
--save_ply flag newer gsplat versions do — see 3d_recons_alternatives/README
for the version this project's install actually has. Only the fields
gaussians_to_room_dims.py reads are written (x,y,z,opacity,scale_0..2) —
no color/SH, room-dimension extraction never touches those.

Usage:
  python 3d_recons_alternatives/ckpt_to_ply.py \
      --ckpt results/IMG_3126/ckpts/ckpt_29999_rank0.pt \
      --out results/IMG_3126/ply/point_cloud.ply
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    splats = data["splats"]
    means = splats["means"].detach().numpy().astype(np.float32)
    scales = splats["scales"].detach().numpy().astype(np.float32)   # already log-scale
    opacities = splats["opacities"].detach().numpy().astype(np.float32)  # already logit
    n = len(means)
    print(f"Loaded {n} Gaussians from step {data.get('step')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float opacity\n"
            "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
            "end_header\n"
        ).encode("ascii")
        f.write(header)
        rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                 ("opacity", "<f4"),
                                 ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4")])
        rec["x"], rec["y"], rec["z"] = means[:, 0], means[:, 1], means[:, 2]
        rec["opacity"] = opacities
        rec["scale_0"], rec["scale_1"], rec["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
        f.write(rec.tobytes())
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
