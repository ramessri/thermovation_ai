# 9.2 / 9.3 — SfM vs 3D Gaussian Splatting vs NeRF vs MASt3R vs VGGT: uni PC runbook

Compares classical SfM (existing pipeline) against 3D Gaussian Splatting
(via `gsplat`), NeRF (via `nerfstudio`'s `nerfacto`), and two feed-forward
transformer reconstruction methods (`MASt3R`, `VGGT`) on the same two
videos — **IMG_3126** (clean 1080p single-room orbit, best-characterized
video) and **WhatsApp 09.07.37** (576p, the one where classical SfM
fragments into multiple sub-models) — judged on whole-room reconstruction
completeness and accuracy, not just the ceiling-height symptom.

3DGS and NeRF (Branches A, B) reuse the *existing* SfM run for each video
(camera poses, marker scale) rather than re-deriving anything, so any
difference in their room-dimension result comes from the reconstruction
method itself, not from different input data. **MASt3R and VGGT (Branches
C, D) reconstruct completely independently** — neither gets SfM's camera
poses fed in — but they differ from each other in a way worth knowing before
running either: MASt3R's checkpoint is trained for genuine metric-scale
output (Branch C tests whether that removes marker dependency entirely),
while VGGT's own documentation confirms its raw output "inherently lacks
metric scale" — so Branch D calibrates it externally, by comparing VGGT's
own camera-to-camera distances against the marker-scaled SfM camera
trajectory for the same frames. VGGT is testing something different: a much
simpler feed-forward architecture (single pass over all views, no pairwise
matching + global alignment) and whether it still reconstructs the
SfM-fragmented section, not marker independence.

All five arms are judged by the exact same dimension-extraction code
(`05_geometry/room_dims.py`'s `compute_room_dims()`), not five separate
hand-rolled implementations.

## 0. One-time setup

```
pip install gsplat nerfstudio plyfile   # already added to requirements.txt
git clone https://github.com/nerfstudio-project/gsplat.git /tmp/gsplat-repo
git clone --recursive https://github.com/naver/mast3r.git /tmp/mast3r-repo
pip install -r /tmp/mast3r-repo/requirements.txt -r /tmp/mast3r-repo/dust3r/requirements.txt
git clone https://github.com/facebookresearch/vggt.git /tmp/vggt-repo
pip install -r /tmp/vggt-repo/requirements.txt
export PYTHONPATH=/tmp/mast3r-repo:/tmp/vggt-repo:$PYTHONPATH   # needed every session, or add to your shell profile
```

`gsplat`'s pip package is the CUDA rasterizer + Python bindings; the actual
training script (`examples/simple_trainer.py`) lives only in the repo, not
the pip package — that's why the clone is needed alongside the pip install.
`nerfstudio` installs its CLI tools (`ns-train`, `ns-process-data`,
`ns-export`) directly, no separate clone needed. `mast3r` and `vggt` aren't
pip-installable at all — clone both (mast3r recursively, it vendors
`dust3r` as a submodule) and put both on `PYTHONPATH`. VGGT's checkpoint
(`facebook/VGGT-1B`) downloads automatically on first run via
`VGGT.from_pretrained()`, same HF-hub pattern as everything else in this
project.

**Flags below were verified against gsplat's, nerfstudio's, MASt3R's, and
VGGT's public docs/source as of 2026-08, but CLI/API details drift between
releases — run `python /tmp/gsplat-repo/examples/simple_trainer.py --help`,
`ns-train nerfacto --help`, `ns-export pointcloud --help` first and adjust
flag names if anything here doesn't match. MASt3R's and VGGT's Python APIs in
particular (`mast3r_reconstruct.py`, `vggt_reconstruct.py`) were verified
against each repo's public source, not by actually running them — check the
printed point/pose counts look sane the first time.**

---

## Branch A: 3D Gaussian Splatting (gsplat)

### A1. Prepare the dataset (per video)

Reuses the existing SfM output — no new reconstruction, no undistortion
step (gsplat's own COLMAP loader undistorts on load):

```
python 3d_recons_alternatives/prepare_gsplat_dataset.py \
    --sfm-dir output/sfm/IMG_3126 --frames-dir dataset/frames/IMG_3126_sfm \
    --model 1 --out 3d_recons_alternatives/data/IMG_3126

python 3d_recons_alternatives/prepare_gsplat_dataset.py \
    --sfm-dir output/sfm/09.07.37 --frames-dir dataset/frames/09.07.37_sfm \
    --model 1 --out 3d_recons_alternatives/data/09.07.37
```

Adjust `--sfm-dir`/`--frames-dir` to wherever your actual `run.py` output
landed for these two videos — check `output/pipeline/<video>/` (or
`output/sfm/<video>/` if you ran the stage scripts directly) and the
matching `frames/` dir.

### A2. Train (per video)

```
cd /tmp/gsplat-repo
python examples/simple_trainer.py default \
    --data_dir <repo>/3d_recons_alternatives/data/IMG_3126 \
    --data_factor 1 \
    --result_dir <repo>/3d_recons_alternatives/results/IMG_3126 \
    --save_ply
```

Repeat with `09.07.37` in place of `IMG_3126`. **Important for a fair
comparison**: do NOT enable pose optimization (leave any `--pose_opt`-style
flag off/default) — this run needs to keep the exact camera poses SfM
already solved, not let gsplat re-optimize them, otherwise a better result
would be partly "pose refinement helped" rather than purely "denser
Gaussian geometry helped." Check the trainer's printed config at startup to
confirm poses aren't being optimized.

Note the wall-clock time gsplat prints when training finishes — you'll pass
it to the Compare step.

`--save_ply` writes the trained Gaussians to
`<result_dir>/ply/point_cloud_<iters>.ply` (exact filename depends on the
final iteration count — check `<result_dir>/ply/` after training finishes).

### A3. Extract room dimensions

```
python 3d_recons_alternatives/gaussians_to_room_dims.py \
    --ply 3d_recons_alternatives/results/IMG_3126/ply/point_cloud_29999.ply \
    --sfm-dir output/sfm/IMG_3126 --model 1 \
    --out 3d_recons_alternatives/results/IMG_3126
```

(Adjust the `.ply` filename to whatever A2 actually produced.) Filters
Gaussians by opacity and physical size before treating their centers as a
point cloud — see the script's docstring for why (raw PLY stores opacity as
a logit and scale as log-scale; both need inverse-transforming first).
Repeat for `09.07.37`.

---

## Branch B: NeRF (nerfstudio's nerfacto)

### B1. Convert the existing SfM model to nerfstudio format (per video)

No separate prep script needed — nerfstudio's own `ns-process-data` can
import an existing COLMAP model directly via `--skip-colmap`:

```
ns-process-data images \
    --data dataset/frames/IMG_3126_sfm \
    --output-dir 3d_recons_alternatives/nerf_data/IMG_3126 \
    --skip-colmap --colmap-model-path output/sfm/IMG_3126/sparse/1
```

Repeat for `09.07.37`. If this errors looking for `images_2`/`images_4`/
`images_8` downscaled folders (a known rough edge with `--skip-colmap`),
pass `--num-downscales 0` or generate those folders first — see the
troubleshooting note in nerfstudio's GitHub issues for `--skip-colmap` if
it comes up.

### B2. Train

```
ns-train nerfacto --data 3d_recons_alternatives/nerf_data/IMG_3126 \
    --output-dir 3d_recons_alternatives/nerf_results/IMG_3126
```

Repeat for `09.07.37`. Note the wall-clock time nerfstudio prints when
training finishes — you'll pass it to the Compare step.

### B3. Export a point cloud in the ORIGINAL (COLMAP) scale/frame

```
ns-export pointcloud \
    --load-config 3d_recons_alternatives/nerf_results/IMG_3126/nerfacto/<timestamp>/config.yml \
    --output-dir 3d_recons_alternatives/nerf_results/IMG_3126 \
    --save-world-frame
```

**`--save-world-frame` is not optional here.** Nerfstudio's training
dataparser auto-scales and reorients the scene internally by default
(`auto_scale_poses`, axis reorientation) — without this flag the exported
point cloud would be in that internal frame, not the original COLMAP one,
which would silently invalidate the `cm_per_unit` scale reused in the next
step. `<timestamp>` is whatever directory `ns-train` created under
`nerfacto/` — check `3d_recons_alternatives/nerf_results/IMG_3126/` after
training.

### B4. Extract room dimensions

```
python 3d_recons_alternatives/nerf_pointcloud_to_room_dims.py \
    --ply 3d_recons_alternatives/nerf_results/IMG_3126/point_cloud.ply \
    --sfm-dir output/sfm/IMG_3126 --model 1 \
    --out 3d_recons_alternatives/results/IMG_3126_nerf
```

This script also prints a **world-frame sanity check** (point-cloud
centroid vs. SfM camera-trajectory centroid, in cm) — if that number is
huge for a boiler-room-scale video, `--save-world-frame` isn't behaving as
expected on the installed nerfstudio version and these room-dimension
numbers shouldn't be trusted until that's resolved. Repeat for `09.07.37`.

---

## Branch C: MASt3R (feed-forward transformer, no training, no marker)

No prep step — this arm reads video frames directly, nothing from the
existing SfM run.

### C1. Reconstruct

```
python 3d_recons_alternatives/mast3r_reconstruct.py \
    --frames-dir dataset/frames/IMG_3126_sfm \
    --out 3d_recons_alternatives/results/IMG_3126_mast3r
```

Repeat for `09.07.37`. Prints and saves its own `runtime_s` — no manual
console-watching needed for this arm's timing, unlike gsplat/nerfstudio.

**Targeting the section where classical SfM fragmented** (per 9.3's second
checklist item) — check the main README's SfM baseline section for which
frames that corresponds to on your video, then:

```
python 3d_recons_alternatives/mast3r_reconstruct.py \
    --frames-dir dataset/frames/IMG_3126_sfm --frame-range 180:226 \
    --out 3d_recons_alternatives/results/IMG_3126_mast3r_fragment
```

`--frame-range start:end` slices the sorted frame list by index *before*
`--every-n-frames` subsampling, so it targets a specific stretch of the
video rather than the whole thing.

### C2. What's different about this arm's output

`mast3r_room_dims.json` has no `cm_per_unit` derived from any scale.json —
its `scale_source` field literally says
`"mast3r_native_metric (no marker, no scale.json)"`. That's the point: this
is the one arm actually testing whether metric-scale training removes
marker dependency, not just a 5th geometry method compared under the
marker's scale like the other three.

---

## Branch D: VGGT (feed-forward transformer, single unified pass)

Also reads video frames directly, but — unlike MASt3R — still needs the
existing SfM model + `scale.json` for one thing: calibrating its own
otherwise-arbitrary scale afterward (see the note at the top of this file
on why VGGT and MASt3R differ here).

### D1. Reconstruct + calibrate scale

```
python 3d_recons_alternatives/vggt_reconstruct.py \
    --frames-dir dataset/frames/IMG_3126_sfm \
    --sfm-dir output/sfm/IMG_3126 --model 1 \
    --out 3d_recons_alternatives/results/IMG_3126_vggt
```

Repeat for `09.07.37`. Prints and saves its own `runtime_s`, same as
MASt3R. The `--sfm-dir`/`--model` here are used only to fit VGGT's scale
ratio after the fact (matching frames by filename, comparing camera-to-
camera distances) — VGGT's own reconstruction never sees them.

**Targeting the fragmented section**, same pattern as Branch C:

```
python 3d_recons_alternatives/vggt_reconstruct.py \
    --frames-dir dataset/frames/IMG_3126_sfm --frame-range 180:226 \
    --sfm-dir output/sfm/IMG_3126 --model 1 \
    --out 3d_recons_alternatives/results/IMG_3126_vggt_fragment
```

Note: if you target a fragment where classical SfM *itself* failed to
register images, there's no SfM camera trajectory for those exact frames to
calibrate against — the scale-fitting step needs at least 5 frames in the
targeted range that SfM *did* register. If VGGT reconstructs a section SfM
couldn't get poses for at all, that's a real success on its own (worth
reporting qualitatively), but this script's `--sfm-dir` calibration won't
produce a metric number for it.

### D2. What's different about this arm's output

`vggt_room_dims.json`'s `scale_source` is
`"camera_trajectory_ratio_vs_sfm_marker_scale"`, with a `scale_fit_cv_pct`
field showing how tight that calibration was — a high CV% here means
VGGT's own camera trajectory shape disagrees with SfM's more than the
tightest-cluster aggregation likes, which is itself informative (VGGT's
relative geometry may be less consistent with the real capture path than
its raw speed/simplicity would suggest).

---

## Compare all five arms

```
python 3d_recons_alternatives/compare_reconstructions.py \
    --video IMG_3126 \
    --sfm-room-dims output/sfm/IMG_3126/room_dims.json \
    --gaussian-room-dims 3d_recons_alternatives/results/IMG_3126/gaussian_room_dims.json \
    --nerf-room-dims 3d_recons_alternatives/results/IMG_3126_nerf/nerf_room_dims.json \
    --mast3r-room-dims 3d_recons_alternatives/results/IMG_3126_mast3r/mast3r_room_dims.json \
    --vggt-room-dims 3d_recons_alternatives/results/IMG_3126_vggt/vggt_room_dims.json \
    --sfm-runtime-s <from run.py's printed summary> \
    --gsplat-runtime-s <from gsplat's training-summary output, A2> \
    --nerf-runtime-s <from ns-train's printed summary, B2> \
    --gt-length-cm 525 --gt-width-cm 152
```

`--gaussian-room-dims`, `--nerf-room-dims`, `--mast3r-room-dims`, and
`--vggt-room-dims` are all optional — pass whichever arms you've actually
run (e.g. just SfM vs 3DGS if the others aren't done yet). MASt3R's and
VGGT's runtimes are picked up automatically from their own JSON, no
`--mast3r-runtime-s`/`--vggt-runtime-s` needed unless you want to override
them. `--gt-*-cm` are optional too — pass them if you have tape
measurements for this video (IMG_3126: L=525cm, W=152cm; 09.07.37: L=464cm,
W=237cm — both from the main README's ground-truth table), omit for a
coverage/runtime-only comparison.

Prints a side-by-side table and saves
`3d_recons_alternatives/results/<video>/comparison.json`.

## What "success" looks like here

Per the roadmap: does a denser/continuous method (3DGS or NeRF) recover the
ceiling where SfM's sparse cloud structurally can't (see the documented
ceiling-detection bug — flat textureless ceilings have almost no SfM
features to triangulate, but both alternates optimize photometric
consistency rather than needing feature matches, so they may fill in that
gap). If `gaussian_room_dims.json`'s or `nerf_room_dims.json`'s
`height_reliable` comes back `true` where SfM's was `false`/lower-bound-
only, that's the headline finding. If it doesn't, that's still a real,
useful result — document why (floaters/background inflating the point
cloud, insufficient view coverage of the ceiling in the source video,
etc.), same as the roadmap's own "if no: document why, still a valid
robustness finding" framing. A difference between the two alternates
themselves (does explicit Gaussian geometry or implicit NeRF density do
better here) is its own worthwhile finding, not just each-vs-SfM.

For MASt3R specifically, success is a different shape: does
`mast3r_room_dims.json`'s length/width/height land close to the other
arms' (and eventually your ground truth) numbers **despite using zero
marker information** — that's the "reduces marker dependency" question
from the roadmap answered directly. A large mismatch is also a real
finding (metric-scale training isn't precise enough yet for this domain),
not a failure of the script. Separately: check whether it reconstructs at
all on the `--frame-range`-targeted fragment where classical SfM broke —
succeeding there where SfM couldn't is the robustness angle 9.3 asked for,
independent of how accurate the resulting dimensions turn out to be.

For VGGT, success has two independent parts, and it's worth keeping them
separate rather than collapsing to one verdict: (1) **speed and robustness**
— does it reconstruct the fragmented section at all, and how does its
runtime compare to MASt3R and classical SfM (a single-pass architecture
over all views should be the fastest of the three, per its own "seconds,
even for hundreds of views" claim — confirm that's actually true on this
hardware and this footage); (2) **accuracy once externally calibrated** —
given the same marker-scale ruler MASt3R skips, does VGGT's length/width/
height land close to the other arms'? A low `scale_fit_cv_pct` with a large
room-dimension error would mean VGGT's own geometry is internally
consistent but not very accurate — a different finding than MASt3R getting
noisy self-calibrated scale, and worth reporting as such rather than
folding both into one "feed-forward transformer arm" number.
