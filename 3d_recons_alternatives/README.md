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

## 2026-08-20/21 — Full 35-video NeRF + 3DGS batch, no admin rights

Ran both remaining arms (NeRF, 3DGS) across every video in `sfm 1`, not
just IMG_3126/09.07.37, via `run_nerf_gsplat_batch.py`. Constraint: no
admin rights on this machine, for either method.

**NeRF** turned out to need no compiler at all — nerfstudio's `nerfacto`
has a pure-`torch` implementation (`--pipeline.model.implementation torch`)
that skips `tinycudann`'s CUDA hash-encoding kernel entirely. It only
needed two portable, no-installer system binaries added to user-level
`PATH` (not admin, just `[Environment]::SetEnvironmentVariable(...,
"User")`):
- ffmpeg: static build from `BtbN/FFmpeg-Builds` GitHub releases
- COLMAP CLI: portable `colmap-x64-windows-nocuda.zip` from `colmap/colmap`
  releases (this is the CLI binary `ns-process-data` shells out to — a
  separate thing from the `pycolmap` Python bindings the rest of this repo
  already uses)

**3DGS (gsplat)** does need a real C++/CUDA host compiler at JIT-compile
time for its rasterizer — no way around that. Got one without admin via
[mmozeiko's `portable-msvc.py`](https://gist.github.com/mmozeiko/7f3162ec2988e81e56d5c4e22cde9977),
which downloads MSVC + Windows SDK components directly from the same
Microsoft package feed the VS installer uses, into a plain folder (no
installer, no registry writes, no admin) — 479MB, `cl.exe` verified
working standalone before touching gsplat. Two extra fixes were needed
beyond just having `cl.exe` on `PATH`:
- CUDA 13.2's headers require the conforming preprocessor:
  `NVCC_APPEND_FLAGS=-Xcompiler /Zc:preprocessor`, or nvcc dies with a
  `C1189` fatal error compiling `cccl/cuda/std/__cccl/preprocessor.h`.
- `torch.utils.cpp_extension`'s build path calls setuptools'
  `_get_vc_env()`, which tries to *rediscover* MSVC itself via
  vswhere/registry rather than trusting the `INCLUDE`/`LIB`/`PATH` already
  set — and fails, since a portable install isn't a registered VS
  instance. Fixed with `DISTUTILS_USE_SDK=1` and `MSSdk=1`, which are the
  standard distutils/setuptools vars for "trust the environment, don't
  auto-detect."

Both fixes are baked into `run_nerf_gsplat_batch.py`'s `gsplat_env()`/
`nerf_env()` helpers.

Two more real bugs surfaced only once actual training runs completed:
- `ns-export pointcloud`'s checkpoint load hit PyTorch 2.6+'s new
  `weights_only=True` default and couldn't unpickle nerfstudio's own
  checkpoint (`numpy._core.multiarray.scalar` not an allowed global).
  Patched `nerfstudio/utils/eval_utils.py`'s `torch.load(...)` call to
  pass `weights_only=False` — safe here since it's always our own
  just-trained checkpoint, never an untrusted download.
- Default `ns-export pointcloud` wants the model to predict normals
  (`normal_method="model_output"`), which plain `nerfacto` doesn't unless
  trained with predicted normals on. Fixed by passing
  `--normal-method open3d` (estimates normals from the depth-based point
  cloud instead of asking the model for them).

**Iteration budgets were deliberately tiny** — `NERF_ITERS=500`,
`GSPLAT_STEPS=1500`, versus each method's typical default of ~30,000 —
purely to fit all 35 videos inside the available time window (measured
~4.8 it/s for NeRF-torch, ~17.5 it/s for gsplat, both steady-state after
one-time JIT warmup). This is a coverage/pipeline-validation pass, not a
quality benchmark — see the known limitation below before reading any
absolute number as ground truth.

### Known issue: NeRF floater points, not a scale bug

The first full run produced obviously-wrong NeRF room dimensions (e.g.
IMG_3126: 72m ceiling height). The instinctive suspect — nerfstudio's
`--save-world-frame` flag putting the export in the wrong coordinate
frame/scale — turned out to be **not the cause**: independently
reconstructing `transform_poses_to_original_space()`'s math from the CPU
side (no model/GPU involved) and comparing recovered camera centers
against the original SfM reconstruction's camera centers matched to
~1e-6cm, for every registered image. The frame/scale recovery is exact.

The actual cause: at only 500 training steps, `nerfacto`'s density field
hasn't sharpened around true surfaces yet, so ray-marched depth for a
meaningful fraction of pixels lands far past the real geometry (a
well-known early-training NeRF characteristic — density starts spread
along the ray and only concentrates at the correct depth as training
progresses). Those far points single-handedly wreck the density-peak-based
room-dims estimate. Fixed in `nerf_pointcloud_to_room_dims.py` with a
floater filter: reject any point farther than `max(1.5 × camera-trajectory
bounding-box diagonal, 500cm)` from the camera-trajectory centroid, before
the gravity/room-dims fit ever sees it — same spirit as
`gaussians_to_room_dims.py`'s existing opacity/scale filter for 3DGS
floaters. IMG_3126 went from 72m to a plausible 4.8m ceiling height after
the fix; all 10 videos processed before the fix landed were re-run through
the (cheap, GPU-free) room-dims step only, not retrained.

### Known limitation: systematic NeRF-vs-3DGS bias at these iteration counts

Across all 24 successfully-scaled videos, NeRF's room dimensions come out
**3-5x larger than gsplat's** for the same video, consistently — too
uniform across 24 independent reconstructions to be per-video noise. Both
numbers are plausibly biased in opposite directions, both traceable to the
same root cause (severe undertraining relative to each method's normal
iteration count), not an implementation bug in either arm:
- **NeRF over-estimates**: per the floater discussion above, undertrained
  density spreads ray-marched depth outward past true surfaces even after
  the floater filter removes the most extreme outliers — the filter's
  500cm+ radius cap is generous enough to still admit moderately-biased
  points.
- **3DGS under-estimates**: gsplat's Gaussians are seeded directly from
  the sparse SfM point cloud and grow outward via densification during
  training; at only 1500 steps they likely haven't finished expanding to
  the true room extent yet, especially near walls/ceiling far from the
  seed points.

Net: **treat every number in the table below as a coverage/pipeline
validation result, not a measurement.** A trustworthy absolute comparison
needs each method's normal training budget (~30k steps — roughly 40-80x
more compute per video than this pass used), which is a follow-up task,
not something this 4-hour, no-admin-rights pass was trying to establish.

### Results — 24/35 videos (11 skipped: SfM reconstructed but marker-scale
calibration never succeeded for them, `scale.json` missing — a pre-existing
gap in the earlier `04_scale` stage, out of scope here since there's no
metric scale to convert to without it)

| video | gsplat L/W/H (cm) | nerf L/W/H (cm) |
|---|---|---|
| Albert_Mayer_IMG_1831 | 132/124/65 | 624/424/301 |
| Andreas__Scholz__VID_20260802_142216161 | 122/92/94 | 597/528/210 |
| Bjoern-Harald_Malluche_Heizraum_Video | 376/169/208 | 1318/1242/471 |
| Christian__Heimes_IMG_4716 | 266/198/125 | 849/602/203 |
| Dietmar_Baudisch_VID_20260731_142741 | 209/109/71 | 771/550/201 |
| Dominik__Lindhorst__PXL_20260731_124618342 | 163/146/58 | 376/351/109 |
| Friedhelm_Bednarz_VIDEO-2026-08-06-17-56-31 | 336/195/195 | 862/678/244 |
| Helmut_Schlierf | 207/170/168 | 502/411/217 |
| IMG_3126 | 124/100/59 | 475/470/188 |
| IMG_3128 | 135/129/64 | 734/672/238 |
| Johannes_Steinhauser_VID_20260804_192842 | 131/70/67 | 608/409/232 |
| Klaus_Rombergg | 116/85/63 | 510/420/140 |
| Leif_Malluche_20260802_115810 | 281/163/224 | 1455/1268/292 |
| Manfred_Hahn_IMG_2663 | 99/90/28 | 461/354/190 |
| Michael_Speth_IMG_4149 | 146/113/72 | 591/536/169 |
| Monika_Adldinger_20260801_174346 | 114/54/48 | 664/545/214 |
| Monika__Mulock_IMG_6278 | 114/55/65 | 650/546/167 |
| Moritz__Schneider_IMG_7306 | 144/86/94 | 610/591/252 |
| Renate_Hefele_20260802_180013 | 181/112/123 | 327/299/120 |
| Ulli_Roessle_Heizkeller_Lechermann | 302/252/82 | 690/442/249 |
| WhatsApp_Video_2026-07-05_at_17.41.45 | 345/159/124 | 566/444/350 |
| WhatsApp_Video_2026-07-05_at_17.41.49 | 216/154/106 | 452/389/326 |
| WhatsApp_Video_2026-07-09_at_09.07.37 | 53/42/44 | 514/502/145 |
| Wolfram_Koestler_VID-20260801-WA0002 | 141/60/123 | 498/439/128 |

All heights are `[LOW CONFIDENCE]`/lower-bound flags per
`room_dims.py`'s own reliability gates — expected at these iteration
counts, not a new issue.

### Next step

Re-run both arms at normal iteration budgets (30k steps) on a small subset
first (e.g. IMG_3126 + 09.07.37, the two videos with real tape-measured
ground truth) to get one trustworthy NeRF-vs-3DGS-vs-SfM comparison before
deciding whether it's worth the ~40-80x compute cost to do that for all 24.

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
