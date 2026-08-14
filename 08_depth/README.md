#  — Monocular depth models: 

Two independent questions, both using the same five depth models:

1. **Ceiling fix**: does dense per-pixel depth recover a ceiling estimate
   where sparse SfM structurally cannot (IMG_3126 and 09.07.37 both show a
   hard cutoff — zero SfM points above ~150-160cm, see the main README's
   ground-truth section)? → `ceiling_check.py`
2. **Scale alternative**: can a depth model's metric predictions replace
   the marker (Zhang calibration) as the source of `cm_per_unit`, and how
   does it compare? → wired into `04_scale/scale_sfm.py --scale-method
   monocular_depth`, using `depth_scale.py`

Both reuse the *existing* SfM camera poses (question 1 needs the existing
marker-based scale to already be trustworthy for that video; question 2 is
exactly testing whether that dependency can be removed). Both are judged by
code shared with the rest of the reconstruction-method work:
`compute_room_dims()` for question 1 (the same function SfM, 3DGS, and NeRF
all use — see `3d_recons_alternatives/`), and the same ratio-based
aggregation `depth_ratio_scale` already uses for question 2.

## 0. One-time setup

```
pip install depth-anything-3   # DA3Metric; already added to requirements.txt
git clone https://github.com/metric-anything/metric-anything.git /tmp/metric-anything-repo
```

DepthAnythingV2-Metric, ZoeDepth, and DepthPro all load via `transformers`
(already installed for GroundingDINO) — no extra setup for those three.
MetricAnything needs its repo on the Python path:
`export PYTHONPATH=/tmp/metric-anything-repo:$PYTHONPATH` (or the Windows
equivalent) before running either script with `--depth-model metric_anything`.

**Model IDs/APIs here were verified against public docs/model cards as of
2026-08 (DA3 was published 2026-03-04, MetricAnything 2026-01-29 — both
very recent), but check `08_depth/depth_models.py`'s docstrings for what to
verify against the actual repo/model card the first time you run each one.**

## 1. Ceiling check (per video, per depth model)

```
python 08_depth/ceiling_check.py \
    --sfm-dir output/sfm/IMG_3126 --frames-dir dataset/frames/IMG_3126_sfm \
    --model 1 --depth-model depthanything_v2_metric \
    --out 08_depth/results/IMG_3126_depthanything_v2_metric
```

Adjust `--sfm-dir`/`--frames-dir` to wherever `run.py`'s output actually
landed. Repeat for `09.07.37`, and repeat across `--depth-model`
{`depthanything_v2_metric`, `zoedepth`, `depthpro`, `metric_anything`,
`depth_anything_v3`} — each is independent, run whichever you have time
for; DepthAnythingV2-Metric is the cheapest to start with since it needs
no extra install.

Prints the existing SfM-only height for comparison, then the
monocular-depth height, then an explicit **Verdict** line (recovers a
ceiling SfM couldn't / no improvement / SfM was already fine), and saves
`depth_room_dims.json` + `depth_room_topdown.png` per run — this is the
"document the metrics" deliverable, one JSON per (video, model) pair, all
directly comparable since they share the same `compute_room_dims()` fields
as `room_dims.json`.

**If the verdict comes back "no improvement" — that's still the required
deliverable, not a dead end.** Document why using what's already in the
output: low `points_used` after gravity-fitting (poor camera coverage of
the ceiling in the source video, not a model failure) vs. a ceiling band
that got detected but rejected by the same confidence thresholds
`room_dims.py` already applies (genuinely marginal evidence) vs. no gap
found at all (the depth model's ceiling predictions are noisy/inconsistent
across frames, blending into the equipment mass the same way SfM's sparse
points did). Each is a distinct, real robustness finding.

## 2. Scale comparison (per video, per depth model)

```
python 04_scale/scale_sfm.py output/sfm/IMG_3126/sparse/1 dataset/frames/IMG_3126_sfm \
    --scale-method monocular_depth --depth-model depthanything_v2_metric
```

Writes `scale_monocular_depth_depthanything_v2_metric.json` (does **not**
touch `scale.json`, which the rest of the pipeline reads). Compare its
`cm_per_unit` directly against:

- The video's existing `scale.json` (whatever `run.py --calibrate` already
  produced — marker triangulation or depth-ratio fallback, both Zhang-
  calibration-dependent).
- A forced re-run of each individual method on the same video, for a clean
  three-way comparison with nothing else changing:
  ```
  python 04_scale/scale_sfm.py output/sfm/IMG_3126/sparse/1 dataset/frames/IMG_3126_sfm --scale-method marker
  python 04_scale/scale_sfm.py output/sfm/IMG_3126/sparse/1 dataset/frames/IMG_3126_sfm --scale-method depth_ratio
  python 04_scale/scale_sfm.py output/sfm/IMG_3126/sparse/1 dataset/frames/IMG_3126_sfm --scale-method monocular_depth --depth-model depthanything_v2_metric
  ```
  (`--scale-method marker`/`depth_ratio` also write to their own
  `scale_<method>.json`, not `scale.json` — see `scale_sfm.py`'s
  `--scale-method` help text.)

Each run prints `frames_used`/`points_used` and a `core_cv_pct` (tightness
of the robust-aggregated ratio cluster) — a monocular-depth estimate with
high CV% relative to marker triangulation's own spread is a real precision
finding, not just a headline cm/unit number to compare.

### 2b. Apply the depth-derived scale to get actual room dimensions

A scale number alone doesn't answer "does this give me the right L×W×H" —
apply it to the *same* SfM point cloud (`room_dims.py`'s `--scale-json`
override swaps only the scale, nothing about the geometry, so this isolates
whether the scale source specifically is doing the work):

```
python 05_geometry/room_dims.py output/sfm/IMG_3126 --model 1 \
    --scale-json output/sfm/IMG_3126/scale_monocular_depth_depthanything_v2_metric.json
```

Writes `room_dims_scale_monocular_depth_depthanything_v2_metric.json` (and
matching `room_topdown_*.png`) alongside the existing marker-based
`room_dims.json` — never overwrites it. This is the number to put next to
`room_dims.json`'s length/width/height, and eventually your own ground
truth, for the actual per-video comparison. Repeat for `--scale-method
marker`/`depth_ratio`'s own `scale_<method>.json` files too, if you want
all four scale sources' room dimensions side by side for one video.

## What "success" looks like here

Same framing as `3d_recons_alternatives/README.md`: a clear yes on either
question is the headline result (ceiling recovered, or scale
Zhang-independent and comparably precise). A clear no, with documented
metrics explaining why, is still exactly what 9.2b asked for — a
robustness finding, not a failed task.
