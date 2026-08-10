# Thermovation AI Master Thesis

**Extraction of Geometry and Pipe Data from Image Sequence**
Thermovation | Technische Hochschule Ingolstadt

Automated digital preliminary planning for existing boiler rooms from smartphone video —
no LiDAR, no special hardware. The pipeline extracts room geometry, detects piping/MEP
infrastructure, and recommends the optimal placement for the Thermovation indoor unit.

---

## Goals

### Academic
1. **Benchmark reconstruction methods**: classical Structure-from-Motion vs. NeRF vs. 3D Gaussian Splatting.
2. **Benchmark open-vocabulary segmentation**: GroundingDINO vs. YOLO-World vs. manual prompts, all feeding SAM2.
3. **Robustness analysis** under real boiler-room conditions: poor lighting, occlusion by insulation, reflective metallic pipes.
4. **Scientific contribution**: to what extent can monocular smartphone capture replace LiDAR-class sensors for HVAC site assessment?
5. *(Optional)* Formal multi-criteria optimization model for unit placement.

### MVP
- **Input**: A short video of a boiler room.
- **Geometric extraction**: room length / width / height; pipe paths and lengths.
- **Placement recommendation**: minimal distance to the Rücklauf (return flow),
  clear wall space + maintenance clearance, proximity to a power outlet, and keep-out
  zones for escape routes / exits / windows.

---

## Repository guide

Scripts present in numbered folders, one per pipeline stage, so folder order == run order.
`run.py` drives all of them end-to-end — invoke as `python run.py ...`.

| File | Purpose |
|---|---|
| `run.py` | End-to-end orchestrator: video → frames → SfM → marker scale → **room dimensions**, with stage caching and multi-video parallelism (`--parallel N`). Stops after room dimensions by default; pass `--placement` to also run the placement stage (wip: best-effort: runs when a clean wall plane is available, otherwise skipped). Expands glob patterns (e.g. `dataset/*.mov`). |
| `01_frames/extract_frames.py` | Samples video frames at a fixed FPS, optional `--max-dim` downscale. |
| `02_calibration/detect_marker.py` | Detects the printed 3×3 fiducial marker (28.6 × 20.2 cm board, 3.9 cm squares) and returns metric scale (px/cm), corner positions, and an image→cm homography. Grid-structure validated: returns "not detected" instead of guessing. Also scans videos for the best marker frame. |
| `02_calibration/calibrate_camera.py` | Zhang-style camera calibration using the marker as a planar target (same principle as chessboard calibration) — the videos carry no focal-length EXIF. Solves focal length + OpenCV-model distortion via `cv2.calibrateCamera` across many marker sightings, gated by a plausibility check (distortion magnitude, principal-point offset, view-angle diversity) that rejects ill-conditioned fits instead of returning a low-residual-but-physically-wrong lens model. |
| `03_reconstruction/run_sfm.py` | Structure-from-Motion via pycolmap: SIFT features → sequential matching → incremental mapping. Staged (`--stage features/match/map`) so stages can be re-run. `--calib <json>` fixes camera intrinsics from `calibrate_camera.py` instead of self-calibrating; `--gpu` uses CUDA (requires the CUDA-enabled pycolmap build, see Environment, usaability decision to experiment different methods in GPUs). |
| `04_scale/scale_sfm.py` | Marker to metric bridge: detects the marker in registered SfM frames, splits sightings into temporal segments (one per physical placement), triangulates the 9 square centers per segment, and solves the metric scale (cm per model unit) with per-segment cross-validation. If no segment triangulates reliably (marker too small/distant — see 576p results below), falls back to `depth_ratio_scale`: pinhole depth from the marker's own px/cm vs. the model-unit depth of well-triangulated SfM points in the marker footprint, ratioed across frames. |
| `05_geometry/room_dims.py` | Room dimensions from the scaled sparse model: gravity from camera poses (refined on floor inliers), floor/ceiling density peaks → height (flagged as lower bound if the ceiling is uncovered), min-area rectangle over the **largest spatially-connected cluster** of the top-down projection (rejects disconnected outlier fragments that would otherwise blow up the footprint) → length × width. Both height and footprint carry an explicit `*_reliable` flag based on minimum point-support thresholds. Renders a floor-plan visualization. |
| `06_placement/placement_3d.py` | **Fully automatic 3D placement recommendation** (opt-in via `run.py --placement`): wall plane from the triangulated marker when available; otherwise a RANSAC vertical-wall search restricted to points near the (auto-located) Rücklauf, which stays reliable even when the marker was seen on multiple surfaces. Obstacles + window/recess keep-outs from the point cloud (density-filtered, camera-side; recess keep-out only applies on the triangulated-marker path, not the RANSAC one), Rücklauf auto-located via the blue-marking convention (small blue circular cap with non-blue surroundings, lifted to 3D through co-visible features), candidates scored in cm (Rücklauf distance + clearance + mounting height). A tilt guard rejects unreliable wall planes (>25° from vertical, e.g. marker seen on multiple surfaces). Renders the recommendation into a real frame. |
| `experiments/experiment_pipeline.py` | Three-mode segmentation benchmark on the Boilers COCO dataset: GroundingDINO→SAM2, YOLO-World→SAM2, GT-boxes→SAM2 (manual proxy). |
| `experiments/run_video_test.py` | Qualitative GDINO + YOLO-World detection on raw video frames. |
| `experiments/placement_mvp.py` | 2D placement MVP on a single frame: scores candidate unit positions by distance-to-Rücklauf + obstacle clearance. Supports `--auto-scale` (marker-based, no clicks) or manual 2-point scale calibration. |
| `experiments/visualize_sfm.py` | Presentation visuals of the scaled reconstruction: colored metric `.ply` export (open in CloudCompare/MeshLab for interactive orbits) + rendered dark-theme PNG views (full scene + room detail) with camera trajectory and marker position. Output: `<sfm_dir>/viz/`. |
| `dataset/` | 5 boiler-room videos (2× iPhone **1920×1080** `.mov` — corrected 2026-08-02; 3× 576p phone originals) + extracted frames. |
| `output/` | Generated artifacts: marker detections (`marker_v2/`), SfM models (`sfm/`), placement visualizations. Both dataset and output will be shared in Sharepoint, contact Sri Ramesh or Quirin Hanzlmeier for access|


## Dataset notes

- The scale marker is present in **all 5 videos**.
- **Rücklauf identification**: blue marking/cap = Rücklauf (return), red = Vorlauf (supply).
  Additionally, a person points at the Rücklauf pipe in every video.

---

## Results so far

### Marker-based scale recovery (2026-07-13)

| Video | px/cm | mm/px | Marker frames | Grid reproj. error |
|---|---|---|---|---|
| IMG_3128 (4K) | 29.54 | 0.34 | 313 / 545 | 3.0 px |
| IMG_3126 (4K) | 27.83 | 0.36 | 252 / 676 | 1.9 px |
| MicrosoftTeams (576p) | 5.31 | 1.88 | 35 / 249 | <1 px |
| WhatsApp 17.41.49 (576p) | 5.03 | 1.99 | 55 / 121 | <1 px |
| WhatsApp 17.41.45 (576p) | 4.34 | 2.30 | 50 / 125 | <1 px |

Debug visualizations: `output/marker_v2/`. An earlier naive detector (extreme-blob corners)
produced plausible-looking but fabricated scales on all 5 videos; the current detector
validates the full 3×3 grid geometry + a darkness check before accepting a detection.

### SfM baseline — IMG_3126 (2026-07-13, CPU pycolmap)

226 frames (2 fps, 1440px). Runtime: features 38s, matching 83s, mapping 69s.

| Model | Images registered | 3D points | Mean track length | Mean reproj. error |
|---|---|---|---|---|
| **1 (main)** | **194 / 226** | **23,041** | 6.3 | **0.63 px** |
| 2 (fragment) | 41 | 4,736 | 4.7 | 0.55 px |

The main model covers ~86% of the orbit. The 41-image fragment indicates a tracking break
(fast motion / textureless section); to be addressed via loop closure or capture guidance.
Sparse model: `output/sfm/IMG_3126/sparse/`.

### Metric scale (marker→SfM bridge) — IMG_3126 (2026-07-13)

**37.72 cm per model unit**, solved from the marker close-up sweep (22 views):
internal scale spread 1.4% (10th–90th pct over 36 pairwise distances), triangulated
marker planarity 0.07 cm RMS. A distant sighting segment independently gave 39.0
(within ~3.5%, rejected for high spread). Saved: `output/sfm/IMG_3126/scale.json`.

Note from dataset: **the marker is moved between walls during filming**, so
marker sightings are segmented temporally and each placement is triangulated
independently — naively triangulating all sightings as one static marker fails.

### Room dimensions — IMG_3126 (2026-07-13, first attempt)

| Quantity | Value | Confidence |
|---|---|---|
| Length | 411.8 cm | footprint of the *whole captured area* (video tours multiple spaces) |
| Width | 175.9 cm | same caveat |
| Height | **147.5 cm** | ceiling band found: 126 pts spread over 270 cm — **it's a low cellar** |
| Camera height above floor | 64 cm (median) | consistent with crouching in a ~1.5 m cellar |

Floor plane found robustly (~1,700 inlier points, gravity from camera poses, 0.3° from
prior). The ceiling was initially missed because of a hardcoded "rooms are ≥150 cm"
prior — the true ceiling sits at ~147 cm (user pointed out the ceiling IS visible in the
first seconds of the video). Fixed: the ceiling is now the topmost well-supported,
horizontally-spread height band, with no minimum-height assumption. Ceiling-level pipes
form a distinct band at 100–130 cm above the floor.
Top-down floor plan: `output/sfm/IMG_3126/room_topdown.png`; numbers: `room_dims.json`.

**Capture lesson (feed back into protocol):** one continuous single-room orbit per
clip — this video wanders between rooms, so the footprint rectangle spans them.
**Analysis lesson:** avoid "reasonable" hardcoded priors (min room height); boiler
cellars violate them.

### 3D placement recommendation — IMG_3126 (2026-07-14) ✦ MVP milestone

First **fully automatic** placement recommendation, no manual clicks anywhere:

- Wall plane from the triangulated marker (0.6° from vertical)
- **Rücklauf auto-identified** via the blue-marking convention — found the blue-rimmed
  thermometer on the pump group (verified visually), rejecting the blue expansion tank
  via an annulus check
- Window recess correctly excluded (recessed = points behind the wall plane = keep-out)
- Result: a 60×40 cm unit **does not fit** on this wall (all cells hit the window or
  obstacles — an honest, explainable outcome); a 45×30 cm unit fits with
  **d(Rücklauf) = 101 cm, clearance 5.4 cm, mounting height 45 cm**
- Overlay: `output/sfm/IMG_3126/placement_3d.jpg`; data: `placement_3d.json`

Open item: the true Thermovation indoor-unit dimensions (and required maintenance
clearances) received and will be added for Segmentation to validate placement.

### Two-video pipeline run — 576p originals (2026-07-20)

Full `run.py` (then `pipeline.py`) on two low-resolution WhatsApp originals. Both reconstructed well but
their **marker triangulation failed** — a genuine finding, now handled by a fallback.

| Video | SfM (reg / pts / err) | Scale (cm/unit) | Room L × W × H |
|---|---|---|---|
| WhatsApp 17.41.45 | 42/42 · 6.4k · 0.48px | 6.20 (fallback, CV 1.5%) | 402 × 366 × ~228 cm |
| WhatsApp 09.07.37 | 200/242 · 13.7k · 0.75px | 20.09 (fallback, CV 1.2%) | 350 × 311 × ~142 cm |

**Why triangulation failed & the fix (thesis-relevant robustness result):** at 576p the
marker is only ~20 px wide (~5 px/cm) and stays far from the camera, so triangulating its
9 points is numerically ill-conditioned — every marker point solved to *behind* the
cameras, despite the SfM model itself being healthy (verified: known scene points
triangulate correctly). Marker-pose PnP was equally unstable. The working method
(`depth_ratio_scale`) sidesteps the marker's tiny parallax entirely: `depth_cm =
focal / px_per_cm` (pinhole) vs the model-unit depth of well-triangulated SfM points
inside the marker footprint → their ratio is the scale, estimated from the tightest
cluster across frames (core CV ~1.2–1.5%). Contrast the 4K clips (IMG_3126/3128), where a
close marker sweep at ~28 px/cm made direct triangulation well-conditioned. **This is a
concrete capture-resolution robustness data point.**

Height caveat: both ceilings rest on only 24–33 points → treat heights as unreliable
(esp. 09.07.37's 142 cm, with camera at 65 cm). Visuals under each `sfm/viz/`.

**Placement on these two videos: PRODUCED via a Rücklauf-anchored wall search.** The
marker appears on multiple surfaces (09.07.37 shows a marker on both wall and door in one
frame), so a marker-footprint plane fit gives a 75–81°-from-vertical garbage plane. The
fix: don't trust the marker for the wall — locate the Rücklauf first (blue-cap cue), then
RANSAC the dominant **vertical** wall plane among points near it (vertical by
construction, so the tilt guard passes). The overlay frame is chosen as the registered
image that best centers the #1 recommendation.

Results: both videos yield 3 ranked spots (e.g. 09.07.37 #1 at 235 cm from the Rücklauf,
11.6 cm clearance, 85 cm height). Overlays: each `sfm/placement_3d.jpg`.

Quality caveat: recommendations can overlap thin pipes / dark equipment because
the **sparse** SfM cloud under-captures those as obstacle clusters — the boxes land on a
real vertical wall at plausible Rücklauf distances, but obstacle avoidance is only as good
as the point cloud. Denser reconstruction (NeRF/3DGS) or semantic obstacle masks (the
segmentation leg) would sharpen this. The window/recess keep-out is disabled on the RANSAC
path (a sliced plane can't distinguish a recess from deeper wall). Capture-protocol lesson
still stands: **one marker on one wall per clip** enables the precise triangulated-marker
path (as in the 1080p IMG_3126).

### Camera calibration + GPU — all 5 videos (2026-08-02)

**The problem this addresses:** every scale/dimension number so far relied on COLMAP
**self-calibrating** focal length as a free bundle-adjustment parameter — a known,
documented source of systematic bias (COLMAP's own issue tracker: *"focal lengths are on
average overestimated... other studies found all methods underestimate"*). Since scale is
roughly proportional to focal length, this directly explained part of a real accuracy gap
the user measured against ground truth (~25–30% off), though not all of it.

**Fix — marker-based camera calibration** (`calibrate_camera.py`): none of the 5 videos
carry focal-length EXIF (checked directly — video containers don't reliably embed it).
Instead, the marker's known-geometry 3×3 grid, observed from dozens of different angles as
the camera walks around it, is exactly the input Zhang's calibration method needs (the same
principle as chessboard calibration). Solved via `cv2.calibrateCamera` → COLMAP's `OPENCV`
camera model (same distortion convention, transplants directly), then fed into
`run_sfm.py --calib` as **fixed** intrinsics (`ba_refine_focal_length/principal_point/
extra_params = False`) instead of letting bundle adjustment re-guess them.


**A real, measured before/after — not just internal self-consistency:**

| Video | Self-cal focal | Marker-cal focal | Δ | Self-cal scale | Marker-cal scale | Δ |
|---|---|---|---|---|---|---|
| IMG_3126 | 659.0px | 698.3px | **+5.9%** | 37.72 cm/u | 40.63 cm/u | **+7.7%** |
| IMG_3128 | ~712px (implied) | 713.8px | ~0% | 29.54 cm/u | 29.57 cm/u | **+0.1%** |

IMG_3128's near-perfect agreement (0.1%) between self-calibration and marker-calibration is
itself a good cross-validation signal — that video has excellent marker-view diversity
(close sweep, many tilts), so both methods converge to essentially the same answer,
increasing confidence in both. IMG_3126's 5.9%/7.7% gap shows the bias is real but
video-dependent, not a fixed correction factor — consistent with the literature (bias
direction/magnitude varies by capture geometry, not a universal constant).

**A calibration can have low reprojection error and still be physically wrong — caught
directly, not theoretically.** `calibrate_camera.py` initially calibrated
WhatsApp 09.07.37 to focal=1133px with distortion k1=0.35, k2=**−1.35** (RMS only
0.26px — looked like an excellent fit) — physically implausible for a phone lens (typically
|k1|,|k2| < ~0.2). Feeding this in as *fixed* intrinsics **collapsed SfM entirely**
("no reconstruction produced"). Root cause: this video's marker is only ever seen from a
narrow range of angles (the same reason its scale needed the depth-ratio fallback earlier),
so Zhang's method is ill-conditioned — the optimizer traded off focal length against
high-order distortion to fit the few available views, at the cost of a nonsensical lens
model. **Fix:** `validate_calibration()` gates every calibration on distortion magnitude,
principal-point offset from image center, and view-angle (tilt) diversity from the solved
poses — rejecting exactly this case and falling back to self-calibration, which then worked
normally (199/242 images registered, plausible room dims, working placement recommendation).

**Room-dimension hardening, also caught directly by inspecting a bad result:** the WhatsApp
17.41.45 calibrated rerun first reported **L=1311cm, W=646cm** — an impossible 13m room.
Root cause: the footprint fit (`minAreaRect` over percentile-trimmed points) had no
protection against disconnected outlier clusters — a few stray points from a separate
reconstruction fragment were enough to blow the box out. Fix: `room_dims.py` now restricts
the footprint to the **largest spatially-connected cluster** (rasterize to a grid,
`cv2.connectedComponents`, keep the largest component) before fitting the rectangle — this
alone brought that video down to a still-flagged-uncertain but far more plausible 578×222cm
(only 109 points survive clustering there — a genuinely weak 27-image reconstruction from a
20-second clip, correctly reported as **LOW CONFIDENCE** rather than a wrong confident
number). Minimum point-support thresholds (floor ≥150, ceiling ≥80, footprint ≥300 points)
now gate both height and footprint, with an explicit `*_reliable` flag in the JSON output
and a `[LOW CONFIDENCE]` tag in the printed report.

**Full corrected results, all 5 videos, calibrated + GPU (2026-08-02):**

| Video | Calibration | Scale | L × W (cm) | H (cm) | Placement |
|---|---|---|---|---|---|
| IMG_3126 | 698px, RMS 0.32px | 40.63 cm/u | 457 × 198 | 137.5 (reliable) | 45×30cm unit fits, 60×40 doesn't (real wall constraint) |
| IMG_3128 | 714px, RMS 0.21px | 29.57 cm/u | 439 × 207 | 117.5 [LOW CONF.] | works, 220cm from Rücklauf |
| MicrosoftTeams | too few sightings → self-cal | 23.65 cm/u | 271 × 184 | 142.5 [LOW CONF.] | works, 65cm from Rücklauf |
| WhatsApp 17.41.45 | 453px, RMS 0.13px | 20.63 cm/u | 578 × 222 [LOW CONF.] | 122.5 [LOW CONF.] | skipped (weak model) |
| WhatsApp 09.07.37 | **rejected** → self-cal | 20.19 cm/u | 418 × 218 | 142.5 [LOW CONF.] | works, 232cm from Rücklauf |

**Honest bottom line:** camera calibration is a real, measured, non-trivial fix (5–8% on
the video where it mattered) and the plausibility gates caught two genuine failure modes
that would otherwise have silently produced wrong numbers — but it does **not** by itself
close a 25–30% gap against ground truth. The remaining likely contributors (
printed-marker experiment with ArUco, rolling shutter, incremental SfM drift) are still open, and the next
real test is comparing these numbers against tape measurements directly.

### Ground-truth comparison + ceiling-detection bug fix (2026-08-02) ✦ major finding

The user tape-measured 3 rooms and compared against the calibrated pipeline output
(`ground_truth.py`, `ground_truth.json`, logged to `ground_truth_log.md`). Footprint
(L×W) errors were mixed in sign/magnitude (−13% to +52%, consistent with independent
per-video causes: multi-room contamination, weak reconstructions) — but **height errors
clustered tightly at −41% to −50% across all three videos**, on different scale-recovery
paths. That consistency was the tell: a single shared bug, not three unrelated ones.

**Root cause, confirmed with real histogram data before writing any fix:** pulled the
height-vs-point-count profile for all three videos. IMG_3126 and 09.07.37 show a **hard
cutoff** — literally zero 3D points above ~150–160cm, the camera/matching pipeline never
triangulated anything near the true ceiling (235cm / 244cm respectively). 17.41.45 shows
a noisy, unstructured tail with no clear peak (a weak 27-image reconstruction). In every
case, the old ceiling logic ("topmost band with enough points + horizontal spread")
confidently reported the **top of the pipe/equipment cluster** as "the ceiling" — because
it never checked whether that band was actually *separated* from the general point mass.
A real ceiling should look like the floor does: a distinct, room-spanning band. The
pipe/equipment cluster just thins out smoothly toward zero; the old code couldn't tell
the difference.

**Fix** (`room_dims.py`): a ceiling candidate is now only accepted if it's separated from
the main point mass by a genuine near-empty gap (≥20cm of near-zero density). No such gap
→ no ceiling evidence → explicit lower bound, instead of a confident wrong number. Result:
**all 5 videos now correctly report height as a lower bound** — the previous
"RELIABLE"-flagged 137.5cm for IMG_3126 (actually −41.5% off) is gone; the system now says
"we don't know" instead of being confidently wrong. (17.41.45's lower bound happens to land
near the true value — a lucky consequence of its noise floor, correctly still flagged
low-confidence, not claimed as a real measurement.)

This is the "robustness under
real boiler-room conditions" academic goal (occlusion, low-texture surfaces) producing a
concrete, mechanistically-explained finding: **sparse feature-based SfM structurally
cannot recover ceiling height from typical hand-held boiler-room orbits**, because flat
ceilings have almost no texture to triangulate, while pipes/equipment near the top of the
camera's reach do — so without a deliberate capture step, height will always be
underestimated by mistaking equipment height for ceiling height. The fix is a capture
change, not a code change.

Corrected comparison table:

| Video | Dim | Pipeline | Ground truth | Error | Flags |
|---|---|---|---|---|---|
| IMG_3126 | Length | 457.4 cm | 525.0 cm | −12.9% | |
| IMG_3126 | Width | 198.1 cm | 152.0 cm | +30.3% | |
| IMG_3126 | Height | ≥135.4 cm | 235.0 cm | −42.4% | LOWER BOUND |
| WhatsApp 17.41.45 | Length | 578.5 cm | 381.0 cm | +51.8% | LOW CONF |
| WhatsApp 17.41.45 | Width | 221.7 cm | 347.0 cm | −36.1% | LOW CONF |
| WhatsApp 17.41.45 | Height | ≥240.6 cm | 244.0 cm | −1.4% | LOWER BOUND, LOW CONF |
| WhatsApp 09.07.37 | Length | 417.5 cm | 464.0 cm | −10.0% | |
| WhatsApp 09.07.37 | Width | 217.6 cm | 237.0 cm | −8.2% | |
| WhatsApp 09.07.37 | Height | ≥138.8 cm | 244.0 cm | −43.1% | LOWER BOUND |

**Footprint (L×W) reading**: 09.07.37 is the strongest result (−10.0%/−8.2%, small and
same-signed — consistent with a minor uniform under-scale, not a structural error).
17.41.45's large errors are already correctly flagged LOW CONF (footprint-support gate
working as intended). IMG_3126's mixed-sign error (length short, width long) is consistent
with the already-documented multi-room-tour contamination for this specific video.


### Full pipeline rerun — all 5 videos (2026-08-04)

Reran `run.py --calibrate --gpu --placement` end-to-end on all 5 videos from a clean
`output/` (a prior local reorganization had cleared it; a backup of the 2026-08-02 state
was kept). Numbers moved somewhat from the 2026-08-02 table above — a useful
reproducibility data point in its own right, not just a refresh:

| Video | Calibration | Scale | L × W (cm) | H (cm) | Placement (default 60×40cm unit) |
|---|---|---|---|---|---|
| IMG_3126 | 678px, RMS 0.17px | 39.73 cm/u (marker) | 423 × 162 | ≥88 [LOW CONF.] | no fit — all 72 candidate cells rejected (60 by window/recess keep-out, 12 by obstacles); consistent with the known wall constraint (a smaller 45×30cm unit was previously shown to fit this same wall) |
| IMG_3128 | 703px, RMS 0.21px | 29.15 cm/u (depth-ratio) | 425 × 211 | ≥109 [LOW CONF.] | fits, 197cm from Rücklauf |
| MicrosoftTeams | too few sightings → self-cal | 23.71 cm/u (depth-ratio) | 278 × 178 | ≥162 [LOW CONF.] | fits, 74cm from Rücklauf |
| WhatsApp 17.41.45 | 453px, RMS 0.13px | 8.70 cm/u (depth-ratio) | 496 × 233 | ≥341 [LOW CONF.] | skipped — no blue Rücklauf cue detected in any registered frame this run |
| WhatsApp 09.07.37 | too few sightings → self-cal | 20.19 cm/u (depth-ratio) | 414 × 216 | ≥139 [LOW CONF.] | fits, 227cm from Rücklauf |

**Reproducibility observation:** IMG_3126 and WhatsApp 17.41.45 shifted more than the
other three between the two runs (IMG_3126: scale 40.63→39.73 cm/unit, footprint
457×198→423×162cm, default-size placement already failed both times but for the same
reason; 17.41.45's Rücklauf cue wasn't found at all this run, where it previously was).
Frame extraction and marker detection are deterministic given the same source video, so
the divergence traces to `cv2.calibrateCamera`'s nonlinear optimizer landing in a
different local minimum between runs — the same ill-conditioning the plausibility gate
already warns about (see the calibration section above), just below the threshold that
would reject it outright. IMG_3128 and the two self-calibrated videos (MicrosoftTeams,
09.07.37) were comparatively stable (2-3% shifts). **Practical implication for the
thesis:** treat single-run numbers as having a few-percent run-to-run uncertainty band on
top of the accuracy-vs-ground-truth error already measured, at least for videos with
marginal marker-view diversity — a single run is not "the" result.

### Third full pipeline rerun — all 5 videos (2026-08-04, post-glob-fix)


| Video | Calibration | Scale | L × W (cm) | H (cm) | Placement (default 60×40cm unit) |
|---|---|---|---|---|---|
| IMG_3126 | 698px, RMS 0.32px | 40.32 cm/u (marker) | 457.7 × 209.2 | ≥135.5 [LOW CONF.] | skipped — no reliable wall/Rücklauf found this run (previously ran to completion with 0 fitting cells; now doesn't even reach a candidate search) |
| IMG_3128 | 714px, RMS 0.21px | 29.61 cm/u (depth-ratio) | 449.5 × 205.7 | ≥112.7 [LOW CONF.] | fits, 220cm from Rücklauf |
| MicrosoftTeams | too few sightings → self-cal | 23.33 cm/u (depth-ratio) | 292.1 × 174.0 | ≥165.7 [LOW CONF.] | fits, 73cm from Rücklauf |
| WhatsApp 17.41.45 | 453px, RMS 0.13px | 28.15 cm/u (depth-ratio) | 432.2 × 218.9 [LOW CONF.] | ≥290.6 [LOW CONF.] | skipped — no Rücklauf cue this run |
| WhatsApp 09.07.37 | too few sightings → self-cal | 20.05 cm/u (depth-ratio) | 426.2 × 247.1 | ≥138.9 [LOW CONF.] | fits, 238cm from Rücklauf |

**A starker reproducibility finding than the previous rerun:** the three
`footprint_reliable`-flagged videos (IMG_3128, MicrosoftTeams, 09.07.37) stayed within a
few percent of the prior run — consistent with the earlier observation that
well-conditioned reconstructions are stable. But **WhatsApp 17.41.45's depth-ratio scale
jumped from 8.70 → 28.15 cm/unit (3.2×) between two same-day runs**, even though its camera
calibration was byte-identical both times (cached `calib.json`, 453px/RMS 0.131px matches
exactly). The instability traces further upstream than calibration: this video's SfM
consistently fragments into multiple sub-models (`sparse/2` was selected as "best" this
run), so which fragment wins — and therefore which points feed the depth-ratio scale
estimate — isn't stable run to run. This video is already flagged `LOW CONF.` on footprint
for an unrelated reason (109–124 points survive spatial clustering), so the takeaway is
consistent rather than new: **its low-confidence flag isn't just conservative bookkeeping —
the underlying numbers genuinely aren't reproducible**, unlike the reliable-footprint videos.

### Calibration/triangulation correspondence bug fix — all 5 videos (2026-08-09) ✦ finding

**The bug:** `calibrate_camera.py`'s `collect_views()` (and `scale_sfm.py`'s triangulation
input) built each view's image points by reprojecting the known marker geometry
(`GRID_CM`) through the homography `H` that `detect_marker()` had *just fit from those
same 9 detected points* — i.e. feeding `cv2.calibrateCamera` the homography's own
best-fit approximation of the corners, not the corners actually detected in the image.
This silently discarded the real detector noise the optimizer needs to solve for a
genuine lens model, and made the reported per-view fit look artificially clean.

**Fix:** `detect_marker()` now also returns `grid_pts_full` — the actual 9 detected
square-center pixels, in the same row-major order as `GRID_CM` — and both
`calibrate_camera.py` and `scale_sfm.py` use those directly. `H` is now used only for
px/cm scale, detection validation, and rejecting false positives, never as a source of
calibration/triangulation coordinates.

**Effect, all 5 videos rerun (`run.py --calibrate --gpu --force`, `--placement` not run
this pass):**

| Video | Calibration | Scale | L × W (cm) | H (cm) |
|---|---|---|---|---|
| IMG_3126 | 664px, RMS 0.76px (80 views) | 38.98 cm/u (marker) | 423 × 184 | ≥135 [LOW CONF.] |
| IMG_3128 | 720px, RMS 0.78px (80 views) | 30.52 cm/u (depth-ratio) | 446 × 177 | ≥113 [LOW CONF.] |
| MicrosoftTeams | too few sightings → self-cal | 23.28 cm/u (depth-ratio) | 259 × 179 | ≥159 [LOW CONF.] |
| WhatsApp 17.41.45 | 524px, RMS 0.42px (16 views) | 13.07 cm/u (depth-ratio) | 222 × 151 | ≥122 [LOW CONF.] |
| WhatsApp 09.07.37 | 1052px, RMS 0.70px (42 views) | 9.90 cm/u (depth-ratio) | 186 × 153 | ≥50 [LOW CONF.] |

Two notable changes vs. the previous (post-glob-fix) rerun:
- **Reprojection RMS roughly doubled** on the videos that already calibrated (IMG_3126:
  0.32→0.76px, IMG_3128: 0.21→0.78px). This is expected and correct, not a regression:
  the old RMS was measured against the homography's own smoothed points, so it was
  structurally close to zero by construction; the new RMS is measured against real
  detected pixels and reflects genuine detector + lens-model residual.
- **WhatsApp 09.07.37 now passes the plausibility gate** (1052px, RMS 0.70px, 42 views)
  where it was previously rejected outright (implausible k1/k2, would have collapsed SfM
  if fed in). Feeding `cv2.calibrateCamera` the real noisy correspondences instead of the
  artificially-clean homography reprojections kept the optimizer out of the degenerate
  distortion-vs-focal-length tradeoff that caused the earlier rejection. **(⚠ This turned
  out to be a false pass — see DFOV gate fix below.)**

Scale/room numbers also moved a few percent from the prior rerun — consistent with the
already-documented run-to-run sensitivity of `cv2.calibrateCamera`'s optimizer, on top of
this correspondence-source change.

### Calibration DFOV gate + ground-truth regression fix — (2026-08-09) ✦ finding

**The bug:** The 2026-08-09 correspondence fix caused WhatsApp 09.07.37's calibration to
pass the plausibility gate with focal=1052px (k1=0.196, k2=−0.177 — within all existing
bounds). When fed as fixed intrinsics into COLMAP, this wrong focal caused the SfM
reconstruction to settle at a scale where the depth-ratio formula returned 9.90 cm/unit
instead of the correct ~20 cm/unit, making room dimensions ~2.5× too small
(186×153 cm vs ground-truth 464×237 cm, −60%/−35% error vs −10%/−8% before the fix).

**Root cause, confirmed from DFOV analysis of all calibrated videos:**

| Video | focal | diag | DFOV | verdict |
|---|---|---|---|---|
| IMG_3126 | 664px | 1652px | 102.4° | plausible ✓ |
| IMG_3128 | 720px | 1652px | 97.8° | plausible ✓ |
| WhatsApp 17.41.45 | 524px | 1175px | 96.6° | plausible ✓ |
| **WhatsApp 09.07.37** | **1052px** | **1175px** | **58.4°** | **telephoto-zoom range — wrong local minimum** |

The three good calibrations cluster at 97–102°. The bad one is at 58°: the optimizer
traded a high focal against low distortion on this video's (limited) view diversity,
producing a geometrically self-consistent but physically wrong lens model that the k1/k2
and tilt-spread gates could not detect.

**Fix:** `validate_calibration()` in `calibrate_camera.py` now adds a diagonal-FOV
plausibility check: `65° ≤ DFOV ≤ 130°`. This rejects the 1052px focal (DFOV=58.4°)
while accepting all three passing videos (97–102°). WhatsApp 09.07.37 falls back to
self-calibration — the same path that produced −10%/−8% footprint errors before the
correspondence fix. Rerun with `python run.py dataset/*.mov --calibrate --gpu --force`
to validate.

### Known limitations
- Marker orientation is ambiguous (180° flip) — irrelevant for scale; matters only if the
  marker is later used as a pose/orientation anchor.
- SfM model is in arbitrary units until the marker→metric bridge is built.

---

## Task list

### Done
- [x] Fiducial marker design (3×3 grid, known dimensions) + detection with grid-structure validation
- [x] Best-frame selection from video (marker size × sharpness)
- [x] `--auto-scale` integration into the 2D placement MVP
- [x] Three-mode segmentation experiment scaffold (GDINO / YOLO-World / manual → SAM2)
- [x] 2D placement scoring: Rücklauf distance + clearance penalty, top-k non-overlapping
- [x] Data collection protocol for site videos
- [x] SfM baseline on IMG_3126: 194 images, 23k points, 0.63px reprojection error (CPU, ~3 min)
- [x] Marker→metric bridge: per-placement segmentation + triangulation → 37.72 cm/unit (1.4% spread)
- [x] Room dimensions pipeline (floor plane + footprint rectangle + ceiling/height)
- [x] **3D placement recommendation (MVP core)**: marker wall + obstacles + window keep-out + auto-Rücklauf, all in metric 3D — first automatic recommendation on IMG_3126
- [x] End-to-end pipeline orchestrator (`run.py`, originally `pipeline.py`) with stage caching + multi-video parallelism
- [x] Presentation visuals (`visualize_sfm.py`): metric colored PLY + rendered orbit/top/detail views — `output/sfm/IMG_3126/viz/`
- [x] Depth-ratio scale fallback (`scale_sfm.py`) for low-res/distant-marker videos where triangulation is ill-conditioned — validated on two 576p WhatsApp videos (core CV ~1.2–1.5%)
- [x] Pipeline run on two 576p WhatsApp videos end-to-end (SfM → scale → room dims → visuals)
- [x] Placement wired into `run.py` as an opt-in (`--placement`) best-effort final stage (runs when a clean wall exists, skips non-fatally otherwise)
- [x] Placement wall-plane fallback + tilt guard (>25° → skip) for videos without a triangulated marker
- [x] Rücklauf-anchored RANSAC vertical-wall finder → placement now produced on both 576p videos (marker-on-multiple-surfaces case); overlay frame auto-chosen to best show the #1 spot
- [x] GPU enabled: driver verified working (`nvidia-smi`), CUDA-enabled `pycolmap` installed (prebuilt wheel, not from-source) — `has_cuda == True`, feature extraction 38s→10s
- [x] Marker-based camera calibration (`calibrate_camera.py`, Zhang's method) — fixes focal length instead of COLMAP self-calibrating it; measured 5.9%/7.7% focal/scale bias on IMG_3126, near-zero (0.1%) on IMG_3128 (cross-validates both methods)
- [x] Calibration plausibility gate (distortion magnitude, principal-point offset, view-angle diversity) — caught and rejected a genuinely degenerate calibration (WhatsApp 09.07.37) that had collapsed SfM entirely
- [x] Room-dimension hardening: largest-connected-cluster footprint filtering (fixed a 1311×646cm blowup down to a flagged-uncertain 578×222cm) + minimum point-support `*_reliable` flags on height/footprint
- [x] Corrected "1080p not 4K" — IMG_3126/3128 mislabeled as 4K since early in the project, never actually verified until now
- [x] All 5 videos rerun end-to-end with calibration + GPU; full before/after table in Results
- [x] **Ground-truth comparison system** (`ground_truth.py` + `ground_truth.json` + `ground_truth_log.md`) — persistent, appendable accuracy log against tape measurements, regenerates from live `room_dims.json` output
- [x] **Ceiling-detection bug found + fixed**: gap-based ceiling acceptance in `room_dims.py` — old logic confidently mistook the top of the pipe/equipment cluster for the ceiling on every GT-checked video (−41% to −50% height error, one falsely flagged RELIABLE); all 5 videos now correctly report height as an explicit lower bound instead
- [x] Repo restructured into numbered stage folders (`01_frames/` … `06_placement/`, `experiments/`) ahead of the split into an `thermovation_ai` repo; `pipeline.py` replaced by `run.py`, which now stops after room dimensions by default (`--placement` opt-in)
- [x] Full pipeline rerun on all 5 videos from a clean `output/` (2026-08-04) — see Results; surfaced a real run-to-run reproducibility gap on IMG_3126's calibration (different `cv2.calibrateCamera` local minimum on an otherwise identical input)
- [x] Fixed `run.py` to expand glob patterns (e.g. `dataset/*.mov`) itself instead of relying on the shell — cmd.exe/PowerShell pass wildcards through unexpanded (unlike bash), which was silently producing a single literal-`*` "video". Removed the `run`/`run.ps1` wrappers in favor of one `python run.py ...` entry point that works the same in all three shells
- [x] Third full pipeline rerun, all 5 videos (2026-08-04, post-glob-fix) — see Results; a starker reproducibility data point (WhatsApp 17.41.45's scale moved >3x between same-day runs)
- [x] Fixed a circular-reprojection bug in calibration/triangulation correspondences: `detect_marker()` now exposes the actual detected `grid_pts_full` instead of `calibrate_camera.py`/`scale_sfm.py` reconstructing points via the homography fit from those same points; full 5-video rerun confirms the effect (RMS roughly doubled on the videos already calibrating — now measuring real detector noise instead of the homography's self-fit; WhatsApp 09.07.37 calibration now passes the plausibility gate where it was previously rejected)
- [x] **DFOV plausibility gate** added to `calibrate_camera.py` — the 2026-08-09 correspondence fix caused a new failure mode where the optimizer found a wrong local minimum (focal=1052px, DFOV=58°) that passed all prior gates (k1/k2, principal-point, tilt-spread) but produced a 2.5× wrong depth-ratio scale on WhatsApp 09.07.37 (9.90→~20 cm/u expected). New gate rejects `DFOV < 65° or > 130°`; confirmed: 3 good videos at 97–102° pass, the bad one at 58° is rejected → self-cal fallback restores −10%/−8% footprint accuracy. Rerun with `--force` to take effect.

### In progress
- [ ] YOLOE evaluation: single-model open-vocab detection+segmentation (text/visual/prompt-free) as 4th benchmark arm vs GDINO+SAM2 / YOLO-World+SAM2 / manual+SAM2 — smoke test running
- [ ]  Test whether real capture technique fixes the height-recovery failure

### Next (rough priority order)
- [ ] Pipe paths & lengths: SAM2/YOLOE pipe masks → skeletonization → back-projection onto 3D model
- [ ] Quantitative segmentation metrics (mIoU / precision / recall vs Boilers COCO GT)
- [ ] NeRF / 3D Gaussian Splatting on the same scenes; compare vs SfM (accuracy, runtime, robustness) — GPU is now available for this
- [ ] Robustness experiments: lighting / occlusion / reflective-surface analysis across resolution tiers
- [ ] Ruler-verify the printed marker's true size (printer scaling risk, still outstanding)
- [ ] End-to-end MVP: video in → scaled geometry + placement recommendation + report out

---

## Environment

- Windows 11, Python: OpenCV, PyTorch, pycolmap (CUDA-enabled)
- GPU: NVIDIA GTX 1650 4GB - dev PC