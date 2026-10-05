# 2D wall rectification — placement from one photo per wall

A cheap alternative to the 3D placement chain (`06_placement/placement_3d.py`),
which needs a full SfM+MVS video scan. Here each wall gets **one still photo**,
rectified to metric scale with a plain 2D homography (4 clicked corners plus the
wall's known width and height). Obstacle detection, Rücklauf localization and the
mounting-spot search all run directly in that flat image. There's no point cloud.

The method, as specified:

1. Click each wall's 4 corners and apply a homography to get a metric canvas
   (`PX_PER_CM = 4`).
2. Sanity-check the scale against objects of known size: ArUco markers (15 cm)
   and the boiler (56 × 129 cm) / buffer tank (181 cm).
3. Detect obstacles with GroundingDINO and refine them to SAM2 masks.
4. Find the Rücklauf by HSV cap colour (blue = Rücklauf, red = Vorlauf),
   preferring a blue blob with a red one next to it.
5. Build an occupancy grid, inflate obstacles by a service clearance and
   grid-search the unit footprint. Score candidates by distance to the Rücklauf,
   or by clearance if there's no Rücklauf.
6. Output the top 3 candidates as an overlay plus text, e.g.
   `#1  110 cm from floor, 45 cm from left corner, 38 cm from Rücklauf`.

Limitations accepted up front: a flat photo has no depth, so a pipe standing
10 cm proud of the wall still "looks free". This is mitigated with extra
clearance, not a depth check. Lens distortion isn't modelled. A Rücklauf on the
boiler or floor rather than a wall isn't handled.

## 1. Files

| File | Purpose |
|---|---|
| `config.py` | Paths, the room's ground truth, per-wall specs, manual overrides. |
| `corners.json` | Clicked wall corners (TL, TR, BR, BL; original-image px), written by `corner_picker.py`. |
| `corner_picker.py` | Local click tool with a 6× magnifier. Pure stdlib web server on `localhost:8765`. |
| `wall_rectify.py` | **The pipeline.** Rectify → sanity checks → detect → Rücklauf → search → overlay. Its functions are reused by the scripts below. |
| `qwen_pipe_detection.py` | Qwen2-VL-2B vs. GDINO on the pipes GDINO missed, run in isolation (§5.1). |
| `qwen_obstacle_placement.py` | Qwen2-VL-2B as the obstacle detector in the full pipeline (§5.1). |
| `vlm.py` | Shared Qwen loading/prompting helpers. |
| `room_dims_qwen_text.py` | Can a VLM estimate the room dimensions itself? 5 prompt variants (§5.2). |
| `room_dims_qwen_pointing.py` | The VLM only points at landmarks and Python does the maths. 3 variants (§5.2). |
| `room_dims_molmo.py` | The same with Molmo, a pointing-specialised VLM (§5.2). |
| `room_dims_solver.py` | Wall height from wall corners + reference objects of known height. |

## 2. Setup

Use the repo's main environment (see the top-level README): `opencv-python`
(4.7+ for `cv2.aruco.ArucoDetector`), `torch` with CUDA, `transformers`, `sam2`,
`python-dotenv`. The Qwen 7B scripts also need `bitsandbytes`. SAM2 is loaded
through `experiments/experiment_pipeline.py` (`sam2.1-hiera-small`); if it fails
to load, box masks are used instead. All models download from HuggingFace on
first use. A 10 GB GPU runs everything, one model at a time.

**Photos.** These are read from `<DATASET_DIR>/Quirin room images/` (`DATASET_DIR`
comes from the repo's `.env`). Override this with `WALL_PHOTOS_DIR`. The set is 4
photos, one per wall: `Image.jfif`, `Image (1).jfif`, `Image (2).jfif` and
`Image (3).jfif`. Outputs go to `output/wall_rectify/`.

**Molmo** needs its own venv. Its `trust_remote_code` model file (Sept 2024)
breaks against transformers 5.x in 3 places, so the venv is pinned to
`transformers==4.45.2`. Patching the shared environment instead would break the
Qwen 2.5 scripts.

```
python -m venv %USERPROFILE%\.venvs\molmo-venv
%USERPROFILE%\.venvs\molmo-venv\Scripts\pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
%USERPROFILE%\.venvs\molmo-venv\Scripts\pip install transformers==4.45.2 accelerate bitsandbytes einops opencv-python pillow
```

(This matches the venv the results were produced with: Python 3.12, torch
2.4.1+cu124, transformers 4.45.2.)

**Ground truth needed before starting** (it lives in `config.py`). Without the
fixture sizes, the scale check would be circular.
- The room is 392 × 249 × 229 cm (L × W × H). The photos alternate width and
  length walls going round the room.
- Fixtures of known size: boiler 56 × 129 cm, tank 181 cm tall.
- Which photo shows which wall, and which walls are adjacent. Here only walls 3
  and 4 share a corner (the tank/boiler corner). Ask for this; don't infer it.

## 3. Running

```
python 2d_wall_rectify/corner_picker.py       # only for a new photo set; corners.json is committed
python 2d_wall_rectify/wall_rectify.py
python 2d_wall_rectify/wall_rectify.py --gdino-model openmmlab-community/mm_grounding_dino_large_all
```

Each run writes `{wall}_rectified.jpg`, `{wall}_overlay.png` and `results.json`
to `output/wall_rectify/<checkpoint>/`. The overlay shows occupancy (red tint),
the Rücklauf and the top-3 candidates, with their distances printed on the
image. **Always check the overlay against the photo.** A plausible-looking
"110 cm from floor" says nothing about whether that spot is actually free.

## 4. What it took to make it work, in order

### 4.1 Corners: click them, don't eyeball them

The first pass estimated corners by eye from a coordinate-grid overlay of each
photo. **Every independent reference failed:** markers measured 30–52 % small,
and the boiler/tank were 40–90 % off. The corner positions were the error, not
the assumed wall sizes. With `corner_picker.py` the marker error dropped to
**1.0–6.0 %**. This was the single biggest fix in the experiment.

Markers are weaker references than the boiler. A few pixels of corner jitter is
a large share of a 15 cm marker, so with eyeballed corners one marker read 52 %
off while the boiler on the same homography read 11–15 % off. Markers that
aren't on the wall plane (e.g. the one taped to the table in wall1's photo)
warp to skewed quads, and a squareness check below 0.85 excludes them
automatically.

### 4.2 Rücklauf: retune HSV for the actual lighting, then accept a manual override

- The standard S ≥ 80 / V ≥ 50 floors are tuned for bright plastic caps. The
  manifold dials in this dim basement sample at **S ≈ 50–57, V ≈ 44–62**, so
  the detector skipped the real fixture and picked an unrelated blue surface.
  The floors are now 35/35. On a new dataset, sample the real fixture's pixels
  first; don't guess.
- A blue blob with a red blob within ~15 cm (the Vorlauf/Rücklauf pair) beats
  a larger lone blue blob.
- Even so, the small manifold dials on wall3 couldn't be isolated from
  JPEG/reflection noise. `MANUAL_RUCKLAUF_PX` overrides the detector for that
  wall. It replaces the detector's result, never averages with it, and the
  detector's own finding is still printed. This is the method's own fallback
  ("if ambiguous, confirm the crop"), not a hack.

### 4.3 Obstacles: missed detections become mounting spots

The score prefers whatever free point is closest to the Rücklauf, and an
undetected obstacle costs nothing. With grounding-dino-base missing the
manifold, boiler and tank on wall3 (and everything on wall4), the search put
the unit on the manifold's valves, then on the boiler's front panel, then on
the tank. `MANUAL_OBSTACLE_BOXES` adds those objects as keep-outs.

GDINO recall was also tuned:
- **Separate `"heating pipe"` prompt.** GDINO grounds the whole concatenated
  prompt as one query, so the phrasing changes the results. Adding it
  recovered a missed hot water tank on wall3.
- **`box_threshold` 0.30 → 0.20.** This surfaces real extra boxes. Empty-label
  junk only starts below ~0.18.
- **IoU dedup per canonical label.** The lower threshold returns overlapping
  boxes for one object (one "tank" measured 14 cm tall against 181 cm).
- `GDINO_MAX_DIM = 1400` isn't a recall limit; full resolution gave identical
  detections.

### 4.4 Smaller fixes

- With zero obstacles, `cv2.distanceTransform` returns ~8.5e37 instead of a
  distance. It's capped at the canvas diagonal.
- Candidate distances are drawn onto the overlay itself, not only printed. When
  two candidates sit close vertically, their labels can overlap (cosmetic).

## 5. Detector and VLM experiments

### 5.1 Obstacle detection: a better GDINO checkpoint, not a different model family

- **moondream2** wouldn't load: its `trust_remote_code` file calls a
  transformers internal that no longer exists in 5.x. Check remote-code models
  against the installed transformers version first.
- **Qwen2-VL-2B, asked to list every pipe in one shot**, degenerated: the same
  box 29 times, or mechanically incrementing coordinates. Repetition penalties
  stopped the loop but broke the JSON. Full-resolution photos OOM 10 GB, so
  images are capped at ~1024 px.
- **Qwen2-VL-2B, one query per object**, returned clean boxes. On wall4 they
  landed on the real copper pipe and gas hose that GDINO missed entirely
  (`qwen_pipe_detection.py`).
- **But in the full pipeline** (`qwen_obstacle_placement.py`) that reversed.
  The same query on the same wall missed a pipe it had found in isolation, so
  a candidate landed on a pipe elbow. One hallucinated wall-sized box marked a
  whole wall occupied. Unqueried objects (two black floor units) were never
  excluded. **A small VLM isn't reliable as the main detector.** It's still
  useful to confirm a single object on request.
- **`openmmlab-community/mm_grounding_dino_large_all`** is Grounding DINO
  retrained on far more data, and a one-line checkpoint swap (`--gdino-model`).
  At the same prompts and threshold it found 33 correctly-placed objects on
  wall4, where base found none, and gave cleaner tank/boiler boxes on wall3.
  Its boxes are sometimes looser than the object, which makes it more
  conservative. For this task that's the right way to err. It still misses
  things, though: §6 shows candidates on insulated pipes and a gas hose.
  (Grounding DINO 1.5/1.6 Pro are API-only, not downloadable.)
- **Neither checkpoint finds the compact valve manifold** (a black box with 4
  round gauges). That's a vocabulary gap, not resolution or model quality: no
  common English word names it. Either test a specific prompt ("valve
  manifold", "distribution manifold") or keep a manual override.

Results of the current code: see §6.

### 5.2 Room dimensions from photos alone

These test whether a VLM can replace the known room size used in §3, given only
the boiler/tank as references.

**Text answers** (`room_dims_qwen_text.py`). All five prompt variants failed:
- **2B, direct JSON:** about 240 cm height on every wall, even walls without
  the references. That's a stock room prior. Widths were 27–54 % off.
- **7B (4-bit) with forced step-by-step reasoning, a one-line plain question
  over all 4 photos, and the structured prompt that worked with Claude Opus:**
  all failed at the arithmetic. They mixed up metres and centimetres, fell back
  to generic-room sizes, and gave reasoning that contradicted the final number.

**Pointing only** (`room_dims_qwen_pointing.py`). The VLM returns pixel
landmarks and Python does all the maths:
- **with-sizes:** the schema also stated each object's cm height. The boiler's
  base→top came back as ~129 px and the tank's as ~180 px, which are the cm
  numbers echoed back as pixels. The two references' scales disagreed by 124 %.
  **Never put the answer's units in a pointing prompt.**
- **decoupled:** no cm values anywhere; sizes are attached in Python. The
  points became honest, but a flat px→cm scale ignores perspective, giving
  25–40 % error on wall3.
- **homography:** also asks for the 4 wall corners and rectifies
  (`room_dims_solver.py`). Qwen returned the image frame edges as the "corners".

**Molmo-7B-D** (`room_dims_molmo.py`) is trained for 2D pointing and re-tests
the corner step, one landmark per call. Its answers are percent coordinates in
`<point x=".." y="..">` tags.

**Solver correction (this port).** The original solver grid-searched the wall's
aspect ratio for the value at which boiler and tank "agreed" on scale, then
reported width and height. Both references are vertical, so changing the
aspect ratio scales them identically: they agree equally at *every* ratio.
**The reported widths were arbitrary.** Height doesn't depend on the ratio, so
it's correct, and it works from a single reference too. On a synthetic wall
with the true 249 × 229 cm the old solver returned 188 × 229. The solver now
returns height only. Width needs a horizontal reference, e.g. the boiler's
56 cm width.

## 6. Results of the current code

Validation run of this code on 2026-10-05 (RTX 3080 10 GB, transformers 5.15).
Re-run the scripts to regenerate the outputs; `output/` isn't committed.

**Scale checks** (`wall_rectify.py`, clicked corners): markers measured
14.1–14.9 cm against 15 cm, i.e. **1.0–6.0 % error** (6 on-plane markers: 5
PASS, 1 WARN at 6.0 %). One off-plane marker was excluded automatically. Wall2
has no reference at all. Detected tank/boiler boxes are poor size references:
partial and loose GDINO boxes measured 11–93 % off the known sizes. Use markers
for the scale check, not detector boxes.

**Placement**, top-3 candidates per wall, both checkpoints run with the manual
overrides from `config.py`:

| Wall | grounding-dino-base | mm_grounding_dino_large_all | Qwen2-VL-2B detector |
|---|---|---|---|
| wall1 | 3 (no Rücklauf → max clearance) | 1 | 3 |
| wall2 | 3 | 3 | 0 |
| wall3 | 0 (fully occupied) | 1 | 3 |
| wall4 | 3 | 3 | 3 |
| detections (w1–w4) | 8 / 16 / 14 / 3 | 29 / 20 / 27 / 33 | 6 / 5 / 5 / 8 |

What checking the overlays shows (counting candidates tells you little):
- **mm_grounding_dino_large_all** has much higher recall (33 vs 3 detections
  on wall4), but **wall3 #1 sits on the black insulated pipes**, and on wall4
  #1 contains a floor valve, #2 overlaps the yellow gas hose and #3 is on the
  black floor units. Better recall alone doesn't make the result trustworthy.
- **grounding-dino-base**, wall1: #2 and #3 are behind the table. "Table"
  isn't in the vocabulary, so it's never excluded.
- **Qwen2-VL-2B** reported a "boiler" on wall1 and wall2, which have none, and
  marked wall2 fully occupied. This matches §5.1.
- **The wall4 Rücklauf is wrong** with every detector. The HSV detector picks
  an unpaired blue blob that maps outside the wall face (candidates are 414–506
  cm "from the Rücklauf" on a 392 cm wall). An unpaired blob should probably not
  be trusted at all. That's an open item.

**Qwen pipe detection** (`qwen_pipe_detection.py`, single-object mode):
3 boxes on wall4 against GDINO-base's 0, but "pipe" and "heating pipe" return
the identical box. List mode returned 11 boxes on wall4 and 1 on wall3.

**Room dimensions** (truth: height 229 cm):

| Script / variant | Wall3 | Wall4 |
|---|---|---|
| `room_dims_qwen_text --variant 2b-json` | height 240 (5 %), width 120 vs 249 (52 %) | height 240 (5 %), width 180 vs 392 (54 %) |
| `room_dims_qwen_pointing --variant with-sizes` | references disagree 123 %; height 922 cm | 380 cm |
| `room_dims_qwen_pointing --variant decoupled` | references agree (1 %); naive height 286 cm (25 %), width 349 vs 249 cm (40 %) | no boiler point |
| `room_dims_qwen_pointing --variant homography` | 323 cm (41 %) | 1390 cm (degenerate points) |
| `room_dims_molmo` | **231 cm (1 %)**, but the rectified references disagree 29 % | 284 cm (24 %) from impossible corners |

The 2B model answers ~240 cm on every wall (including walls 1/2, which have no
references), so the 5 % height is a room-size prior, not a measurement.
Molmo's single 1 % result comes with 29 % internal disagreement and skewed
corners, so treat it as one lucky sample rather than a working method. The
7B text variants weren't re-run in this validation; they generate long
free-text answers to read by hand.

## 7. Final constants

| Constant | Value | Why |
|---|---|---|
| `PX_PER_CM` | 4.0 | rectified canvas resolution |
| `GDINO_PROMPTS` | boiler, hot water tank, pipe, heating pipe, valve, pump, thermometer, electrical box, window, door, radiator | `"heating pipe"` separately matters (§4.3) |
| `BOX_THRESHOLD` | 0.20 | verified clean on this dataset; re-check on a new one |
| `GDINO_MAX_DIM` | 1400 | speed only; not a recall limit |
| `UNIT_W_CM × UNIT_H_CM` | 31 × 48.5 | Thermovation indoor unit (Länge × Höhe); 31 cm depth not modelled |
| `BASE_CLEARANCE_CM` | 8 | 5–10 cm service clearance |
| `EXTRA_CLEARANCE_CM` | pipe/valve/manifold +12, pump/boiler/tank +20, radiator +8 | depth-blindness mitigation |
| Rücklauf HSV | blue (95,35,35)–(135,255,255); red (0,35,35)–(15,255,255) and (155,35,35)–(179,255,255) | S/V floors sampled from this room (§4.2) |
| `RUCKLAUF_PAIR_MAX_DIST_PX` | 120 at a 1920 px scan | ~15 cm dial spacing |
| Marker check | < 5 % error, squareness > 0.85 | squareness excludes off-plane markers |
| Recommended checkpoint | `mm_grounding_dino_large_all` | §5.1 |

## 8. Open items

1. **Wall width from a VLM:** add a horizontal reference (boiler width) to the
   pointing prompts. Vertical references alone only give height (§5.2).
2. **Lens undistortion:** not applied, because there's no calibration for these
   phone photos. `02_calibration/calibrate_camera.py` could provide it for a
   capture that includes the marker board.
3. **Cross-wall "unfold"** (a wall without a Rücklauf scored by distance across
   the shared corner) only exists for the wall3/wall4 pair. Generalising it
   needs full wall topology, which is exactly what the 3D pipeline provides.
4. **Unpaired blue blobs:** on wall2/wall4 the "Rücklauf" is a lone blue blob,
   which on wall4 lies outside the wall face. Consider rejecting unpaired
   blobs, or blobs that map outside the canvas, and falling back to max
   clearance.
5. **Valve manifold:** a blind spot for every detector tried. Test specific
   prompt terms on real examples, or keep the manual override.
6. **Not evaluated:** OWLv2 (faster open-vocabulary detector) and RF-DETR
   (strongest closed-vocabulary detector, but it needs a labelled fine-tuning
   set for these fixture classes).
