# Thermovation AI

Master's thesis (Thermovation / TH Ingolstadt): extracting room geometry and pipe data
from a smartphone video of a boiler room, and recommending where to mount the
Thermovation indoor unit.

## Setup

Tested on Windows 11, Python 3.12, RTX 3080 (10 GB), CUDA 13.2. Under ~6 GB VRAM the
placement stage will likely run out of memory.

```powershell
python -m venv venv
venv\Scripts\activate
# install the torch build matching your CUDA version first (pytorch.org)
pip install -r requirements.txt
copy .env.example .env   # set DATASET_DIR and OUTPUT_DIR
```

HuggingFace models (GroundingDINO, SAM2.1, DepthPro, SegFormer) download on first run.
MetricAnything is optional and has to be cloned into `08_depth/vendor/metric-anything`.

## Usage

Full pipeline on one or more videos:

```powershell
python run.py "dataset\*.mov" --calibrate --gpu --align --placement
```

Stages are cached, so use `--force` to rerun them. Optional stages: `--pipes`, `--dense`,
`--fix-jumps`, `--export`, `--coverage`. Use `--marker-type aruco` for ArUco markers and
`--parallel N` for several videos at once.

Placement only, on an existing reconstruction:

```powershell
python 06_placement\placement_3d.py output\pipeline\<video>\sfm output\pipeline\<video>\frames --model 0
```

Single photo, without SfM (less reliable):

```powershell
python 06_placement\single_image_placement.py photo.jpg --depth-model depthpro --out results\
```

## Pipeline

| Stage | What it does |
|---|---|
| `01_frames/` | Frame extraction |
| `02_calibration/` | Marker detection (custom 3×3 board or ArUco) and camera calibration from marker views |
| `03_reconstruction/` | SfM (pycolmap), DISK+LightGlue variant, dense MVS, trajectory jump fix |
| `04_scale/` | Metric scale from the marker (triangulation, with a depth-ratio fallback) |
| `05_geometry/` | Room length, width and height; gravity-up room frame and top-down plan |
| `06_placement/` | Unit placement: wall plane, obstacles, Rücklauf detection, scoring |
| `07_pipes/` | Pipe paths and lengths, Vorlauf/Rücklauf pairing |
| `08_depth/` | Monocular depth models (DA2-Metric, ZoeDepth, DepthPro, MetricAnything) |
| `09_export/` | Point cloud and mesh export |
| `10_quality/` | Scan coverage scoring and re-shoot suggestions |
| `2d_wall_rectify/` | No-scan placement from one photo per wall, plus VLM tests |
| `3d_recons_alternatives/` | Comparison against NeRF, 3DGS, MASt3R and VGGT |
| `experiments/` | Segmentation benchmarks (GDINO / YOLO-World / YOLOE + SAM2) |

## Done

- Marker detection, camera calibration and metric scale recovery
- SfM reconstruction and room dimensions, compared against tape measurements on 3 rooms
- Automatic 3D placement, including Rücklauf detection from the blue marking
- Pipe path and length extraction
- Segmentation benchmark: GDINO+SAM2 is best for pipes, YOLO-World finds no pipes
- Depth-model sweep for ceiling height across 24 videos
- Reconstruction comparison across 8 methods on the 34-video dataset
  (`3d_recons_alternatives/Measurements(SfM).csv`)
- Dense MVS, ArUco scale, mesh export and coverage scoring added to `run.py`
- 2D one-photo-per-wall placement experiment

## Results

Ground truth vs. SfM pipeline, footprint:

| Video | Length | Width |
|---|---|---|
| IMG_3126 | −12.9% | +30.3% |
| WhatsApp 09.07.37 | −10.0% | −8.2% |
| WhatsApp 17.41.45 | +51.8% (low confidence) | −36.1% (low confidence) |

- **Height:** sparse SfM can't see the ceiling (flat ceilings have too little texture),
  so height is only reported as a lower bound. Adding monocular depth on top of the SfM
  poses gets much closer. On IMG_3126 (true height 235 cm):
  DA2-Metric ≥231 cm, ZoeDepth ≥224 cm, DepthPro ≥222 cm, SfM alone ≥138 cm.
- **Calibration:** a camera calibration can show low reprojection error and still be
  wrong, so calibrations are checked for distortion, principal point, view diversity and
  diagonal FOV (65–130°).
- **Repeatability:** results change by a few percent from run to run, and by much more
  on weak reconstructions.
- **Reconstruction methods:** MetricAnything and VGGT were more accurate than classical
  SfM.
