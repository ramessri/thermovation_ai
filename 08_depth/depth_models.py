"""
depth_models.py — pluggable loaders + predictors for monocular metric depth
models, for 9.2b: does dense per-pixel depth recover a ceiling estimate
where sparse SfM structurally cannot, and can it independently cross-check
the marker-based (Zhang) scale?

Three of the five candidate models load identically via `transformers`'
generic depth-estimation API (already a dependency, no new install):
  - DepthAnythingV2-Metric (indoor variant — matches a boiler-room/cellar)
  - ZoeDepth
  - DepthPro
The other two need their own package:
  - MetricAnything (github.com/metric-anything/metric-anything, HF checkpoint)
  - Depth Anything 3 Metric (`pip install depth-anything-3`)

Every predictor returns a metric depth map in CENTIMETERS, same resolution
as the input image, so callers (depth_scale.py, ceiling_check.py) don't
need to know which underlying model produced it.

Usage (as a library — see depth_scale.py / ceiling_check.py for the CLIs
that actually use this):
  from depth_models import DEPTH_MODELS
  load_fn, predict_fn = DEPTH_MODELS["depthanything_v2_metric"]
  model = load_fn("cuda")
  depth_cm = predict_fn(image_rgb, model, "cuda")
"""

import numpy as np


# ── shared HF AutoModelForDepthEstimation path (DA2-Metric, ZoeDepth, DepthPro) ─

_HF_MODEL_IDS = dict(
    depthanything_v2_metric="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    zoedepth="Intel/zoedepth-nyu",
    depthpro="apple/DepthPro-hf",
)


def _load_hf_depth_model(model_id: str, device: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()
    print(f"  {model_id} loaded")
    return processor, model


def _predict_hf_depth_cm(image_rgb: np.ndarray, loaded, device: str) -> np.ndarray:
    """image_rgb -> per-pixel metric depth in cm, resized back to input resolution.

    All three HF models here output metric depth in METERS."""
    import torch
    from PIL import Image as PILImage

    processor, model = loaded
    pil_img = PILImage.fromarray(image_rgb)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    depth_m = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image_rgb.shape[0], image_rgb.shape[1])]
    )[0]["predicted_depth"]
    return depth_m.cpu().numpy().astype(np.float64) * 100.0   # m -> cm


def load_depthanything_v2_metric(device: str):
    return _load_hf_depth_model(_HF_MODEL_IDS["depthanything_v2_metric"], device)


def load_zoedepth(device: str):
    return _load_hf_depth_model(_HF_MODEL_IDS["zoedepth"], device)


def load_depthpro(device: str):
    return _load_hf_depth_model(_HF_MODEL_IDS["depthpro"], device)


def predict_hf(image_rgb: np.ndarray, loaded, device: str, **_kwargs) -> np.ndarray:
    return _predict_hf_depth_cm(image_rgb, loaded, device)


# ── MetricAnything (own package, own HF checkpoint) ─────────────────────────

def load_metric_anything(device: str):
    """Requires `git clone https://github.com/metric-anything/metric-anything`
    on the Python path first — see 08_depth/README.md. Checkpoint:
    yjh001/metricanything_student_depthmap."""
    from depth_model import MetricAnythingDepthMap
    model = MetricAnythingDepthMap.from_pretrained(
        "yjh001/metricanything_student_depthmap", filename="student_depthmap.pt",
    ).to(device).eval()
    print("  MetricAnything (student_depthmap) loaded")
    return model


def predict_metric_anything(image_rgb: np.ndarray, model, device: str, **_kwargs) -> np.ndarray:
    """Returns metric depth in cm. Verify the exact pre/post-processing
    (input size, output units) against the model repo's own infer.py when
    you first run this — MetricAnything was published 2026-01-29, its
    inference API may have changed since this was written."""
    import torch
    from torchvision.transforms import v2

    h, w = image_rgb.shape[:2]
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    x = transform(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_m = model(x)
    depth_m = torch.nn.functional.interpolate(
        depth_m if depth_m.dim() == 4 else depth_m.unsqueeze(1), size=(h, w),
        mode="bilinear", align_corners=False,
    ).squeeze().cpu().numpy().astype(np.float64)
    return depth_m * 100.0   # m -> cm


# ── Depth Anything 3 Metric (own package) ───────────────────────────────────

def load_depth_anything_v3_metric(device: str):
    """Requires `pip install depth-anything-3`."""
    from depth_anything_3 import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3Metric-Large").to(device).eval()
    print("  Depth Anything 3 Metric (DA3Metric-Large) loaded")
    return model


def predict_depth_anything_v3(image_rgb: np.ndarray, model, device: str,
                              focal_px: float = None, **_kwargs) -> np.ndarray:
    """Returns metric depth in cm. DA3Metric's raw output isn't already in
    meters — the model card gives the exact conversion:
    metric_depth_m = focal_px * net_output / 300. Needs the camera's focal
    length in pixels (from the SfM model this is being cross-checked
    against), unlike the other four models here which are self-contained —
    callers MUST pass focal_px=... for this one model. Verify this formula
    against the model card when you first run this — DA3 was published
    2026-03-04, very recent."""
    if focal_px is None:
        raise ValueError("predict_depth_anything_v3 requires focal_px (camera focal length in pixels)")
    import torch
    from torchvision.transforms import v2

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    x = transform(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        net_output = model(x)
    depth_m = focal_px * net_output.squeeze().cpu().numpy().astype(np.float64) / 300.0
    return depth_m * 100.0   # m -> cm


# ── registry ─────────────────────────────────────────────────────────────────

# name -> (load_fn(device) -> loaded, predict_fn(image_rgb, loaded, device, **kw) -> depth_cm)
DEPTH_MODELS = dict(
    depthanything_v2_metric=(load_depthanything_v2_metric, predict_hf),
    zoedepth=(load_zoedepth, predict_hf),
    depthpro=(load_depthpro, predict_hf),
    metric_anything=(load_metric_anything, predict_metric_anything),
    depth_anything_v3=(load_depth_anything_v3_metric, predict_depth_anything_v3),
)
