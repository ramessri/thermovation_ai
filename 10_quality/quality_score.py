"""
quality_score.py — composite 0-100 reconstruction quality score.

Mirrors the sibling photogram project's QualityScoreCard weighting
philosophy (registration rate, geometric accuracy, coverage, scale
confidence — each scored 0-1, combined by weight, renormalized over
whichever components are actually available) but reads from this repo's
own JSON outputs (sfm_stats.json, scale.json, room_dims.json,
coverage.json) instead of a live DB.

Cheap — pure JSON aggregation, no point-cloud loading. Always safe to run
right after room_dims.py; picks up scale/coverage numbers if those stages
ran too, degrades gracefully (fewer components, renormalized weights) if
not.

Usage:
  python 10_quality/quality_score.py output/sfm/IMG_3126 --model 0
"""

import argparse
import json
from pathlib import Path


def _registration_score(rate: float | None) -> float | None:
    if rate is None:
        return None
    return min(1.0, rate / 0.85)   # 85%+ registered = full score, matches photogram


def _reprojection_score(err_px: float | None) -> float | None:
    if err_px is None:
        return None
    return min(1.0, max(0.0, 1.0 - (err_px - 0.5) / 2.5))   # <0.5px=1.0, 1px=0.8, 2px=0.5, >3px=0


def _scale_confidence_score(scale_info: dict) -> float | None:
    method = scale_info.get("method")
    if method == "marker_triangulation":
        spread = scale_info.get("cross_segment_spread_pct")
    elif method == "aruco_baseline_triangulation":
        cm_per_unit = scale_info.get("cm_per_unit")
        std = scale_info.get("scale_std_m_per_unit")
        spread = (100 * std / (cm_per_unit / 100)) if (cm_per_unit and std is not None) else None
    elif method == "depth_ratio_fallback":
        spread = scale_info.get("depth_ratio_core_cv_pct")
    else:
        spread = None
    if spread is None:
        return 0.6 if scale_info.get("cm_per_unit") else 0.0   # scale exists but unmeasured spread
    # 0% spread = 1.0, 10%+ spread = 0.0 (linear)
    return max(0.0, 1.0 - spread / 10.0)


def _dims_reliability_score(dims: dict) -> float | None:
    if not dims:
        return None
    flags = [dims.get("footprint_reliable"), dims.get("height_reliable")]
    known = [f for f in flags if f is not None]
    if not known:
        return None
    return sum(1.0 for f in known if f) / len(known)


def compute_quality_score(sfm_dir: Path, model: str) -> dict:
    stats_path = sfm_dir / "sfm_stats.json"
    scale_path = sfm_dir / "scale.json"
    dims_path = sfm_dir / "room_dims.json"
    coverage_path = sfm_dir / "coverage.json"

    reg_score = reproj_score = None
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        model_stats = stats.get("models", {}).get(model) or stats.get("models", {}).get(stats.get("best_model"))
        if model_stats:
            reg_score = _registration_score(model_stats.get("registration_rate"))
            reproj_score = _reprojection_score(model_stats.get("mean_reprojection_error_px"))

    scale_score = None
    if scale_path.exists():
        scale_score = _scale_confidence_score(json.loads(scale_path.read_text()))

    dims_score = None
    if dims_path.exists():
        dims_score = _dims_reliability_score(json.loads(dims_path.read_text()))

    coverage_score = None
    if coverage_path.exists():
        coverage_score = json.loads(coverage_path.read_text()).get("coverage_score")

    components = {
        "registration_rate": (reg_score, 0.25),
        "geometric_accuracy": (reproj_score, 0.25),
        "scale_confidence": (scale_score, 0.20),
        "coverage": (coverage_score, 0.20),
        "dimension_reliability": (dims_score, 0.10),
    }
    available = {k: (s, w) for k, (s, w) in components.items() if s is not None}
    if not available:
        return dict(quality_score=None, components={}, reason="no component data available")

    weight_sum = sum(w for _, w in available.values())
    overall = sum(s * w for s, w in available.values()) / weight_sum

    return dict(
        quality_score=round(overall * 100, 1),
        components={k: round(s * 100, 1) for k, (s, _) in available.items()},
        components_missing=[k for k in components if k not in available],
    )


def main():
    parser = argparse.ArgumentParser(description="Composite reconstruction quality score")
    parser.add_argument("sfm_dir", type=Path)
    parser.add_argument("--model", default="0")
    args = parser.parse_args()

    result = compute_quality_score(args.sfm_dir, args.model)
    if result["quality_score"] is None:
        print(f"FAILED: {result['reason']}")
        return

    print(f"Quality score: {result['quality_score']:.1f}/100")
    for k, v in result["components"].items():
        print(f"  {k}: {v:.1f}/100")
    if result.get("components_missing"):
        print(f"  (missing: {', '.join(result['components_missing'])})")

    out_path = args.sfm_dir / "quality_score.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
