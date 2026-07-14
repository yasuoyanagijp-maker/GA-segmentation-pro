"""Smoke tests for pre-aligned ROI correction (no re-registration)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from engine.series_roi_correction import (
    DEFAULT_REFINE_METHOD,
    apply_roi_correction_to_series,
)

_PKG_ROOT = Path(__file__).resolve().parents[1] / "ga_roi_correction"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ga_roi_correction.aligned_correct import choose_refine_method  # noqa: E402
from ga_roi_correction.qc import AnomalyThresholds  # noqa: E402


def _circle(size: int, center: tuple[int, int], radius: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    return (dist <= radius).astype(np.uint8) * 255


def test_default_refine_method_is_band_adaptive():
    assert DEFAULT_REFINE_METHOD == "band_adaptive_threshold"


def test_choose_refine_method_escalation():
    th = AnomalyThresholds()
    # first-frame mild oversize (just above first_ratio 1.5): band
    assert choose_refine_method(None, 160, 100, th) == "band_adaptive_threshold"
    # severe (~2.0×): chan_vese
    assert choose_refine_method(None, 200, 100, th) == "chan_vese"
    # extreme (~2.25×+): grabcut
    assert choose_refine_method(None, 230, 100, th) == "grabcut"
    # shrink mild (6% drop > rel_drop 5%): band
    assert choose_refine_method(100, 94, 100, th) == "band_adaptive_threshold"
    # shrink severe (25%+): chan_vese
    assert choose_refine_method(100, 70, 100, th) == "chan_vese"
    # shrink extreme (50%+): grabcut
    assert choose_refine_method(100, 40, 100, th) == "grabcut"


def test_apply_roi_correction_to_series_shrink_and_flag():
    size = 120
    center = (60, 60)
    images = [np.full((size, size), 160, dtype=np.uint8) for _ in range(3)]
    for i, img in enumerate(images):
        m = _circle(size, center, 20 + 2 * i) > 0
        img[m] = 50
    masks = [
        _circle(size, center, 40),   # oversized first
        _circle(size, center, 22),
        _circle(size, center, 10),   # shrink anomaly
    ]
    results = [
        {
            "image": images[i],
            "mask": masks[i],
            "date": f"2026-0{i + 1}-01",
            "filename": f"f{i}.jpg",
            "scale_um_px": 10.0,
            "fovea_xy": (60.0, 60.0),
            "stats": {"area_mm2": 1.0, "dist_um": 100.0},
            "reg_method": "reference" if i == 0 else "affine",
        }
        for i in range(3)
    ]
    out = apply_roi_correction_to_series(results)
    assert len(out["results"]) == 3
    assert "mask_pre_roi" in out["results"][0]
    assert out["results"][0]["mask"].shape == (size, size)
    areas_raw = [int(np.count_nonzero(m)) for m in masks]
    areas_corr = [int(np.count_nonzero(r["mask"])) for r in out["results"]]
    assert areas_corr[0] <= areas_raw[0]
    assert any(r.get("roi_needs_review") for r in out["results"]) or areas_corr != areas_raw
    # anomalous frames that were refined must use an allowed method
    for method in out["refine_methods"]:
        if method is not None:
            assert method in (
                "band_adaptive_threshold", "chan_vese", "grabcut",
            )
    # frame0 is largely oversized vs frame1 → should escalate past band
    assert out["refine_methods"][0] in ("chan_vese", "grabcut")
