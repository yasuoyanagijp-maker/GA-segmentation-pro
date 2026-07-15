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
from ga_roi_correction.qc import AnomalyThresholds, check_endpoint_containment, dice  # noqa: E402
from ga_roi_correction.refine import (  # noqa: E402
    interpolate_prior_mask,
    position_guided_monotonic_refine,
    trusted_prior_mask,
)


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
    # Growing series: first ⊆ last (gate ok). Middle is oversized vs neighbors.
    masks = [
        _circle(size, center, 18),
        _circle(size, center, 40),   # oversized middle
        _circle(size, center, 26),
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
    assert out["gate_failed"] is False
    assert len(out["results"]) == 3
    assert "mask_pre_roi" in out["results"][0]
    assert out["results"][0]["mask"].shape == (size, size)
    # Endpoints locked as anchors
    assert np.array_equal(out["results"][0]["mask"] > 0, masks[0] > 0)
    assert np.array_equal(out["results"][2]["mask"] > 0, masks[2] > 0)
    assert out["refine_methods"][0] is None
    assert out["refine_methods"][2] is None
    areas_corr = [int(np.count_nonzero(r["mask"])) for r in out["results"]]
    areas_raw = [int(np.count_nonzero(m)) for m in masks]
    assert areas_corr[1] <= areas_raw[1]
    # Middle must stay within dilate(last, 3)
    from skimage.morphology import dilation, disk
    upper = dilation(masks[2] > 0, disk(3))
    assert not ((out["results"][1]["mask"] > 0) & ~upper).any()
    # anomalous middle that was refined must use an allowed method
    if out["refine_methods"][1] is not None:
        assert out["refine_methods"][1] in (
            "band_adaptive_threshold", "chan_vese", "grabcut",
            "position_guided_monotonic", "endpoint_bounds",
            "image_guided_monotonic",
        )
    # 案B: middle ⊇ prev and ⊆ dilate(last)
    mid = out["results"][1]["mask"] > 0
    prev = out["results"][0]["mask"] > 0
    assert not (prev & ~mid).any()
    qc_log = out["qc_log"]
    assert len(qc_log) == 3
    assert qc_log[0]["filename"] == "f0.jpg"
    for key in (
        "raw_area", "corrected_area", "dice_with_prev_raw",
        "dice_with_prev_corrected", "needs_review", "reason",
    ):
        assert key in qc_log[0]


def test_image_guided_monotonic_envelope_union_and_clip():
    from ga_roi_correction.refine import image_guided_monotonic_envelope
    from skimage.morphology import dilation, disk

    size = 160
    center = (80, 80)
    image = np.full((size, size), 160, dtype=np.uint8)
    true_mid = _circle(size, center, 28) > 0
    image[true_mid] = 50
    prev = _circle(size, center, 24) > 0
    # curr shrinks vs prev and also spills outside last
    curr = _circle(size, center, 18) > 0
    last = _circle(size, center, 32) > 0
    # add a blob outside last
    curr = curr | (_circle(size, (130, 80), 8) > 0)

    out = image_guided_monotonic_envelope(
        image, curr, prev_mask=prev, last_mask=last, last_margin_px=3,
    ) > 0
    assert not (prev & ~out).any()
    upper = dilation(last, disk(3))
    assert not (out & ~upper).any()
    # Should recover toward the dark lesion better than raw shrunk curr
    assert dice(out, true_mid) > dice(curr & upper, true_mid)


def test_roi_containment_gate_blocks_when_first_not_in_last():
    size = 120
    center = (60, 60)
    images = [np.full((size, size), 160, dtype=np.uint8) for _ in range(2)]
    masks = [
        _circle(size, center, 40),
        _circle(size, center, 15),
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
        for i in range(2)
    ]
    out = apply_roi_correction_to_series(results)
    assert out["gate_failed"] is False
    assert out["gate"]["ok"] is False
    assert out["endpoint_reconcile"] is not None
    assert out["endpoint_reconcile"]["removed_from_first_px"] > 0
    assert out["endpoint_reconcile"]["added_to_last_px"] > 0
    # After reconcile + lock: first ⊆ dilate(last, 3)
    first = out["results"][0]["mask"] > 0
    last = out["results"][1]["mask"] > 0
    from skimage.morphology import dilation, disk
    assert not (first & ~dilation(last, disk(3))).any()
    # Leak pixels moved onto last
    assert int(np.count_nonzero(last)) >= int(np.count_nonzero(masks[1] > 0))


def test_check_endpoint_containment_margin():
    size = 80
    first = _circle(size, (40, 40), 10)
    last = _circle(size, (40, 40), 12)
    assert check_endpoint_containment(first, last, margin_px=3)["ok"] is True
    # Shifted first leaks outside last even with margin
    first_shift = _circle(size, (55, 40), 10)
    bad = check_endpoint_containment(first_shift, last, margin_px=3)
    assert bad["ok"] is False
    assert bad["leak_pixels"] > 0


def test_reconcile_endpoint_masks_transfers_leak():
    from ga_roi_correction.qc import reconcile_endpoint_masks

    size = 80
    first = _circle(size, (40, 40), 20) > 0
    last = _circle(size, (40, 40), 10) > 0
    out = reconcile_endpoint_masks(first, last, margin_px=3)
    assert out["changed"] is True
    assert out["removed_from_first_px"] == out["added_to_last_px"] > 0
    first_n = out["first_mask"].astype(bool)
    last_n = out["last_mask"].astype(bool)
    from skimage.morphology import dilation, disk
    assert not (first_n & ~dilation(last_n, disk(3))).any()
    # Pixels removed from first appear on last
    leak = first & ~dilation(last, disk(3))
    assert (last_n & leak).sum() == leak.sum()


def test_interpolate_prior_mask_midpoint():
    size = 120
    prev = _circle(size, (60, 60), 20) > 0
    nxt = _circle(size, (60, 60), 30) > 0
    prior = interpolate_prior_mask(prev, nxt, t=0.5) > 0
    # midpoint of two concentric circles ≈ circle with the mean radius
    expected = _circle(size, (60, 60), 25) > 0
    assert dice(prior, expected) > 0.95
    # one-sided fallback returns the available neighbor
    assert np.array_equal(interpolate_prior_mask(prev, None) > 0, prev)
    assert np.array_equal(interpolate_prior_mask(None, nxt) > 0, nxt)
    assert interpolate_prior_mask(None, None) is None


def test_trusted_prior_mask_rejects_disagreeing_neighbors():
    size = 160
    prev = _circle(size, (40, 40), 20) > 0
    far_next = _circle(size, (120, 120), 20) > 0
    assert trusted_prior_mask(prev, far_next) is None
    close_next = _circle(size, (42, 42), 24) > 0
    assert trusted_prior_mask(prev, close_next) is not None


def test_position_guided_monotonic_refine_recovers_shifted_roi():
    size = 200
    center = (100, 100)
    radii = (28, 32, 36)
    image = np.full((size, size), 160, dtype=np.uint8)
    true_mask = _circle(size, center, radii[1]) > 0
    image[true_mask] = 50

    prev_mask = _circle(size, center, radii[0]) > 0
    next_mask = _circle(size, center, radii[2]) > 0
    shifted = _circle(size, (160, 100), radii[1]) > 0  # same area, wrong place

    assert dice(shifted, true_mask) < 0.2
    refined = position_guided_monotonic_refine(
        image, shifted, prev_mask=prev_mask, next_mask=next_mask,
    ) > 0
    assert dice(refined, true_mask) > 0.9
    # monotonicity: prev ⊆ result ⊆ dilate(next, margin)
    assert not (prev_mask & ~refined).any()
    from skimage.morphology import dilation, disk
    upper = dilation(next_mask, disk(3))
    assert not (refined & ~upper).any()


def test_apply_roi_correction_flags_position_anomaly():
    size = 200
    center = (100, 100)
    images = [np.full((size, size), 160, dtype=np.uint8) for _ in range(3)]
    radii = [24, 28, 32]
    for i, img in enumerate(images):
        img[_circle(size, center, radii[i]) > 0] = 50
    masks = [
        _circle(size, center, radii[0]),
        _circle(size, (160, 100), radii[1]),  # area normal but mislocated
        _circle(size, center, radii[2]),
    ]
    results = [
        {
            "image": images[i],
            "mask": masks[i],
            "date": f"2026-0{i + 1}-01",
            "filename": f"f{i}.jpg",
            "scale_um_px": 10.0,
            "fovea_xy": (100.0, 100.0),
            "stats": {"area_mm2": 1.0, "dist_um": 100.0},
            "reg_method": "reference" if i == 0 else "affine",
        }
        for i in range(3)
    ]
    out = apply_roi_correction_to_series(results)
    rec = out["qc_records"][1]
    assert rec.position_anomaly
    assert "position_anomaly" in rec.reason
    assert not rec.area_anomaly  # area QC alone cannot catch this case
    assert out["refine_methods"][1] == "position_guided_monotonic"
    corrected1 = out["results"][1]["mask"] > 0
    true1 = _circle(size, center, radii[1]) > 0
    assert dice(corrected1, true1) > 0.9
    # neighbors stay untouched
    assert np.array_equal(out["results"][0]["mask"] > 0, masks[0] > 0)
    assert np.array_equal(out["results"][2]["mask"] > 0, masks[2] > 0)
