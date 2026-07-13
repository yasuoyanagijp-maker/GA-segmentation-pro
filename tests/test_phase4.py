import numpy as np
import pytest
from datetime import date
from pathlib import Path

from engine.registration import register_stack
from engine.timeseries import (
    build_series_from_saved,
    estimate_progression_rate_um_per_year,
    estimate_growth_mm2_per_year,
    estimate_growth_mm_sqrt_per_year,
    guess_image_date,
    overlays_to_gif_bytes,
    resize_to_reference,
)


def test_register_stack_recovers_translation():
    base = np.zeros((120, 120), dtype=np.uint8)
    cv2 = __import__('cv2')
    cv2.circle(base, (60, 60), 20, 200, -1)

    shifted = np.zeros_like(base)
    shifted[5:115, 8:118] = base[0:110, 0:110]

    stack = np.stack([base, shifted], axis=0)
    aligned, matrices = register_stack(stack)
    assert aligned.shape == stack.shape
    assert len(matrices) == 2
    assert not np.array_equal(aligned[1], shifted)


def test_guess_image_date_from_filename():
    path = Path('/tmp/patient_20240315_faf.png')
    assert guess_image_date(path) == date(2024, 3, 15)


def test_estimate_progression_rate_positive():
    results = [
        {'date': '2020-01-01', 'stats': {'dist_um': 200.0}},
        {'date': '2022-01-01', 'stats': {'dist_um': 100.0}},
    ]
    rate = estimate_progression_rate_um_per_year(results)
    assert rate is not None
    assert rate > 0


def test_estimate_growth_rates():
    results = [
        {'date': '2020-01-01', 'stats': {'area_mm2': 1.0, 'area_mm_sqrt': 1.0}},
        {'date': '2021-01-01', 'stats': {'area_mm2': 1.5, 'area_mm_sqrt': 1.22}},
    ]
    mm2_rate = estimate_growth_mm2_per_year(results)
    sqrt_rate = estimate_growth_mm_sqrt_per_year(results)
    assert mm2_rate == pytest.approx(0.5, abs=0.01)
    assert sqrt_rate == pytest.approx(0.22, abs=0.01)


def test_overlays_to_gif_bytes():
    frames = [
        np.zeros((40, 40, 3), dtype=np.uint8),
        np.full((40, 40, 3), 255, dtype=np.uint8),
    ]
    data = overlays_to_gif_bytes(frames, duration_ms=100)
    assert data[:6] == b'GIF89a'


def test_resize_to_reference():
    a = np.zeros((50, 60), dtype=np.uint8)
    b = np.zeros((80, 90), dtype=np.uint8)
    out = resize_to_reference([a, b])
    assert out[1].shape == a.shape


def test_build_series_from_saved():
    from engine.stats import ga_stats

    img1 = np.zeros((40, 40), dtype=np.uint8)
    img2 = np.ones((40, 40), dtype=np.uint8) * 128
    mask1 = np.zeros((40, 40), dtype=np.uint8)
    mask2 = np.zeros((40, 40), dtype=np.uint8)
    mask2[10:20, 10:20] = 255
    fovea = (20.0, 20.0)
    scale = 5.0
    stats1 = ga_stats(mask1, scale, fovea)
    stats2 = ga_stats(mask2, scale, fovea)
    per_file = {
        "a.jpg": {
            "path": Path("/tmp/a.jpg"),
            "filename": "a.jpg",
            "date": "2020-01-01",
            "image": img1,
            "mask": mask1,
            "stats": stats1,
            "fovea_xy": fovea,
            "scale_um_px": scale,
        },
        "b.jpg": {
            "path": Path("/tmp/b.jpg"),
            "filename": "b.jpg",
            "date": "2021-01-01",
            "image": img2,
            "mask": mask2,
            "stats": stats2,
            "fovea_xy": fovea,
            "scale_um_px": scale,
        },
    }
    files = [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")]
    dates = {"a.jpg": "2020-01-01", "b.jpg": "2021-01-01"}
    results = build_series_from_saved(per_file, files, dates, register=False)
    assert len(results) == 2
    assert results[0]["filename"] == "a.jpg"
    assert results[1]["stats"]["area_mm2"] == stats2["area_mm2"]


def test_build_series_uses_reference_scale_for_all_timepoints():
    """Later images with a different scale_um_px still get ref scale after rebuild."""
    from engine.stats import ga_stats
    from engine.timeseries import series_metric_regressions

    img = np.zeros((40, 40), dtype=np.uint8)
    mask1 = np.zeros((40, 40), dtype=np.uint8)
    mask1[8:18, 8:18] = 255
    mask2 = np.zeros((40, 40), dtype=np.uint8)
    mask2[8:22, 8:22] = 255
    fovea = (20.0, 20.0)
    ref_scale = 5.0
    other_scale = 12.0
    stats1 = ga_stats(mask1, ref_scale, fovea)
    # Intentionally store stats computed with the wrong (own) scale.
    stats2_wrong = ga_stats(mask2, other_scale, fovea)
    expected2 = ga_stats(mask2, ref_scale, fovea)
    per_file = {
        "a.jpg": {
            "path": Path("/tmp/a.jpg"),
            "filename": "a.jpg",
            "date": "2020-01-01",
            "image": img.copy(),
            "mask": mask1,
            "stats": stats1,
            "fovea_xy": fovea,
            "scale_um_px": ref_scale,
        },
        "b.jpg": {
            "path": Path("/tmp/b.jpg"),
            "filename": "b.jpg",
            "date": "2021-01-01",
            "image": img.copy(),
            "mask": mask2,
            "stats": stats2_wrong,
            "fovea_xy": fovea,
            "scale_um_px": other_scale,
        },
    }
    files = [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")]
    dates = {"a.jpg": "2020-01-01", "b.jpg": "2021-01-01"}
    results = build_series_from_saved(per_file, files, dates, register=False)
    assert results[0]["scale_um_px"] == ref_scale
    assert results[1]["scale_um_px"] == ref_scale
    assert results[1]["stats"]["area_mm2"] == expected2["area_mm2"]
    assert results[1]["stats"]["area_mm2"] != stats2_wrong["area_mm2"]


def test_build_series_applies_prior_matrices_without_recompute():
    from engine.stats import ga_stats

    img1 = np.zeros((40, 40), dtype=np.uint8)
    img2 = np.zeros((40, 40), dtype=np.uint8)
    img2[5:15, 5:15] = 200
    mask1 = np.zeros((40, 40), dtype=np.uint8)
    mask2 = np.zeros((40, 40), dtype=np.uint8)
    mask2[5:15, 5:15] = 255
    fovea = (20.0, 20.0)
    scale = 5.0
    per_file = {
        "a.jpg": {
            "path": Path("/tmp/a.jpg"),
            "filename": "a.jpg",
            "date": "2020-01-01",
            "image": img1,
            "mask": mask1,
            "stats": ga_stats(mask1, scale, fovea),
            "fovea_xy": fovea,
            "scale_um_px": scale,
        },
        "b.jpg": {
            "path": Path("/tmp/b.jpg"),
            "filename": "b.jpg",
            "date": "2021-01-01",
            "image": img2,
            "mask": mask2,
            "stats": ga_stats(mask2, scale, fovea),
            "fovea_xy": fovea,
            "scale_um_px": scale,
        },
    }
    files = [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")]
    dates = {"a.jpg": "2020-01-01", "b.jpg": "2021-01-01"}
    # Translate right by 3 px
    prior = {"b.jpg": np.array([[1, 0, 3], [0, 1, 0]], dtype=np.float32)}
    results = build_series_from_saved(
        per_file, files, dates, register=True, prior_matrices=prior,
    )
    assert np.allclose(results[1]["reg_matrix"], prior["b.jpg"])
    assert results[1]["reg_ok"] is True
    # Mask content should have shifted right
    assert int(results[1]["mask"][10, 8]) == 255
    assert int(results[1]["mask"][10, 5]) == 0


def test_series_metric_regressions_detects_decrease():
    from engine.timeseries import series_metric_regressions

    ok = [
        {"stats": {"area_mm2": 1.0, "area_mm_sqrt": 1.0, "gawain": 10}},
        {"stats": {"area_mm2": 1.2, "area_mm_sqrt": 1.1, "gawain": 12}},
    ]
    assert series_metric_regressions(ok) == []
    bad = [
        {"stats": {"area_mm2": 2.0, "area_mm_sqrt": 1.4, "gawain": 20}},
        {"stats": {"area_mm2": 1.5, "area_mm_sqrt": 1.2, "gawain": 18}},
    ]
    alerts = series_metric_regressions(bad)
    assert len(alerts) == 3
    assert any("再解析" in a for a in alerts)