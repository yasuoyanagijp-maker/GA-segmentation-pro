import numpy as np
import pytest
from datetime import date
from pathlib import Path

from engine.timeseries import (
    build_series_from_saved,
    estimate_progression_rate_um_per_year,
    estimate_growth_mm2_per_year,
    estimate_growth_mm_sqrt_per_year,
    guess_image_datetime,
    overlays_to_gif_bytes,
    resize_to_reference_with_scales,
)


def test_guess_image_date_from_filename():
    path = Path('/tmp/patient_20240315_faf.png')
    assert guess_image_datetime(path).date() == date(2024, 3, 15)


def test_extract_image_datetime_filename_confirmed():
    from engine.timeseries import extract_image_datetime, is_datetime_source_confirmed

    path = Path('/tmp/patient_20240315_faf.png')
    dt, source = extract_image_datetime(path)
    assert dt.date() == date(2024, 3, 15)
    assert source == "filename"
    assert is_datetime_source_confirmed(source)
    assert not is_datetime_source_confirmed("mtime")
    assert is_datetime_source_confirmed("manual")


def test_landmarks_ready_for_registration():
    from engine.timeseries import landmarks_ready_for_registration

    assert not landmarks_ready_for_registration(None)
    assert not landmarks_ready_for_registration({})
    assert not landmarks_ready_for_registration({"fovea_xy": (10.0, 20.0)})
    assert not landmarks_ready_for_registration(
        {"disc_pts": [(0.0, 0.0), (10.0, 0.0)]}
    )
    assert landmarks_ready_for_registration(
        {
            "fovea_xy": (50.0, 60.0),
            "disc_pts": [(0.0, 0.0), (10.0, 0.0)],
        }
    )
    assert landmarks_ready_for_registration(
        {"fovea_xy": (1.0, 2.0), "disc_xy": (3.0, 4.0)}
    )
    assert landmarks_ready_for_registration(
        {
            "fovea_xy": (10.0, 20.0),
            "scale_um_px": 11.5,
            "calibration_source": "width",
        }
    )
    assert not landmarks_ready_for_registration(
        {
            "fovea_xy": (10.0, 20.0),
            "scale_um_px": 11.5,
            "calibration_source": "disc_1800um",
        }
    )
    assert not landmarks_ready_for_registration(
        {"fovea_xy": (10.0, 20.0), "calibration_source": "width"}
    )


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
    out, scales = resize_to_reference_with_scales([a, b])
    assert out[1].shape == a.shape
    assert scales[0] == (1.0, 1.0)
    assert scales[1] == (60 / 90, 50 / 80)


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
    events: list[tuple[int, int, str]] = []
    results = build_series_from_saved(
        per_file, files, dates, register=True, prior_matrices=prior,
        progress_callback=lambda d, t, lab: events.append((d, t, lab)),
        resegment_after_reg=False,
    )
    assert np.allclose(results[1]["reg_matrix"], prior["b.jpg"])
    assert results[1]["reg_ok"] is True
    # Mask content should have shifted right
    assert int(results[1]["mask"][10, 8]) == 255
    assert int(results[1]["mask"][10, 5]) == 0
    assert results[1]["mask_source"] == "warp"
    assert events == [(1, 1, "適用 b.jpg")]


def test_build_series_progress_callback_counts_moving_frames():
    """progress_callback reports one tick per non-reference timepoint."""
    from engine.stats import ga_stats
    from engine.timeseries import build_series_from_saved

    fovea = (20.0, 20.0)
    scale = 5.0
    per_file = {}
    files = []
    dates = {}
    for i, name in enumerate(["a.jpg", "b.jpg", "c.jpg"]):
        img = np.zeros((40, 40), dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.uint8)
        per_file[name] = {
            "path": Path(f"/tmp/{name}"),
            "filename": name,
            "date": f"2020-0{i + 1}-01",
            "image": img,
            "mask": mask,
            "stats": ga_stats(mask, scale, fovea),
            "fovea_xy": fovea,
            "scale_um_px": scale,
        }
        files.append(Path(f"/tmp/{name}"))
        dates[name] = f"2020-0{i + 1}-01"
    prior = {
        "b.jpg": np.eye(2, 3, dtype=np.float32),
        "c.jpg": np.eye(2, 3, dtype=np.float32),
    }
    events: list[tuple[int, int, str]] = []
    build_series_from_saved(
        per_file, files, dates, register=True, prior_matrices=prior,
        progress_callback=lambda d, t, lab: events.append((d, t, lab)),
    )
    assert [e[:2] for e in events] == [(1, 2), (2, 2)]
    assert all("適用" in e[2] for e in events)


def test_series_metric_regressions_detects_decrease():
    from engine.timeseries import series_metric_regressions

    ok = [
        {"stats": {"area_mm2": 1.0, "area_mm_sqrt": 1.0, "gawain": 10, "dist_um": 200.0}},
        {"stats": {"area_mm2": 1.2, "area_mm_sqrt": 1.1, "gawain": 12, "dist_um": 150.0}},
    ]
    assert series_metric_regressions(ok) == []
    bad = [
        {"stats": {"area_mm2": 2.0, "area_mm_sqrt": 1.4, "gawain": 20, "dist_um": 100.0}},
        {"stats": {"area_mm2": 1.5, "area_mm_sqrt": 1.2, "gawain": 18, "dist_um": 120.0}},
    ]
    alerts = series_metric_regressions(bad)
    assert len(alerts) == 4
    assert any("再解析" in a for a in alerts)
    assert any("中心窩距離" in a for a in alerts)


def test_build_series_recalculates_dist_after_registration():
    """dist_um must come from registered mask + fovea, not saved per-file stats."""
    from engine.stats import ga_stats

    img = np.zeros((40, 40), dtype=np.uint8)
    mask1 = np.zeros((40, 40), dtype=np.uint8)
    mask2 = np.zeros((40, 40), dtype=np.uint8)
    mask2[10:15, 5:10] = 255
    fovea = (35.0, 20.0)
    stale_fovea = (5.0, 20.0)
    scale = 5.0
    per_file = {
        "a.jpg": {
            "path": Path("/tmp/a.jpg"),
            "filename": "a.jpg",
            "date": "2020-01-01",
            "image": img.copy(),
            "mask": mask1,
            "stats": ga_stats(mask1, scale, fovea),
            "fovea_xy": fovea,
            "scale_um_px": scale,
        },
        "b.jpg": {
            "path": Path("/tmp/b.jpg"),
            "filename": "b.jpg",
            "date": "2021-01-01",
            "image": img.copy(),
            "mask": mask2,
            "stats": ga_stats(mask2, scale, stale_fovea),
            "fovea_xy": fovea,
            "scale_um_px": scale,
        },
    }
    files = [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")]
    dates = {"a.jpg": "2020-01-01", "b.jpg": "2021-01-01"}
    prior = {"b.jpg": np.array([[1, 0, 15], [0, 1, 0]], dtype=np.float32)}
    results = build_series_from_saved(
        per_file, files, dates, register=True, prior_matrices=prior,
        resegment_after_reg=False,
    )
    expected = ga_stats(results[1]["mask"], scale, results[1]["fovea_xy"])
    assert results[1]["stats"]["dist_um"] == expected["dist_um"]
    assert results[1]["stats"]["dist_um"] != per_file["b.jpg"]["stats"]["dist_um"]


def test_build_series_resegment_after_reg_marks_mask_source():
    """resegment_after_reg=True で mask_source が設定され、空再segは warp に戻る。"""
    from engine.stats import ga_stats

    img = np.zeros((40, 40), dtype=np.uint8)
    mask1 = np.zeros((40, 40), dtype=np.uint8)
    mask2 = np.zeros((40, 40), dtype=np.uint8)
    mask2[10:15, 5:10] = 255
    fovea = (20.0, 20.0)
    scale = 5.0
    per_file = {
        "a.jpg": {
            "path": Path("/tmp/a.jpg"),
            "filename": "a.jpg",
            "date": "2020-01-01",
            "image": img.copy(),
            "mask": mask1,
            "stats": ga_stats(mask1, scale, fovea),
            "fovea_xy": fovea,
            "scale_um_px": scale,
            "calibration_source": "dfd_4.76mm",
        },
        "b.jpg": {
            "path": Path("/tmp/b.jpg"),
            "filename": "b.jpg",
            "date": "2021-01-01",
            "image": img.copy(),
            "mask": mask2,
            "stats": ga_stats(mask2, scale, fovea),
            "fovea_xy": fovea,
            "scale_um_px": scale,
            "calibration_source": "dfd_4.76mm",
        },
    }
    files = [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")]
    dates = {"a.jpg": "2020-01-01", "b.jpg": "2021-01-01"}
    prior = {"b.jpg": np.eye(2, 3, dtype=np.float32)}
    results = build_series_from_saved(
        per_file, files, dates, register=True, prior_matrices=prior,
        resegment_after_reg=True,
    )
    assert results[0]["mask_source"] == "reference"
    assert results[0]["calibration_source"] == "dfd_4.76mm"
    assert results[1]["calibration_source"] == "dfd_4.76mm"
    # 全黒画像では再segが空 → warp フォールバック
    assert results[1]["mask_source"] == "warp_fallback_empty"
    assert int(results[1]["mask"][12, 7]) == 255


def test_build_series_allows_prep_only_without_mask():
    """時系列本線: landmark のみの prep スナップショットでも registration 可能。"""
    from engine.timeseries import segment_aligned_series

    img = np.zeros((40, 40), dtype=np.uint8)
    fovea = (20.0, 20.0)
    scale = 5.0
    per_file = {
        "a.jpg": {
            "path": Path("/tmp/a.jpg"),
            "filename": "a.jpg",
            "case_id": "patient1",
            "date": "2020-01-01",
            "image": img.copy(),
            "mask": None,
            "stats": None,
            "prep_only": True,
            "fovea_xy": fovea,
            "scale_um_px": scale,
        },
        "b.jpg": {
            "path": Path("/tmp/b.jpg"),
            "filename": "b.jpg",
            "case_id": "patient1",
            "date": "2021-01-01",
            "image": img.copy(),
            "mask": None,
            "stats": None,
            "prep_only": True,
            "fovea_xy": fovea,
            "scale_um_px": scale,
        },
    }
    files = [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")]
    dates = {"a.jpg": "2020-01-01", "b.jpg": "2021-01-01"}
    results = build_series_from_saved(
        per_file, files, dates, register=True, resegment_after_reg=False,
        prior_matrices={"b.jpg": np.eye(2, 3, dtype=np.float32)},
    )
    assert len(results) == 2
    assert results[0]["case_id"] == "patient1"
    # 整列後 seg（全黒 → empty / warp_fallback）
    segs = segment_aligned_series(results, allow_warp_fallback=True)
    assert len(segs) == 2
    assert segs[0]["mask_source"] in ("reference", "empty", "resegment")
    assert segs[1]["stats"] is not None