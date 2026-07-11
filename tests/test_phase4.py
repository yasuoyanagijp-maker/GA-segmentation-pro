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
    guess_image_datetime,
    datetime_to_plot_x,
    datetime_to_iso,
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
    assert guess_image_datetime(path) == __import__('datetime').datetime(2024, 3, 15)


def test_datetime_to_plot_x_is_continuous():
    from datetime import datetime

    d0 = datetime(2020, 1, 1)
    d1 = datetime(2020, 2, 1)
    x0 = datetime_to_plot_x(d0)
    x1 = datetime_to_plot_x(d1)
    assert abs((x1 - x0) - 31.0) < 0.01


def test_parse_user_datetime_input_accepts_date_and_time():
    from datetime import datetime

    from engine.timeseries import parse_user_datetime_input

    assert parse_user_datetime_input("2024-03-15") == datetime(2024, 3, 15)
    assert parse_user_datetime_input("2024-03-15 14:30") == datetime(2024, 3, 15, 14, 30)


def test_refresh_series_dates_skips_manual():
    from engine.timeseries import refresh_series_dates_for_files

    files = [Path("/tmp/patient_20240315_faf.png")]
    out = refresh_series_dates_for_files(
        files, {"patient_20240315_faf.png"}, skip_manual={"patient_20240315_faf.png"},
    )
    assert out == {}
    out2 = refresh_series_dates_for_files(files, {"patient_20240315_faf.png"})
    assert "patient_20240315_faf.png" in out2


def test_sync_date_to_snapshots():
    from engine.timeseries import sync_date_to_snapshots

    per = {"a.png": {"date": "old", "filename": "a.png"}}
    sync_date_to_snapshots(per, "a.png", "2025-06-01T10:00:00")
    assert per["a.png"]["date"] == "2025-06-01T10:00:00"


def test_build_series_uses_refreshed_series_dates():
    per = {
        'a.png': {
            'path': Path('a.png'), 'filename': 'a.png', 'date': '1999-01-01',
            'image': np.zeros((64, 64), dtype=np.uint8),
            'mask': np.zeros((64, 64), dtype=np.uint8),
            'stats': {'area_mm2': 1.0}, 'fovea_xy': (32.0, 32.0),
        },
        'b.png': {
            'path': Path('b.png'), 'filename': 'b.png', 'date': '1999-06-01',
            'image': np.zeros((64, 64), dtype=np.uint8),
            'mask': np.zeros((64, 64), dtype=np.uint8),
            'stats': {'area_mm2': 2.0}, 'fovea_xy': (32.0, 32.0),
        },
    }
    files = [Path('a.png'), Path('b.png')]
    series_dates = {'a.png': '2020-01-01', 'b.png': '2021-01-01'}
    results = build_series_from_saved(
        per, files, series_dates, register=False,
    )
    assert results[0]['date'] == '2020-01-01'
    assert results[1]['date'] == '2021-01-01'


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
