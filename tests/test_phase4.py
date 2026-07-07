import numpy as np
import pytest
from datetime import date
from pathlib import Path

from engine.registration import register_stack
from engine.timeseries import (
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
