"""tests for crop and per-lesion stats (phase 1)."""

import numpy as np
import pytest

from engine.crop import apply_crop, normalize_box
from engine.stats import ga_lesion_stats, ga_stats


def test_normalize_box_orders_points():
    box = normalize_box(80, 90, 10, 20, width=100, height=100)
    assert box == (10, 20, 80, 90)


def test_apply_crop_extracts_region():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    cropped = apply_crop(img, (2, 3, 6, 8))
    assert cropped.shape == (5, 4)
    assert cropped[0, 0] == img[3, 2]


def test_normalize_box_too_small_raises():
    with pytest.raises(ValueError):
        normalize_box(5, 5, 5, 5, width=20, height=20)


def test_ga_lesion_stats_multiple_components():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    mask[20:24, 20:24] = 255

    lesions = ga_lesion_stats(mask, scale_um_px=2.0, fovea_xy=(15, 15))

    assert len(lesions) == 2
    assert lesions[0]['area_px'] == 16
    assert lesions[1]['area_px'] == 16
    assert all(l['dist_um'] is not None for l in lesions)


def test_ga_stats_includes_lesions_and_closest_distance():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0, 0] = 255
    mask[18, 18] = 255

    result = ga_stats(mask, scale_um_px=1.0, fovea_xy=(0, 0))

    assert result['lesion_count'] == 2
    assert len(result['lesions']) == 2
    assert result['dist_um'] == pytest.approx(0.0)
    assert result['nearest_xy'] == (0, 0)
