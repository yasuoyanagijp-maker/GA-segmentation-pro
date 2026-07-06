import numpy as np
import pytest

from engine.stats import ga_stats


def test_ga_stats_area_and_distance():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0, 0] = 255  # 1 pixel GA lesion at row=0, col=0

    result = ga_stats(mask, scale_um_px=2.0, fovea_xy=(3, 4))

    assert result['area_px'] == 1
    # area_mm2はround(値, 3)で丸められるため、1px分のごく小さい面積は0.0になる
    assert result['area_mm2'] == pytest.approx(0.0, abs=1e-9)
    # euclidean px distance from (col=3,row=4) to (col=0,row=0) = 5px -> 10um
    assert result['dist_um'] == pytest.approx(10.0)
    assert result['nearest_xy'] == (0, 0)


def test_ga_stats_no_fovea():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:4, 2:4] = 255

    result = ga_stats(mask, scale_um_px=1.5, fovea_xy=None)

    assert result['area_px'] == 4
    assert result['dist_um'] is None
    assert result['nearest_xy'] is None


def test_ga_stats_empty_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    result = ga_stats(mask, scale_um_px=1.0, fovea_xy=(5, 5))
    assert result['area_px'] == 0
    assert result['dist_um'] is None
