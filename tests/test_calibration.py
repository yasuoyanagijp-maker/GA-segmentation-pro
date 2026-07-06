import pytest

from engine.calibration import compute_scale_um_per_px


def test_compute_scale_basic():
    scale = compute_scale_um_per_px((0, 0), (100, 0), disc_diameter_um=1800)
    assert scale == pytest.approx(18.0)


def test_compute_scale_diagonal():
    scale = compute_scale_um_per_px((0, 0), (3, 4), disc_diameter_um=1000)
    # distance = 5px -> 200 um/px
    assert scale == pytest.approx(200.0)


def test_compute_scale_same_point_raises():
    with pytest.raises(ValueError):
        compute_scale_um_per_px((10, 10), (10, 10))
