import pytest

from engine.calibration import (
    DEFAULT_DFD_UM,
    calibration_source_label,
    compute_scale_um_per_px,
    compute_scale_um_per_px_from_dfd,
    compute_scale_um_per_px_from_field_width,
    source_id_dfd,
    source_id_disc,
)


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


def test_compute_scale_from_field_width():
    scale = compute_scale_um_per_px_from_field_width(1000, 30.0)
    assert scale == pytest.approx(30.0)

    with pytest.raises(ValueError):
        compute_scale_um_per_px_from_field_width(0, 30.0)


def test_compute_scale_from_dfd():
    # fovea at origin, disc 476 px away → 4760/476 = 10 um/px
    scale = compute_scale_um_per_px_from_dfd((0, 0), (476, 0), dfd_um=DEFAULT_DFD_UM)
    assert scale == pytest.approx(10.0)
    assert source_id_dfd() == "dfd_4.76mm"
    assert source_id_disc() == "disc_1800um"


def test_compute_scale_from_dfd_same_point_raises():
    with pytest.raises(ValueError):
        compute_scale_um_per_px_from_dfd((10, 10), (10, 10))


def test_calibration_source_labels():
    assert "DFD" in calibration_source_label("dfd_4.76mm")
    assert "乳頭" in calibration_source_label("disc_1800um")
    assert "乳頭" in calibration_source_label("disc")
    assert calibration_source_label("width") == "幅指定"
