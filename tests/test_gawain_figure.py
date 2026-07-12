"""Tests for GAWAIN figure rendering."""

import numpy as np
import pytest

from engine.gawain_figure import (
    crop_display_margins,
    estimate_fovea_from_mask,
    render_gawain_figure,
    render_gawain_figure_bytes,
    split_optos_dual_panel,
)
from engine.gawain import compute_gawain


def test_split_optos_dual_panel():
    img = np.arange(100 * 200, dtype=np.uint8).reshape(100, 200)
    left = split_optos_dual_panel(img, "R")
    right = split_optos_dual_panel(img, "L")
    assert left.shape == (100, 100)
    assert right.shape == (100, 100)
    assert left[0, 0] == img[0, 0]
    assert right[0, 0] == img[0, 100]


def test_estimate_fovea_from_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 70:90] = 255
    fovea = estimate_fovea_from_mask(mask)
    assert fovea is not None
    assert 75 < fovea[0] < 85
    assert 45 < fovea[1] < 55


def test_render_gawain_figure_uses_agg_backend():
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"
    img = np.full((300, 300), 180, dtype=np.uint8)
    mask = np.zeros_like(img)
    mask[135:165, 135:165] = 255
    fovea = (150.0, 150.0)
    scale = 10.0
    png_bytes, result = render_gawain_figure_bytes(img, mask, fovea, scale)
    assert len(png_bytes) > 1000
    assert result["gawain"] == pytest.approx(
        compute_gawain(mask, fovea, scale)["gawain"], abs=0.01,
    )


def test_render_gawain_figure_writes_png(tmp_path):
    img = np.full((300, 300), 180, dtype=np.uint8)
    img[130:170, 130:170] = 40
    mask = np.zeros_like(img)
    mask[135:165, 135:165] = 255
    fovea = (150.0, 150.0)
    scale = 10.0
    out = tmp_path / "fig.png"
    result = render_gawain_figure(img, mask, fovea, scale, out)
    assert out.is_file()
    assert result["gawain"] == pytest.approx(
        compute_gawain(mask, fovea, scale)["gawain"], abs=0.01,
    )
