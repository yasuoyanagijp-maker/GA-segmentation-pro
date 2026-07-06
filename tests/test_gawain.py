import numpy as np
import pytest

from engine.gawain import compute_gawain, load_annulus_weights, macula_radius_px
from engine.stats import ga_stats


def _make_scene(size=400, scale_um_px=10.0, fovea=(200, 200)):
    mask = np.zeros((size, size), dtype=np.uint8)
    return mask, fovea, scale_um_px


def test_load_annulus_weights_count():
    weights = load_annulus_weights()
    assert len(weights) == 60
    assert weights[0]['inner_mm'] == 0.0
    assert weights[0]['outer_mm'] == pytest.approx(0.05)
    assert weights[-1]['outer_mm'] == pytest.approx(3.0)


def test_gawain_healthy_macula_near_100():
    mask, fovea, scale = _make_scene()
    result = compute_gawain(mask, fovea, scale)
    assert result['gawain'] == pytest.approx(100.0, abs=0.05)


def test_gawain_full_ga_in_center_drops_score():
    mask, fovea, scale = _make_scene()
    radius_px = int(macula_radius_px(scale))
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    dist = np.sqrt((xx - fovea[0]) ** 2 + (yy - fovea[1]) ** 2)
    mask[dist <= radius_px] = 255

    result = compute_gawain(mask, fovea, scale)
    assert result['gawain'] < 20.0


def test_gawain_partial_ga_between_healthy_and_total_loss():
    mask, fovea, scale = _make_scene()
    mask[195:205, 195:205] = 255

    healthy = compute_gawain(mask, fovea, scale)['gawain']
    assert 0.0 < healthy < 100.0


def test_ga_stats_includes_gawain():
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = ga_stats(mask, scale_um_px=5.0, fovea_xy=(50, 50))
    assert result['gawain'] == pytest.approx(100.0, abs=0.05)
