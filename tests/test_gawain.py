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


def test_format_gawain_display():
    from engine.gawain import GAWAIN_VALUE_RANGE, format_gawain_display

    assert format_gawain_display(85.3) == f"85.3 {GAWAIN_VALUE_RANGE}"
    assert format_gawain_display(None) == f"— {GAWAIN_VALUE_RANGE}"


def test_gawain_label_includes_full_name_and_range():
    from engine.gawain import GAWAIN_FULL_NAME, GAWAIN_LABEL, GAWAIN_VALUE_RANGE

    assert GAWAIN_FULL_NAME in GAWAIN_LABEL
    assert GAWAIN_VALUE_RANGE in GAWAIN_LABEL


def test_ga_stats_includes_gawain():
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = ga_stats(mask, scale_um_px=5.0, fovea_xy=(50, 50))
    assert result['gawain'] == pytest.approx(100.0, abs=0.05)


def _compute_gawain_reference(mask, fovea_xy, scale_um_px):
    """ベクトル化前の素朴な実装（環ごとに全画面マスク）— 等価性検証用。"""
    h, w = mask.shape
    fx, fy = fovea_xy
    yy, xx = np.ogrid[:h, :w]
    dist_mm = np.sqrt((xx - fx) ** 2 + (yy - fy) ** 2) * scale_um_px / 1000.0
    ga = (mask > 0).astype(np.float32)
    macula = dist_mm <= 3.0
    total = 0.0
    for spec in load_annulus_weights():
        ring = macula & (dist_mm >= spec['inner_mm']) & (dist_mm < spec['outer_mm'])
        n_ring = int(ring.sum())
        ga_fraction = float(ga[ring].mean()) if n_ring else 0.0
        total += 100.0 * (1.0 - ga_fraction) * spec['area_mm2'] * spec['weight'] / 100.0
    return round(total, 2)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_gawain_vectorized_matches_reference(seed):
    rng = np.random.default_rng(seed)
    size = 300
    scale = float(rng.uniform(8.0, 20.0))
    fovea = (
        float(rng.uniform(size * 0.2, size * 0.8)),
        float(rng.uniform(size * 0.2, size * 0.8)),
    )
    mask = (rng.random((size, size)) < 0.3).astype(np.uint8) * 255
    got = compute_gawain(mask, fovea, scale)['gawain']
    want = _compute_gawain_reference(mask, fovea, scale)
    assert got == pytest.approx(want, abs=0.01)


def test_gawain_vectorized_matches_reference_fovea_near_edge():
    """黄斑が画像端で切れるケースでも旧実装と一致する。"""
    size = 200
    scale = 25.0  # 半径3mm = 120px、fovea を端に置くと大きくはみ出す
    mask = np.zeros((size, size), np.uint8)
    mask[:, :60] = 255
    for fovea in [(5.0, 5.0), (0.0, 100.0), (199.0, 199.0), (100.0, -10.0)]:
        got = compute_gawain(mask, fovea, scale)['gawain']
        want = _compute_gawain_reference(mask, fovea, scale)
        assert got == pytest.approx(want, abs=0.01), fovea
