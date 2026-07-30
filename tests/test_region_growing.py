import numpy as np

from engine.region_growing import (
    DEFAULT_MAX_DELTA_PX,
    DEFAULT_RADIUS_PX,
    GROW_MODE_LSF,
    erase_bright_from_mask,
    erase_oversegmentation_from_mask,
    expand_segmentation_from_mask,
    fast_region_growing,
    fast_region_growing_bright,
    refine_mask_chan_vese_light,
)


def test_fast_region_growing_dark_patch():
    img = np.full((40, 40), 200, dtype=np.uint8)
    img[10:20, 10:20] = 40
    mask = fast_region_growing(img, (15, 15), a=1.0)
    assert mask[15, 15] == 255
    assert mask[0, 0] == 0


def test_fast_region_growing_bright_patch():
    img = np.full((40, 40), 40, dtype=np.uint8)
    img[10:20, 10:20] = 220
    mask = fast_region_growing_bright(img, (15, 15), a=1.0)
    assert mask[15, 15] == 255
    assert mask[35, 35] == 0


def test_erase_overseg_from_dark_ga_click():
    """暗い GA 上をクリック → 明るい過分割縁をマスクから除去。"""
    img = np.full((40, 40), 180, dtype=np.uint8)
    img[12:18, 12:18] = 50
    img[10:20, 10:20] = np.maximum(img[10:20, 10:20], 140)
    ga_mask = np.zeros((40, 40), dtype=np.uint8)
    ga_mask[10:20, 10:20] = 255
    out, removed = erase_oversegmentation_from_mask(img, ga_mask, (15, 15), a=4.0)
    assert removed > 0
    assert np.sum(out) < np.sum(ga_mask)
    assert out[15, 15] == 255


def test_erase_overseg_no_false_change():
    img = np.full((40, 40), 50, dtype=np.uint8)
    ga_mask = np.zeros((40, 40), dtype=np.uint8)
    ga_mask[10:20, 10:20] = 255
    out, removed = erase_oversegmentation_from_mask(img, ga_mask, (15, 15), a=0.5)
    assert removed == 0
    assert np.array_equal(out, ga_mask)


def test_erase_from_outside_mask_near_bright_edge():
    """緑外（暗部）クリック → 強度連結で近傍マスクの明るい過分割を除去。"""
    img = np.full((40, 40), 180, dtype=np.uint8)
    img[12:18, 12:18] = 50
    img[10:20, 10:20] = np.maximum(img[10:20, 10:20], 140)
    img[9, 15] = 50
    ga_mask = np.zeros((40, 40), dtype=np.uint8)
    ga_mask[10:20, 10:20] = 255
    out, removed = erase_oversegmentation_from_mask(img, ga_mask, (9, 15), a=4.0)
    assert removed > 0
    assert np.sum(out) < np.sum(ga_mask)


def test_erase_outside_mask_far_away_no_change():
    img = np.full((40, 40), 200, dtype=np.uint8)
    ga_mask = np.zeros((40, 40), dtype=np.uint8)
    ga_mask[10:20, 10:20] = 255
    out, removed = erase_oversegmentation_from_mask(img, ga_mask, (2, 2), a=2.0)
    assert removed == 0


def test_erase_bright_from_mask_only_touches_mask():
    img = np.full((40, 40), 50, dtype=np.uint8)
    img[10:20, 10:20] = 230
    ga_mask = np.zeros((40, 40), dtype=np.uint8)
    ga_mask[12:18, 12:18] = 255
    out = erase_bright_from_mask(img, ga_mask, (15, 15), a=1.5)
    assert np.sum(out) < np.sum(ga_mask)


def test_expand_from_dark_outside_mask():
    """暗部クリック → 未マスクの暗い GA を追加。"""
    img = np.full((40, 40), 200, dtype=np.uint8)
    img[10:20, 10:20] = 40
    ga_mask = np.zeros((40, 40), dtype=np.uint8)
    ga_mask[12:14, 12:14] = 255
    out, added = expand_segmentation_from_mask(img, ga_mask, (15, 15), a=2.0)
    assert added > 0
    assert np.sum(out) > np.sum(ga_mask)
    assert out[15, 15] == 255


def test_expand_no_change_on_bright_click():
    img = np.full((40, 40), 40, dtype=np.uint8)
    img[10:20, 10:20] = 220
    ga_mask = np.zeros((40, 40), dtype=np.uint8)
    ga_mask[12:18, 12:18] = 255
    out, added = expand_segmentation_from_mask(img, ga_mask, (15, 15), a=2.0)
    assert added == 0
    assert np.array_equal(out, ga_mask)


def test_clear_mask_in_freehand_region():
    from engine.region_growing import clear_mask_in_freehand_region

    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    pts = [(15, 15), (45, 15), (45, 45), (15, 45)]
    out, removed = clear_mask_in_freehand_region(mask, pts)
    assert removed > 0
    assert out[30, 30] == 0
    assert out[5, 5] == 0


def test_clear_mask_outside_freehand_region():
    from engine.region_growing import clear_mask_in_freehand_region

    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:50, 10:50] = 255
    pts = [(20, 20), (40, 20), (40, 40), (20, 40)]
    out, removed = clear_mask_in_freehand_region(mask, pts, outside=True)
    assert removed > 0
    assert out[30, 30] == 255  # inside kept
    assert out[12, 12] == 0    # outside cleared


def test_expand_radius_cap_prevents_far_growth():
    """半径キャップでシード遠方の暗部は追加しない。"""
    img = np.full((80, 80), 40, dtype=np.uint8)
    img[0:80, 0:80] = 40  # 全面暗部（爆発しやすい）
    ga_mask = np.zeros((80, 80), dtype=np.uint8)
    ga_mask[38:42, 38:42] = 255
    out, added = expand_segmentation_from_mask(
        img, ga_mask, (40, 40), a=2.0, radius_px=10, max_delta_px=None,
    )
    assert added > 0
    assert added <= (2 * 10 + 1) ** 2
    # 半径外は未追加
    assert out[5, 5] == 0
    assert DEFAULT_RADIUS_PX >= 10


def test_expand_max_delta_px_cap():
    img = np.full((60, 60), 40, dtype=np.uint8)
    ga_mask = np.zeros((60, 60), dtype=np.uint8)
    ga_mask[28:32, 28:32] = 255
    out, added = expand_segmentation_from_mask(
        img, ga_mask, (30, 30), a=3.0, radius_px=40, max_delta_px=50,
    )
    assert 0 < added <= 50
    assert np.sum(out > 0) <= np.sum(ga_mask > 0) + 50


def test_expand_disconnected_island_grows_local_component():
    """既存マスクと非隣接の暗い島クリック → 局所コンポーネントを新規追加。"""
    img = np.full((60, 60), 200, dtype=np.uint8)
    img[5:15, 5:15] = 40
    img[40:50, 40:50] = 40
    ga_mask = np.zeros((60, 60), dtype=np.uint8)
    ga_mask[7:10, 7:10] = 255
    out, added = expand_segmentation_from_mask(
        img, ga_mask, (45, 45), a=2.0, require_adjacency=True,
    )
    assert added > 0
    assert out[45, 45] == 255
    # 既存中央島は維持、遠方島が追加される
    assert out[8, 8] == 255
    assert np.any(out[40:50, 40:50] > 0)
    # 明るい背景は追加しない
    assert out[30, 30] == 0


def test_expand_standard_nonzero_on_dark_edge_seed():
    """標準モード: 明部に接する暗部境界クリックでも非ゼロ成長する。"""
    img = np.full((80, 80), 180, dtype=np.uint8)
    img[20:60, 20:60] = 45
    ga_mask = np.zeros((80, 80), dtype=np.uint8)
    ga_mask[28:52, 28:52] = 255
    # マスク外・暗部境界（局所窓に明部が混ざるケース）
    out, added = expand_segmentation_from_mask(
        img, ga_mask, (22, 40), a=0.9, mode="standard",
    )
    assert added > 0
    assert out[22, 40] == 255
    assert np.sum(out > 0) > np.sum(ga_mask > 0)


def test_expand_lsf_mode_smoke():
    img = np.full((40, 40), 200, dtype=np.uint8)
    img[10:20, 10:20] = 40
    ga_mask = np.zeros((40, 40), dtype=np.uint8)
    ga_mask[12:14, 12:14] = 255
    out, added = expand_segmentation_from_mask(
        img, ga_mask, (15, 15), a=1.5, mode=GROW_MODE_LSF,
    )
    assert added > 0
    assert out[15, 15] == 255


def test_erase_max_delta_symmetric_cap():
    img = np.full((40, 40), 180, dtype=np.uint8)
    img[12:18, 12:18] = 50
    img[10:20, 10:20] = np.maximum(img[10:20, 10:20], 140)
    ga_mask = np.zeros((40, 40), dtype=np.uint8)
    ga_mask[10:20, 10:20] = 255
    out, removed = erase_oversegmentation_from_mask(
        img, ga_mask, (15, 15), a=4.0, max_delta_px=8,
    )
    assert 0 < removed <= 8
    assert DEFAULT_MAX_DELTA_PX >= 8


def test_refine_mask_chan_vese_light_smoke():
    img = np.full((48, 48), 180, dtype=np.uint8)
    img[16:32, 16:32] = 40
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[18:30, 18:30] = 255
    refined = refine_mask_chan_vese_light(img, mask, num_iter=3, pad=8)
    assert refined.shape == mask.shape
    assert np.any(refined > 0)


def test_refine_cv_fills_interior_holes():
    """確定後CV: 内部穴は埋める。"""
    img = np.full((64, 64), 200, dtype=np.uint8)
    img[16:48, 16:48] = 40
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[18:46, 18:46] = 255
    mask[28:36, 28:36] = 0  # enclosed hole
    refined = refine_mask_chan_vese_light(img, mask, num_iter=4, pad=8, max_outward_px=1)
    assert refined[32, 32] == 255  # hole filled
    assert np.all(refined[mask > 0] == 255)  # pre kept


def test_refine_cv_clips_peripheral_expansion():
    """確定後CV: 外周は pre から max_outward_px 超えて外へ広がらない。"""
    from scipy.ndimage import binary_dilation

    img = np.full((64, 64), 40, dtype=np.uint8)  # uniform dark → CV wants to expand
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:40, 24:40] = 255
    r = 1
    refined = refine_mask_chan_vese_light(
        img, mask, num_iter=8, pad=12, max_outward_px=r,
    )
    pre_b = mask > 0
    allowed = binary_dilation(pre_b, iterations=r)
    # newly added pixels must stay inside dilate(pre, r)
    added = (refined > 0) & ~pre_b
    assert not np.any(added & ~allowed)
    # and all of pre is retained
    assert np.all(refined[pre_b] == 255)
