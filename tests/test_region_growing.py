import numpy as np

from engine.region_growing import (
    erase_bright_from_mask,
    erase_oversegmentation_from_mask,
    expand_segmentation_from_mask,
    fast_region_growing,
    fast_region_growing_bright,
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
