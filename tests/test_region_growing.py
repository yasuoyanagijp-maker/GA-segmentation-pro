import numpy as np

from engine.region_growing import (
    erase_bright_from_mask,
    erase_oversegmentation_from_mask,
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


def test_erase_requires_click_on_mask():
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
