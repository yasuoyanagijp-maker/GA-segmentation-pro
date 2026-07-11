import cv2
import numpy as np

from engine.segmentation import (
    cleanup_mask_after_edit,
    remove_small_regions,
    segment_ga,
    MIN_SPECK_PX,
)


def _synthetic_faf_image(size=200, blob_radius=30, bg=200, blob=30) -> np.ndarray:
    """明るい背景に中心暗色円（GAを模した病変）を持つ合成FAF画像を作る。"""
    img = np.full((size, size), bg, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), blob_radius, blob, thickness=-1)
    return img


def test_segment_ga_detects_central_dark_lesion():
    img = _synthetic_faf_image()
    fovea = (100.0, 100.0)
    mask = segment_ga(img, fovea_xy=fovea, scale_um_px=10.0)

    assert mask.shape == img.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask[100, 100] == 255
    assert mask.sum() > 0


def test_segment_ga_no_lesion_on_uniform_image():
    img = np.full((100, 100), 128, dtype=np.uint8)
    mask = segment_ga(img)
    assert mask.shape == img.shape


def test_macula_roi_otsu_ignores_peripheral_dark_blob():
    """黄斑 ROI 内 Otsu: 隅の暗部が閾値推定を支配しない。"""
    img = np.full((200, 200), 210, dtype=np.uint8)
    cv2.circle(img, (100, 100), 25, 40, -1)   # 中心 GA
    cv2.rectangle(img, (0, 0), (40, 40), 5, -1)  # 左上の強い暗部
    fovea = (100.0, 100.0)
    with_fovea = segment_ga(img, fovea_xy=fovea, scale_um_px=10.0)
    without = segment_ga(img)
    assert with_fovea[100, 100] == 255
    assert np.sum(with_fovea[0:45, 0:45] > 0) <= np.sum(without[0:45, 0:45] > 0)


def test_proximity_filter_prefers_fovea_over_image_center():
    """主病変は画像中心ではなく中心窩に最も近い成分。"""
    img = np.full((200, 200), 220, dtype=np.uint8)
    cv2.circle(img, (140, 100), 18, 35, -1)  # 中心窩寄り GA
    cv2.circle(img, (60, 100), 22, 30, -1)   # 画像中心寄り GA
    fovea = (140.0, 100.0)
    mask = segment_ga(img, fovea_xy=fovea, scale_um_px=10.0)
    assert mask[100, 140] == 255
    assert mask[100, 60] == 0


def test_remove_small_regions():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:12, 10:12] = 255          # 4 px speck
    mask[20:35, 20:35] = 255          # large blob
    out = remove_small_regions(mask, min_size=20)
    assert out[10, 10] == 0
    assert out[25, 25] == 255


def test_cleanup_mask_after_edit_uses_speck_threshold():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:7, 5:7] = 255
    mask[15:30, 15:30] = 255
    out = cleanup_mask_after_edit(mask)
    assert out[5, 5] == 0
    assert out[20, 20] == 255
    assert MIN_SPECK_PX >= 1
