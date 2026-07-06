import cv2
import numpy as np

from engine.segmentation import segment_ga


def _synthetic_faf_image(size=200, blob_radius=30, bg=200, blob=30) -> np.ndarray:
    """明るい背景に中心暗色円（GAを模した病変）を持つ合成FAF画像を作る。"""
    img = np.full((size, size), bg, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), blob_radius, blob, thickness=-1)
    return img


def test_segment_ga_detects_central_dark_lesion():
    img = _synthetic_faf_image()
    mask = segment_ga(img)

    assert mask.shape == img.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    # 中心付近に検出されたGA画素があること
    assert mask[100, 100] == 255
    assert mask.sum() > 0


def test_segment_ga_no_lesion_on_uniform_image():
    img = np.full((100, 100), 128, dtype=np.uint8)
    mask = segment_ga(img)
    assert mask.shape == img.shape
