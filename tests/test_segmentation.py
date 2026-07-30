import cv2
import numpy as np

from engine.segmentation import segment_ga, segment_ga_with_optional_cv


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


def test_segment_ga_accepts_macula_radius_and_min_size():
    """高影響パラメータが kwargs で受け付けられ、min_size が大きいと面積が減る。"""
    # 中程度の病変（面積 ~700px）— min_size=2000 で消える
    img = _synthetic_faf_image(size=200, blob_radius=15, bg=200, blob=30)
    fovea = (100.0, 100.0)
    scale = 10.0  # um/px → 6mm ROI は画像全体を覆う
    kept = segment_ga(
        img, fovea_xy=fovea, scale_um_px=scale, macula_radius_mm=6.0, min_size=100,
    )
    dropped = segment_ga(
        img, fovea_xy=fovea, scale_um_px=scale, macula_radius_mm=6.0, min_size=2000,
    )
    assert kept.shape == img.shape and dropped.shape == img.shape
    assert set(np.unique(kept)).issubset({0, 255})
    assert int(np.count_nonzero(kept)) > 0
    assert int(np.count_nonzero(dropped)) < int(np.count_nonzero(kept))
    # macula_radius_mm も実行できること
    narrow = segment_ga(
        img, fovea_xy=fovea, scale_um_px=scale, macula_radius_mm=3.0, min_size=100,
    )
    assert narrow.shape == img.shape


def test_auto_cv_off_matches_segment_ga_alone():
    """auto_cv=False は Otsu のみ（segment_ga）と一致。"""
    img = _synthetic_faf_image()
    otsu = segment_ga(img)
    off = segment_ga_with_optional_cv(img, auto_cv=False)
    assert np.array_equal(off, otsu)


def test_auto_cv_on_fills_interior_holes(monkeypatch):
    """auto_cv=ON: Otsu 後マスクの内部穴を埋める。"""
    img = np.full((64, 64), 200, dtype=np.uint8)
    img[16:48, 16:48] = 40
    base = np.zeros((64, 64), dtype=np.uint8)
    base[18:46, 18:46] = 255
    base[28:36, 28:36] = 0  # enclosed hole

    monkeypatch.setattr(
        "engine.segmentation.segment_ga",
        lambda *a, **k: base.copy(),
    )
    refined = segment_ga_with_optional_cv(img, auto_cv=True)
    assert refined[32, 32] == 255
    assert np.all(refined[base > 0] == 255)


def test_auto_cv_on_clips_peripheral_expansion(monkeypatch):
    """auto_cv=ON: 外周は max_outward_px 超えて外へ広がらない。"""
    from scipy.ndimage import binary_dilation

    img = np.full((64, 64), 40, dtype=np.uint8)  # uniform dark → CV wants expand
    base = np.zeros((64, 64), dtype=np.uint8)
    base[24:40, 24:40] = 255

    monkeypatch.setattr(
        "engine.segmentation.segment_ga",
        lambda *a, **k: base.copy(),
    )
    r = 1
    refined = segment_ga_with_optional_cv(img, auto_cv=True)
    pre_b = base > 0
    allowed = binary_dilation(pre_b, iterations=r)
    added = (refined > 0) & ~pre_b
    assert not np.any(added & ~allowed)
    assert np.all(refined[pre_b] == 255)


def test_auto_cv_skips_empty_mask(monkeypatch):
    """空マスクでは CV を呼ばずそのまま返す。"""
    empty = np.zeros((32, 32), dtype=np.uint8)
    monkeypatch.setattr(
        "engine.segmentation.segment_ga",
        lambda *a, **k: empty.copy(),
    )
    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("CV should be skipped on empty mask")

    monkeypatch.setattr(
        "engine.region_growing.refine_mask_chan_vese_light",
        _boom,
    )
    out = segment_ga_with_optional_cv(np.zeros((32, 32), dtype=np.uint8), auto_cv=True)
    assert np.array_equal(out, empty)
    assert calls["n"] == 0
