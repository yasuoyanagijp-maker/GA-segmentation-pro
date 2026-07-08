import numpy as np

from engine.foveal_correction import (
    apply_foveal_block_correction,
    build_ga_protection_mask,
)


def _make_foveal_block_image(size=200, scale=10.0, fovea=(100, 100)):
    """中心窩のみ滑らかに暗い FAF 模擬。"""
    img = np.full((size, size), 180, dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    dist_px = np.sqrt((xx - fovea[0]) ** 2 + (yy - fovea[1]) ** 2)
    dist_mm = dist_px * scale / 1000.0
    dip = np.exp(-(dist_mm / 0.35) ** 2)
    img = np.clip(180 - 80 * dip, 0, 255).astype(np.uint8)
    return img


def test_foveal_correction_lifts_center():
    fovea = (100, 100)
    img = _make_foveal_block_image(fovea=fovea)
    center_before = int(img[100, 100])
    out = apply_foveal_block_correction(img, fovea, scale_um_px=10.0, strength=1.0)
    assert out[100, 100] > center_before
    periphery = int(out[30, 100])
    assert abs(periphery - int(img[30, 100])) < 15


def test_foveal_correction_zero_strength_unchanged():
    fovea = (100, 100)
    img = _make_foveal_block_image(fovea=fovea)
    out = apply_foveal_block_correction(img, fovea, scale_um_px=10.0, strength=0.0)
    assert np.array_equal(out, img)


def test_gray_for_segmentation_uses_original_base():
    """補正済み表示画像があっても、セグメント入力は原画像から再計算する。"""
    from engine.foveal_correction import gray_for_segmentation

    fovea = (100, 100)
    original = _make_foveal_block_image(fovea=fovea)
    corrected = apply_foveal_block_correction(original, fovea, scale_um_px=10.0, strength=0.5)
    seg_in = gray_for_segmentation(
        corrected,
        original,
        fovea,
        10.0,
        correction_applied=True,
        correction_skipped=False,
        correction_strength=0.5,
    )
    assert np.array_equal(seg_in, corrected)
    assert not np.array_equal(seg_in, original)
    assert seg_in[100, 100] > original[100, 100]


def test_gray_for_segmentation_skipped_uses_original():
    from engine.foveal_correction import gray_for_segmentation

    fovea = (100, 100)
    original = _make_foveal_block_image(fovea=fovea)
    corrected = apply_foveal_block_correction(original, fovea, scale_um_px=10.0, strength=1.0)
    seg_in = gray_for_segmentation(
        corrected,
        original,
        fovea,
        10.0,
        correction_applied=False,
        correction_skipped=True,
        correction_strength=0.5,
    )
    assert np.array_equal(seg_in, original)


def test_correction_effect_stats_mask_area_counts_as_effective():
    from engine.foveal_correction import correction_effect_stats
    from engine.segmentation import segment_ga

    fovea = (100, 100)
    original = _make_foveal_block_image(fovea=fovea)
    # Add a soft dark halo that correction removes from over-segmentation
    yy, xx = np.ogrid[:200, :200]
    dist = np.sqrt((xx - 100) ** 2 + (yy - 100) ** 2)
    halo = (dist < 45) & (dist > 25)
    original = original.copy()
    original[halo] = np.minimum(original[halo], 90)
    corrected = apply_foveal_block_correction(original, fovea, scale_um_px=10.0, strength=1.0)
    m_raw = segment_ga(original)
    m_corr = segment_ga(corrected)
    stats = correction_effect_stats(
        original, corrected, fovea, 10.0, mask_raw=m_raw, mask_corr=m_corr,
    )
    assert "area_reduction_pct" in stats
    if stats["area_reduction_pct"] is not None and stats["area_reduction_pct"] >= 1.0:
        assert stats["effective"] is True
        assert stats["mask_effective"] is True


def test_correction_shrinks_more_with_strength_on_block():
    """同心円ブロックでは補正強度を上げるほどマスク面積が単調に縮小する。"""
    from engine.segmentation import segment_ga

    fovea = (100, 100)
    img = _make_foveal_block_image(fovea=fovea)
    # 中心窩ブロックが過分割マスクを生む状況
    yy, xx = np.ogrid[:200, :200]
    dist = np.sqrt((xx - 100) ** 2 + (yy - 100) ** 2)
    img = img.copy()
    img[dist < 55] = np.minimum(img[dist < 55], 70)
    areas = []
    for s in (0.0, 0.5, 1.0):
        corr = apply_foveal_block_correction(img, fovea, 10.0, s)
        areas.append(int((segment_ga(corr) > 0).sum()))
    assert areas[1] <= areas[0]
    assert areas[2] <= areas[1]


def test_correction_preserves_localized_offcenter_ga():
    """中心窩付近だが非対称に位置する局所 GA は同心円補正で消えない。"""
    from engine.segmentation import segment_ga

    fovea = (100, 100)
    img = np.full((200, 200), 150, dtype=np.uint8)
    # 中心窩から外れた鋭い暗斑（真の GA を模擬）
    yy, xx = np.ogrid[:200, :200]
    lesion = ((xx - 78) ** 2 + (yy - 100) ** 2) < 22 ** 2
    img[lesion] = 20
    raw = int((segment_ga(img) > 0).sum())
    corr = apply_foveal_block_correction(img, fovea, 10.0, 1.0)
    kept = int((segment_ga(corr) > 0).sum())
    # GA の大部分（>70%）が保持される
    assert kept >= 0.7 * raw


def test_segment_ga_clahe_off_is_larger_on_uniformish():
    from engine.segmentation import segment_ga

    img = _make_foveal_block_image()
    with_c = segment_ga(img, use_clahe=True, clahe_clip=2.0)
    without = segment_ga(img, use_clahe=False)
    # On soft foveal-block images, CLAHE typically shrinks false GA
    assert np.sum(without > 0) >= np.sum(with_c > 0)


def test_segment_ga_clahe_clip_option_runs():
    from engine.segmentation import segment_ga

    img = _make_foveal_block_image()
    m2 = segment_ga(img, use_clahe=True, clahe_clip=2.0)
    m6 = segment_ga(img, use_clahe=True, clahe_clip=6.0)
    assert m2.shape == img.shape and m6.shape == img.shape
    assert set(np.unique(m2)).issubset({0, 255})
    assert set(np.unique(m6)).issubset({0, 255})


def test_segment_ga_smaller_with_foveal_correction_on_block():
    """中心窩ブロックのみの画像では補正後セグメントの方が面積が小さい。"""
    from engine.foveal_correction import gray_for_segmentation
    from engine.segmentation import segment_ga

    fovea = (100, 100)
    original = _make_foveal_block_image(fovea=fovea)
    original[70:130, 70:130] = 35
    seg_raw = segment_ga(original)
    seg_corr = segment_ga(
        gray_for_segmentation(
            original,
            original,
            fovea,
            10.0,
            correction_applied=True,
            correction_skipped=False,
            correction_strength=1.0,
        )
    )
    assert seg_corr.sum() <= seg_raw.sum()
    """急峻な GA 状低蛍光は過補正しない。"""
    fovea = (100, 100)
    img = _make_foveal_block_image(fovea=fovea)
    img[95:105, 95:105] = 30
    out = apply_foveal_block_correction(img, fovea, scale_um_px=10.0, strength=1.0)
    ga_region = out[98:102, 98:102]
    assert float(np.mean(ga_region)) < 120
