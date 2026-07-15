"""前処理キャッシュ（_PairContext）導入後の等価性・整合性テスト。

キャッシュ有無で各ステージの計算結果が一致すること（速度最適化が
registration の性能を変えないこと）を確認する。
"""
import cv2
import numpy as np
import pytest

from engine.series_registration import (
    _PairContext,
    _alignment_mae,
    _alignment_mae_downscaled,
    _phase_translation_initial,
    _rescale_ecc_warp,
    _sift_ransac_forward,
    _zoom_scale_sweep_initial,
    compute_forward_matrix,
)


def _synthetic_faf(size=512, seed=0):
    """血管様の構造を持つ FAF 風グレー画像。"""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.hypot(xx - size * 0.45, yy - size * 0.5)
    img += 120.0 * np.exp(-(r / (size * 0.55)) ** 2)
    x, y = size * 0.75, size * 0.5
    for _ in range(12):
        ang = rng.uniform(0, 2 * np.pi)
        px, py = x, y
        for _ in range(150):
            ang += rng.normal(0, 0.2)
            nx, ny = px + 3 * np.cos(ang), py + 3 * np.sin(ang)
            if not (0 <= nx < size and 0 <= ny < size):
                break
            cv2.line(img, (int(px), int(py)), (int(nx), int(ny)), -60.0, 2)
            px, py = nx, ny
    img += rng.normal(0, 5, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def pair():
    ref = _synthetic_faf(seed=0)
    m = cv2.getRotationMatrix2D((256, 256), 2.0, 1.05)
    m[0, 2] += 12
    m[1, 2] -= 8
    mov = cv2.warpAffine(ref, m, ref.shape[::-1])
    return ref, mov


def test_alignment_mae_matches_with_and_without_ctx(pair):
    ref, mov = pair
    ctx = _PairContext(ref, mov)
    mat = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], np.float32)
    assert _alignment_mae(ref, mov, mat) == pytest.approx(
        _alignment_mae(ref, mov, mat, ctx=ctx),
    )
    assert _alignment_mae_downscaled(ref, mov, mat) == pytest.approx(
        _alignment_mae_downscaled(ref, mov, mat, ctx=ctx),
    )


def test_phase_translation_matches_with_and_without_ctx(pair):
    ref, mov = pair
    ctx = _PairContext(ref, mov)
    a = _phase_translation_initial(ref, mov)
    b = _phase_translation_initial(ref, mov, ctx=ctx)
    if a is None or b is None:
        assert a is None and b is None
    else:
        assert np.allclose(a, b)


def test_zoom_sweep_matches_with_and_without_ctx(pair):
    ref, mov = pair
    ctx = _PairContext(ref, mov)
    a = _zoom_scale_sweep_initial(ref, mov)
    b = _zoom_scale_sweep_initial(ref, mov, ctx=ctx)
    if a is None or b is None:
        assert a is None and b is None
    else:
        assert np.allclose(a, b)


def test_sift_ref_features_and_matches_cached(pair):
    ref, mov = pair
    ctx = _PairContext(ref, mov)
    kp1, desc1 = ctx.ref_features()
    kp2, desc2 = ctx.ref_features()
    assert kp1 is kp2 and desc1 is desc2
    init = np.eye(2, 3, dtype=np.float32)
    m1 = ctx.matched_points(init)
    m2 = ctx.matched_points(init.copy())
    assert m1 is m2  # 同じ初期値ならキャッシュヒット


def test_sift_matrix_matches_with_and_without_ctx(pair):
    ref, mov = pair
    ctx = _PairContext(ref, mov)
    a = _sift_ransac_forward(ref, mov, model="similarity", initial=None)
    b = _sift_ransac_forward(ref, mov, model="similarity", initial=None, ctx=ctx)
    if a is None or b is None:
        assert a is None and b is None
    else:
        # RANSAC は乱数を含むため対応点数十点規模では実質決定的だが、
        # 念のためサブピクセル精度での一致まで緩めて確認する。
        pts = np.array([[50.0, 50.0], [450.0, 60.0], [250.0, 400.0]])
        for x, y in pts:
            pa = a @ np.array([x, y, 1.0])
            pb = b @ np.array([x, y, 1.0])
            assert np.hypot(*(pa - pb)) < 0.5


def test_rescale_ecc_warp_translation_scaling():
    warp = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -4.0]], np.float32)
    full = _rescale_ecc_warp(warp, 0.5)
    # 線形部は不変、平行移動は 1/factor 倍
    assert np.allclose(full[:, :2], warp[:, :2])
    assert full[0, 2] == pytest.approx(20.0)
    assert full[1, 2] == pytest.approx(-8.0)


def test_rescale_ecc_warp_rotation_consistency():
    # 回転を含む場合も座標変換の合成として正しいこと:
    # p_small = f * p_full, W_full = S^-1 W S
    warp = cv2.getRotationMatrix2D((100.0, 80.0), 5.0, 1.02).astype(np.float32)
    f = 0.25
    full = _rescale_ecc_warp(warp, f)
    p_full = np.array([37.0, 91.0])
    p_small = p_full * f
    expected_small = warp @ np.array([p_small[0], p_small[1], 1.0])
    got_full = full @ np.array([p_full[0], p_full[1], 1.0])
    assert np.allclose(got_full * f, expected_small, atol=1e-4)


def test_compute_forward_matrix_recovers_known_transform(pair):
    ref, mov = pair
    m_true_inv = cv2.getRotationMatrix2D((256, 256), 2.0, 1.05)
    m_true_inv[0, 2] += 12
    m_true_inv[1, 2] -= 8
    m_true = cv2.invertAffineTransform(m_true_inv)
    m_est = compute_forward_matrix(mov, ref, method="affine", anchor_fovea=False)
    pts = [(100.0, 100.0), (400.0, 120.0), (250.0, 420.0)]
    for x, y in pts:
        pe = m_est @ np.array([x, y, 1.0])
        pt = m_true @ np.array([x, y, 1.0])
        assert np.hypot(*(pe - pt)) < 1.5
