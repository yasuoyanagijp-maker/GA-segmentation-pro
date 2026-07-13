"""Per-image fovea/disc landmarks must not bleed across batch snapshots."""

import math
import numpy as np
import cv2
import pytest
from pathlib import Path

from engine.timeseries import (
    build_series_from_saved,
    disc_xy_from_pts,
    landmarks_from_snapshot,
)
from engine.series_registration import default_method_for, compute_forward_matrix, warp_point


def _make_snap(name: str, date: str, fovea, disc_pts=None):
    img = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(img, (64, 64), 30, 200, -1)
    mask = np.zeros_like(img)
    cv2.circle(mask, (int(fovea[0]) + 10, int(fovea[1])), 6, 255, -1)
    disc_xy = disc_xy_from_pts(disc_pts) if disc_pts else None
    return {
        "path": Path(name),
        "filename": name,
        "date": date,
        "image": img,
        "mask": mask,
        "stats": {"area_mm2": 0.1},
        "fovea_xy": fovea,
        "disc_pts": disc_pts,
        "disc_xy": disc_xy,
        "scale_um_px": 10.0,
        "calibration_source": "disc" if disc_pts else "width",
    }


def test_landmarks_from_snapshot_isolates_copies():
    snap = _make_snap(
        "a.png", "2020-01-01",
        (100.0, 110.0),
        [(50.0, 60.0), (70.0, 80.0)],
    )
    lm = landmarks_from_snapshot(snap)
    assert lm["fovea_xy"] == (100.0, 110.0)
    assert lm["disc_xy"] == (60.0, 70.0)
    snap["fovea_xy"] = (999.0, 999.0)
    lm2 = landmarks_from_snapshot(snap)
    assert lm2["fovea_xy"] == (999.0, 999.0)
    assert lm["fovea_xy"] == (100.0, 110.0)


def test_build_series_uses_per_image_fovea_not_shared():
    per = {
        "a.png": _make_snap("a.png", "2020-01-01", (64.0, 64.0)),
        "b.png": _make_snap("b.png", "2021-01-01", (80.0, 72.0)),
        "c.png": _make_snap("c.png", "2022-01-01", (55.0, 90.0)),
    }
    files = [Path("a.png"), Path("b.png"), Path("c.png")]
    dates = {k: v["date"] for k, v in per.items()}
    results = build_series_from_saved(per, files, dates, register=True)
    assert [r["filename"] for r in results] == ["a.png", "b.png", "c.png"]
    # After registration, all foveae align to reference (a.png)
    assert results[0]["fovea_xy"] == (64.0, 64.0)
    for row in results[1:]:
        fx, fy = row["fovea_xy"]
        assert abs(fx - 64.0) < 0.01 and abs(fy - 64.0) < 0.01


def test_default_method_landmark_when_both_have_disc():
    assert default_method_for(
        (1.0, 2.0), (10.0, 20.0), (3.0, 4.0), (30.0, 40.0),
    ) == "landmark"
    assert default_method_for(
        (1.0, 2.0), None, (3.0, 4.0), None,
    ) == "affine"


def test_scale_landmarks_on_resize():
    from engine.timeseries import resize_to_reference_with_scales, scale_landmarks

    small = np.zeros((100, 80), dtype=np.uint8)
    big = np.zeros((200, 160), dtype=np.uint8)
    resized, scales = resize_to_reference_with_scales([big, small])
    assert resized[0].shape == (200, 160)
    assert resized[1].shape == (200, 160)
    assert scales[1] == (2.0, 2.0)
    lm = scale_landmarks((10.0, 20.0), [(5.0, 6.0), (15.0, 16.0)], None, 2.0, 2.0)
    assert lm["fovea_xy"] == (20.0, 40.0)
    assert lm["disc_xy"] == (20.0, 22.0)


def test_disc_not_leaked_from_first_image_to_second_snapshot():
    """Simulate bug: image2 must not inherit image1 disc_pts when saved separately."""
    per = {
        "img1.png": _make_snap(
            "img1.png", "2020-01-01", (64.0, 64.0),
            [(20.0, 30.0), (40.0, 50.0)],
        ),
        "img2.png": _make_snap(
            "img2.png", "2021-01-01", (70.0, 68.0),
            [(25.0, 35.0), (45.0, 55.0)],
        ),
    }
    lm1 = landmarks_from_snapshot(per["img1.png"])
    lm2 = landmarks_from_snapshot(per["img2.png"])
    assert lm1["disc_xy"] != lm2["disc_xy"]
    assert lm1["fovea_xy"] != lm2["fovea_xy"]


def test_compute_forward_matrix_uses_disc_midpoint_for_all_methods_except_fovea():
    """乳頭ライン中点＋中心窩が揃えば affine/rigid も2点相似にフォールバック。"""
    img = np.zeros((128, 128), dtype=np.uint8)
    ref_fovea = (50.0, 50.0)
    ref_disc = (20.0, 50.0)
    mov_fovea = (60.0, 40.0)
    mov_disc = (35.0, 45.0)  # 中心窩平行移動だけでは ref_disc に一致しない
    kwargs = dict(
        moving_fovea=mov_fovea,
        ref_fovea=ref_fovea,
        moving_disc=mov_disc,
        ref_disc=ref_disc,
    )
    m_landmark = compute_forward_matrix(img, img, method="landmark", **kwargs)
    m_affine = compute_forward_matrix(img, img, method="affine", **kwargs)
    m_rigid = compute_forward_matrix(img, img, method="rigid", **kwargs)
    assert np.allclose(m_landmark, m_affine)
    assert np.allclose(m_landmark, m_rigid)
    assert warp_point(mov_fovea, m_landmark) == pytest.approx(ref_fovea, abs=1e-4)
    assert warp_point(mov_disc, m_landmark) == pytest.approx(ref_disc, abs=1e-4)

    m_fovea = compute_forward_matrix(img, img, method="fovea", **kwargs)
    assert warp_point(mov_fovea, m_fovea) == pytest.approx(ref_fovea, abs=1e-4)
    assert warp_point(mov_disc, m_fovea) != pytest.approx(ref_disc, abs=1e-4)


def test_build_series_without_register_keeps_per_image_landmarks():
    """register=False では各画像の中心窩座標を整列しない。"""
    per = {
        "a.png": _make_snap("a.png", "2020-01-01", (64.0, 64.0)),
        "b.png": _make_snap("b.png", "2021-01-01", (80.0, 72.0)),
    }
    files = [Path("a.png"), Path("b.png")]
    dates = {k: v["date"] for k, v in per.items()}
    results = build_series_from_saved(per, files, dates, register=False)
    assert results[0]["fovea_xy"] == (64.0, 64.0)
    assert results[1]["fovea_xy"] == (80.0, 72.0)


def test_build_series_aligns_disc_midpoint_when_line_drawn():
    per = {
        "a.png": _make_snap(
            "a.png", "2020-01-01", (64.0, 64.0),
            [(20.0, 64.0), (40.0, 64.0)],
        ),
        "b.png": _make_snap(
            "b.png", "2021-01-01", (80.0, 72.0),
            [(36.0, 72.0), (56.0, 72.0)],
        ),
    }
    files = [Path("a.png"), Path("b.png")]
    dates = {k: v["date"] for k, v in per.items()}
    results = build_series_from_saved(per, files, dates, register=True)
    ref_disc = results[0]["disc_xy"]
    assert ref_disc == (30.0, 64.0)
    for row in results[1:]:
        dx, dy = row["disc_xy"]
        assert abs(dx - ref_disc[0]) < 0.01 and abs(dy - ref_disc[1]) < 0.01


def _textured_faf_like(size: int = 256) -> np.ndarray:
    """SIFT が拾えるテクスチャ付き FAF 風画像。"""
    rng = np.random.default_rng(0)
    img = rng.integers(40, 180, (size, size), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 3, 220, 3)
    cv2.line(img, (20, size // 2), (size - 20, size // 2), 200, 2)
    cv2.line(img, (size // 2, 20), (size // 2, size - 20), 200, 2)
    for _ in range(12):
        x, y = rng.integers(30, size - 30, size=2)
        cv2.circle(img, (int(x), int(y)), rng.integers(3, 8), int(rng.integers(100, 240)), -1)
    return img


def _invert_2x3(mat: np.ndarray) -> np.ndarray:
    m = np.vstack([mat.astype(np.float64), [0.0, 0.0, 1.0]])
    return np.linalg.inv(m)[:2, :].astype(np.float32)


def test_sift_recovers_similarity_transform():
    """テクスチャ画像で SIFT+RANSAC が平行移動を復元する。"""
    base = _textured_faf_like()
    dx, dy = 14, -9
    warp = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    shifted = cv2.warpAffine(base, warp, (base.shape[1], base.shape[0]))
    expected = _invert_2x3(warp)

    est = compute_forward_matrix(
        shifted, base, method="sift", anchor_fovea=False,
    )
    corner = np.array([[0.0, 0.0]], dtype=np.float32)
    mapped = cv2.transform(corner.reshape(1, 1, 2), est)[0, 0]
    exp = cv2.transform(corner.reshape(1, 1, 2), expected)[0, 0]
    assert mapped[0] == pytest.approx(exp[0], abs=2.5)
    assert mapped[1] == pytest.approx(exp[1], abs=2.5)


def test_landmark_sift_refines_beyond_two_point_init():
    """landmark は SIFT/位相相関候補から真の変換を復元する。"""
    base = _textured_faf_like()
    angle = math.radians(7.0)
    c, s = math.cos(angle), math.sin(angle)
    cx, cy = 128.0, 128.0
    warp = np.array([[c, -s, 0.0], [s, c, 0.0]], dtype=np.float32)
    warp[0, 2] = cx - c * cx + s * cy + 6.0
    warp[1, 2] = cy - s * cx - c * cy - 4.0
    warped = cv2.warpAffine(base, warp, (base.shape[1], base.shape[0]))
    expected = _invert_2x3(warp)

    hybrid = compute_forward_matrix(warped, base, method="sift", anchor_fovea=False)

    test_pt = (180.0, 90.0)
    got = cv2.transform(
        np.array([[test_pt]], dtype=np.float32).reshape(1, 1, 2), hybrid,
    )[0, 0]
    exp = cv2.transform(
        np.array([[test_pt]], dtype=np.float32).reshape(1, 1, 2), expected,
    )[0, 0]
    assert np.hypot(got[0] - exp[0], got[1] - exp[1]) < 4.0


def test_crop_for_registration_left_half_on_wide_capture():
    from engine.series_registration import _crop_for_registration

    wide = np.zeros((400, 1200), dtype=np.uint8)
    cropped = _crop_for_registration(wide)
    assert cropped.shape == (400, 600)

    square = np.zeros((400, 400), dtype=np.uint8)
    assert _crop_for_registration(square).shape == (400, 400)


def test_zoom_sweep_recovers_scale_difference():
    """ズーム差がある場合にスケール付き相似変換を復元する。"""
    from engine.series_registration import _matrix_similarity_scale

    base = _textured_faf_like(320)
    scale = 1.12
    angle = 4.0
    cx, cy = 160.0, 160.0
    warp = cv2.getRotationMatrix2D((cx, cy), angle, scale).astype(np.float32)
    warp[0, 2] += 8.0
    warp[1, 2] -= 5.0
    warped = cv2.warpAffine(base, warp, (base.shape[1], base.shape[0]))
    expected = _invert_2x3(warp)

    est = compute_forward_matrix(warped, base, method="sift", anchor_fovea=False)
    got_scale = _matrix_similarity_scale(est)
    exp_scale = _matrix_similarity_scale(expected)
    assert got_scale == pytest.approx(exp_scale, abs=0.06)

    test_pt = (220.0, 100.0)
    got = cv2.transform(
        np.array([[test_pt]], dtype=np.float32).reshape(1, 1, 2), est,
    )[0, 0]
    exp = cv2.transform(
        np.array([[test_pt]], dtype=np.float32).reshape(1, 1, 2), expected,
    )[0, 0]
    assert np.hypot(got[0] - exp[0], got[1] - exp[1]) < 8.0


def test_zoom_scale_sweep_recovers_magnification():
    """スケールスイープが等方拡大を復元する。"""
    from engine.series_registration import _zoom_scale_sweep_initial, _matrix_similarity_scale

    base = _textured_faf_like(320)
    scale_true = 1.11
    cx, cy = 160.0, 160.0
    init = cv2.getRotationMatrix2D((cx, cy), 0.0, scale_true).astype(np.float32)
    warped = cv2.warpAffine(base, init, (base.shape[1], base.shape[0]))

    est = _zoom_scale_sweep_initial(base, warped)
    assert est is not None
    got_scale = _matrix_similarity_scale(est)
    # warped→base の順方向行列なので scale ≈ 1/scale_true
    assert got_scale == pytest.approx(1.0 / scale_true, rel=0.04, abs=0.04)
