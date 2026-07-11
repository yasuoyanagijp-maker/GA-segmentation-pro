"""Per-image fovea/disc landmarks must not bleed across batch snapshots."""

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
    """乳頭ライン中点＋中心窩が揃えば affine/rigid も2点相似変換を使う。"""
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
