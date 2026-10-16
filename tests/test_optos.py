"""Optos 専用経路。現行 segment_ga の既定は変えない。"""

import cv2
import numpy as np

from engine.optos import (
    disc_circle_mask,
    drop_border_touching,
    estimate_disc_radius_px,
    flatten_illumination,
    grow_to_edge,
    intact_within_mm2,
    paint_brush,
    propose_fovea_from_disc,
    provisional_scale_from_disc,
    robust_foveal_gap_um,
    scale_disagreement_note,
    scale_from_dfd,
    segment_optos,
    stats_central,
    window_box,
)
from engine.timeseries import describe_foveal_approach
from engine.region_growing import expand_segmentation_from_mask
from engine.segmentation import segment_ga


def _scene():
    """中心窩の左に萎縮、右に暗い乳頭。スケールは 0.9 mm が 20 px。"""
    img = np.full((240, 360), 190, dtype=np.uint8)
    fovea = (140.0, 120.0)
    disc = (230.0, 120.0)
    scale = 45.0  # µm/px → 0.9 mm = 20 px, 6 mm = 133 px
    cv2.circle(img, (230, 120), 18, 20, thickness=-1)
    cv2.circle(img, (80, 90), 12, 30, thickness=-1)
    cv2.circle(img, (100, 170), 10, 28, thickness=-1)
    return img, fovea, disc, scale


def test_segment_ga_default_unchanged_on_synthetic():
    img = np.full((200, 200), 200, dtype=np.uint8)
    cv2.circle(img, (100, 100), 30, 30, thickness=-1)
    mask = segment_ga(img)
    assert mask[100, 100] == 255


def test_propose_fovea_moves_toward_image_center():
    shape = (400, 600)
    disc = (480.0, 200.0)
    fx, fy = propose_fovea_from_disc(disc, shape)
    assert fx < disc[0]
    assert 0 <= fx < 600
    assert 0 <= fy < 400


def _disc_image(radius: int, disc: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    img = np.full(shape, 180, dtype=np.uint8)
    cv2.circle(img, disc, radius, 40, thickness=-1)
    return img


def test_larger_disc_means_higher_zoom_and_finer_scale():
    """同じ画素サイズでも、乳頭が大きい（ズームが高い）画像は µm/px が小さい。"""
    shape = (700, 1400)
    disc = (1100, 350)
    small = _disc_image(40, disc, shape)
    large = _disc_image(80, disc, shape)
    r_small = estimate_disc_radius_px(small, disc)
    r_large = estimate_disc_radius_px(large, disc)
    assert r_small is not None and r_large is not None
    assert r_large > r_small * 1.6
    scale_small = provisional_scale_from_disc(small, disc)
    scale_large = provisional_scale_from_disc(large, disc)
    assert scale_small is not None and scale_large is not None
    assert scale_large < scale_small * 0.7
    fov_small = propose_fovea_from_disc(disc, shape, scale_um_px=scale_small)
    fov_large = propose_fovea_from_disc(disc, shape, scale_um_px=scale_large)
    d_small = abs(fov_small[0] - disc[0])
    d_large = abs(fov_large[0] - disc[0])
    assert d_large > d_small * 1.6
    assert scale_from_dfd(disc, fov_small) == scale_small or abs(
        scale_from_dfd(disc, fov_small) - scale_small
    ) < 1.0


def test_scale_from_dfd_matches_known_distance():
    scale = scale_from_dfd((200.0, 100.0), (100.0, 100.0), dfd_um=4760.0)
    assert scale == 47.6


def test_segment_optos_drops_disc_and_keeps_both_lesions():
    img, fovea, disc, scale = _scene()
    mask = segment_optos(img, fovea, disc, scale)
    assert mask[120, 230] == 0
    assert mask[90, 80] == 255
    assert mask[170, 100] == 255
    assert int(np.sum(disc_circle_mask(mask.shape, disc, scale) & (mask > 0))) == 0


def test_stats_central_ignores_pixels_outside_6mm():
    img, fovea, disc, scale = _scene()
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[90, 80] = 255
    # 中心窩から 200 px = 9 mm。6 mm の外。
    mask[120, 340] = 255
    stats = stats_central(mask, fovea, scale)
    assert stats["area_px"] == 1
    assert stats["outside_quant_px"] == 1


def test_window_box_contains_fovea():
    box = window_box((400, 500), (200.0, 150.0), 20.0)
    x0, y0, x1, y1 = box
    assert x0 <= 200 < x1
    assert y0 <= 150 < y1


def test_flatten_reduces_slow_vignette():
    h, w = 180, 180
    yy, xx = np.mgrid[:h, :w]
    vignette = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w * h)
    img = np.clip(200 - 120 * vignette, 0, 255).astype(np.uint8)
    flat = flatten_illumination(img, 20.0)
    assert flat.shape == img.shape
    assert float(flat.std()) < float(img.std())


def test_paint_brush_adds_and_erases():
    mask = np.zeros((40, 40), dtype=np.uint8)
    added = paint_brush(mask, (20, 20), 10.0, erase=False, radius_mm=0.05)
    assert added[20, 20] == 255
    erased = paint_brush(added, (20, 20), 10.0, erase=True, radius_mm=0.05)
    assert erased[20, 20] == 0


def test_barrier_blocks_growth_across_a_dark_band():
    gray = np.full((80, 80), 40, dtype=np.uint8)
    gray[:, :15] = 200
    gray[:, 65:] = 200
    barrier = np.zeros(gray.shape, dtype=np.uint8)
    barrier[:, 40] = 255
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[40, 20] = 255
    out, added = expand_segmentation_from_mask(
        gray,
        mask,
        (20, 40),
        4.0,
        radius_px=70,
        max_delta_px=4000,
        barrier=barrier,
    )
    assert added > 0
    assert out[40, 30] == 255
    assert out[40, 50] == 0
    assert out[40, 40] == 0


def test_grow_to_edge_returns_mask():
    img, fovea, _disc, scale = _scene()
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[int(fovea[1]), int(fovea[0]) - 40] = 255
    out, added = grow_to_edge(img, mask, (int(fovea[0]) - 40, int(fovea[1])), scale)
    assert out.shape == img.shape
    assert added >= 0


def test_first_fovea_click_segments_and_disc_click_does_not():
    from optos_ui.session import OptosSession

    img, fovea, disc, _scale = _scene()
    session = OptosSession()
    session.gray = img
    session.mask = np.zeros(img.shape, dtype=np.uint8)
    session.place_disc(disc)
    assert session.fovea_xy is not None
    assert session.fovea_user_placed is False
    assert not np.any(session.mask)
    session.place_fovea(fovea)
    assert session.fovea_user_placed is True
    assert session.mask_stale is False
    assert np.any(session.mask)
    kept = session.mask.copy()
    session.place_fovea((fovea[0] + 8.0, fovea[1]))
    assert session.mask_stale is True
    assert np.array_equal(session.mask, kept)


def test_next_instruction_follows_the_visit():
    from pathlib import Path

    from optos_ui.session import OptosSession

    session = OptosSession()
    assert "フォルダ" in session.next_instruction()
    session.folder = "/tmp"
    session.gray = np.zeros((40, 40), dtype=np.uint8)
    session.files = [Path("a.png"), Path("b.png")]
    assert "乳頭" in session.next_instruction()
    session.disc_xy = (30.0, 20.0)
    session.fovea_xy = (10.0, 20.0)
    session.scale_um_px = 20.0
    session.mask = np.zeros((40, 40), dtype=np.uint8)
    assert "中心窩" in session.next_instruction()
    session.fovea_user_placed = True
    session.mask_stale = True
    assert "再セグ" in session.next_instruction()
    session.mask_stale = False
    assert "確定" in session.next_instruction()
    session.confirmed = True
    session.case_id = "eye"
    session._stash()
    assert "次の画像" in session.next_instruction()


def test_osascript_folder_cancel_and_path():
    from optos_ui.folder_dialog import folder_from_osascript

    assert folder_from_osascript(1, "", "execution error: User canceled. (-128)") is None
    assert folder_from_osascript(0, "/tmp/eyes/\n", "") == "/tmp/eyes"
    try:
        folder_from_osascript(1, "", "osascript is not allowed")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "not allowed" in str(exc)


def test_close_tool_keeps_mask_and_unconfirmed_snap():
    from pathlib import Path

    from optos_ui.session import OptosSession

    session = OptosSession()
    session.case_id = "case"
    session.files = [Path("a.png")]
    session.index = 0
    session.gray = np.full((80, 80), 180, dtype=np.uint8)
    session.mask = np.zeros((80, 80), dtype=np.uint8)
    session.mask[40, 40] = 255
    session.disc_xy = (60.0, 40.0)
    session.fovea_xy = (30.0, 40.0)
    session.scale_um_px = 40.0
    session.date_iso = "2020-01-01"
    session.tool = "grow"
    session.close_tool()
    assert session.tool == "pan"
    assert session.mask[40, 40] == 255
    session._stash()
    snap = session.snap_for("a.png")
    assert snap is not None
    assert snap.confirmed is False
    assert snap.mask is not None and snap.mask[40, 40] == 255


def test_robust_gap_ignores_a_one_pixel_spike():
    mask = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(mask, (20, 40), 8, 255, thickness=-1)
    mask[40, 55] = 255
    fovea = (70.0, 40.0)
    scale = 10.0
    gap = robust_foveal_gap_um(mask, fovea, scale)
    assert gap is not None
    assert gap > 200


def test_gap_inside_lesion_is_zero():
    mask = np.zeros((40, 40), dtype=np.uint8)
    cv2.circle(mask, (20, 20), 8, 255, thickness=-1)
    assert robust_foveal_gap_um(mask, (20.0, 20.0), 10.0) == 0.0


def test_intact_within_1mm_counts_only_the_circle():
    mask = np.zeros((80, 80), dtype=np.uint8)
    scale = 100.0  # 1 mm = 10 px
    mask[:, :40] = 255
    intact = intact_within_mm2(mask, (40.0, 40.0), scale, radius_mm=1.0)
    assert 1.2 < intact < 1.8


def test_close_gap_does_not_become_years():
    text = describe_foveal_approach(65.0, 3.6)
    assert "出しません" in text
    assert "18" not in text
    far = describe_foveal_approach(1000.0, 100.0)
    assert "10.00 年" in far


def test_scale_disagreement_note_only_when_far_apart():
    assert scale_disagreement_note(10.0, 12.0) == ""
    note = scale_disagreement_note(15.8, 10.0)
    assert "15.8" in note and "10.0" in note


def test_drop_border_keeps_a_central_blob():
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[0:5, 0:8] = 255
    cv2.circle(mask, (40, 30), 6, 255, thickness=-1)
    kept = drop_border_touching(mask)
    assert kept[0, 0] == 0
    assert kept[30, 40] == 255
