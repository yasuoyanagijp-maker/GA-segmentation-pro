"""Optos 専用経路。現行 segment_ga の既定は変えない。"""

import math
from pathlib import Path

import cv2
import pytest
import numpy as np

from engine.optos import (
    area_mm2_on_24mm_globe,
    disc_circle_mask,
    drop_border_touching,
    estimate_disc_radius_px,
    flatten_illumination,
    grow_to_edge,
    growth_image,
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
    scale = 45.0  # µm/px → 0.9 mm = 20 px、直径 6 mm = 半径 67 px
    cv2.circle(img, (230, 120), 18, 20, thickness=-1)
    cv2.circle(img, (108, 102), 12, 30, thickness=-1)
    cv2.circle(img, (115, 155), 10, 28, thickness=-1)
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


def test_later_visit_keeps_the_first_millimetres_when_zoom_changes():
    from engine.optos import fovea_from_reference, series_scale_um_px, stable_disc_radius

    shape = (700, 1400)
    disc = (900.0, 350.0)
    small = _disc_image(50, (900, 350), shape)
    large = _disc_image(80, (900, 350), shape)
    _center, radius_small = stable_disc_radius(small, disc)
    _center, radius_large = stable_disc_radius(large, disc)
    assert radius_small is not None and radius_large is not None
    assert radius_large > radius_small * 1.4
    fovea = (600.0, 360.0)
    scale_small = scale_from_dfd(disc, fovea)
    scale_large = series_scale_um_px(scale_small, radius_small, radius_large)
    assert scale_large < scale_small * 0.8
    vector = (fovea[0] - disc[0], fovea[1] - disc[1])
    moved = fovea_from_reference(disc, vector, radius_small, radius_large, shape)
    mm_small = math.hypot(*vector) * scale_small
    mm_large = math.hypot(moved[0] - disc[0], moved[1] - disc[1]) * scale_large
    assert mm_large == pytest.approx(mm_small, rel=0.03)
    nudged = (moved[0] + 12.0, moved[1] + 6.0)
    assert scale_large == series_scale_um_px(scale_small, radius_small, radius_large)
    from engine.optos import series_scale_from_dfd
    nudged_scale = series_scale_from_dfd(scale_small, disc, fovea, disc, nudged)
    dfd_anchor = math.hypot(fovea[0] - disc[0], fovea[1] - disc[1])
    dfd_nudged = math.hypot(nudged[0] - disc[0], nudged[1] - disc[1])
    assert nudged_scale == pytest.approx(scale_small * dfd_anchor / dfd_nudged)
    assert dfd_nudged * nudged_scale == pytest.approx(dfd_anchor * scale_small)


def test_fovea_nudge_keeps_the_disc_scale_and_area():
    from pathlib import Path

    from optos_ui.session import OptosSession

    session = OptosSession()
    session.case_id = "OD"
    session.files = [Path("first.png"), Path("later.png")]
    session.index = 0
    session.gray = _disc_image(50, (900, 350), (700, 1400))
    session.mask = np.zeros(session.gray.shape, dtype=np.uint8)
    session.place_disc((900.0, 350.0))
    session.place_fovea((600.0, 360.0))
    scale = session.scale_um_px
    assert scale is not None and session.disc_radius_px
    session.mask = np.zeros(session.gray.shape, dtype=np.uint8)
    cv2.circle(session.mask, (600, 360), 20, 255, thickness=-1)
    area = session.live_stats()["area_mm2"]
    session.place_fovea((612.0, 366.0))
    assert session.scale_um_px == pytest.approx(scale, rel=0.001)
    assert session.live_stats()["area_mm2"] == pytest.approx(area)
    session._stash()
    session.index = 1
    session.gray = _disc_image(80, (900, 350), (700, 1400))
    session.mask = np.zeros(session.gray.shape, dtype=np.uint8)
    session.fovea_user_placed = False
    session.fovea_xy = None
    note = session.place_disc((906.0, 354.0))
    assert "1枚目" in note
    assert session.fovea_xy is not None
    later = session.scale_um_px
    assert later is not None and later < scale * 0.85
    dfd_before = math.hypot(
        session.fovea_xy[0] - session.disc_xy[0],
        session.fovea_xy[1] - session.disc_xy[1],
    )
    note = session.place_fovea((session.fovea_xy[0] + 10.0, session.fovea_xy[1] + 4.0))
    dfd_after = math.hypot(
        session.fovea_xy[0] - session.disc_xy[0],
        session.fovea_xy[1] - session.disc_xy[1],
    )
    assert "比" in note
    assert session.scale_um_px == pytest.approx(later * dfd_before / dfd_after)
    kept_fovea = session.fovea_xy
    session.place_disc((session.disc_xy[0] + 40.0, session.disc_xy[1]))
    assert session.fovea_user_placed is True
    assert session.fovea_xy == pytest.approx(kept_fovea)


def test_scale_from_dfd_matches_known_distance():
    scale = scale_from_dfd((200.0, 100.0), (100.0, 100.0), dfd_um=4760.0)
    assert scale == 47.6


def test_segment_optos_drops_disc_and_keeps_both_lesions():
    img, fovea, disc, scale = _scene()
    mask = segment_optos(img, fovea, disc, scale)
    assert mask[120, 230] == 0
    assert mask[102, 108] == 255
    assert mask[155, 115] == 255
    assert int(np.sum(disc_circle_mask(mask.shape, disc, scale) & (mask > 0))) == 0


def test_stats_central_counts_only_the_6mm_diameter():
    img, fovea, disc, scale = _scene()
    mask = np.zeros(img.shape, dtype=np.uint8)
    # 1.8 mm。直径 6 mm の中。
    mask[120, 180] = 255
    # 4.0 mm。以前の半径 6 mm には入っていたが、直径 6 mm の外。
    mask[120, 229] = 255
    # 9 mm。
    mask[120, 340] = 255
    stats = stats_central(mask, fovea, scale)
    assert stats["area_px"] == 1
    assert stats["outside_quant_px"] == 2
    assert stats["quant_radius_mm"] == 3.0


def test_24mm_model_is_smaller_away_from_the_fovea():
    fovea = (100.0, 100.0)
    scale = 47.6
    near_mask = np.zeros((200, 400), dtype=np.uint8)
    near_mask[100, 100] = 255
    far_mask = np.zeros_like(near_mask)
    far_mask[100, 180] = 255
    near = area_mm2_on_24mm_globe(near_mask, fovea, scale)
    far = area_mm2_on_24mm_globe(far_mask, fovea, scale)
    mm_per_px = scale / 1000.0
    assert near == pytest.approx(mm_per_px ** 2)
    assert far < near
    shifted = area_mm2_on_24mm_globe(near_mask, (110.0, 100.0), scale)
    assert shifted == pytest.approx(near, rel=0.01)
    stats = stats_central(near_mask, fovea, scale, (200.0, 100.0))
    assert stats["area_mm2_24mm"] == pytest.approx(near, abs=0.001)
    assert stats_central(near_mask, fovea, scale)["area_mm2"] == stats_central(
        near_mask, (110.0, 100.0), scale,
    )["area_mm2"]


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


def test_manual_grow_uses_the_segmentation_image(monkeypatch):
    from optos_ui.session import OptosSession

    img, fovea, disc, scale = _scene()
    session = OptosSession()
    session.gray = img
    session.disc_xy = disc
    session.fovea_xy = fovea
    session.scale_um_px = scale
    session.mask = np.zeros(img.shape, dtype=np.uint8)
    session.mask[int(fovea[1]), int(fovea[0]) - 40] = 255
    session.tool = "grow"
    seen = {}

    def fake(gray, mask, seed, scale_um_px, **kwargs):
        seen["gray"] = gray
        return mask, 0

    monkeypatch.setattr("optos_ui.session.grow_to_edge", fake)
    session.edit_at((fovea[0] - 40, fovea[1]), record=False)
    expected = growth_image(img, disc, scale)
    assert np.array_equal(seen["gray"], expected)
    assert not np.array_equal(seen["gray"], img)


def test_confirm_refuses_a_stale_mask():
    from optos_ui.session import OptosSession

    img, fovea, disc, scale = _scene()
    session = OptosSession()
    session.gray = img
    session.disc_xy = disc
    session.fovea_xy = fovea
    session.fovea_user_placed = True
    session.scale_um_px = scale
    session.mask = np.zeros(img.shape, dtype=np.uint8)
    session.mask[int(fovea[1]), int(fovea[0]) - 20] = 255
    session.mask_stale = True
    assert "再セグ" in session.confirm()
    assert session.confirmed is False


def test_earliest_visit_is_the_anchor_and_updates_later_ones():
    from pathlib import Path

    from engine.optos import series_scale_from_dfd
    from optos_ui.session import OptosSession

    # ファイル名順では a_20240601 が先。検査日では z_20200101 が1枚目。
    late_name = "a_20240601.png"
    early_name = "z_20200101.png"
    early_img = _disc_image(50, (900, 350), (700, 1400))
    late_img = _disc_image(80, (900, 350), (700, 1400))
    session = OptosSession()
    session.case_id = "OD"
    session.files = [Path(late_name), Path(early_name)]
    session.index = 0
    session.date_iso = "2024-06-01"
    session.gray = late_img
    session.mask = np.zeros(late_img.shape, dtype=np.uint8)
    session.place_disc((900.0, 350.0))
    session.place_fovea((700.0, 350.0))
    session.mask = np.zeros(late_img.shape, dtype=np.uint8)
    session.mask[350, 700] = 255
    session.mask_stale = False
    assert "確定" in session.confirm()
    own = session.scale_um_px
    late_radius = session.disc_radius_px
    assert own and late_radius
    assert session.snap_for(late_name).confirmed is True

    session.index = 1
    session.date_iso = "2020-01-01"
    session.gray = early_img
    session.mask = np.zeros(early_img.shape, dtype=np.uint8)
    session.fovea_xy = None
    session.fovea_user_placed = False
    session.disc_xy = None
    session.disc_radius_px = None
    session.scale_um_px = None
    session.place_disc((900.0, 350.0))
    session.place_fovea((600.0, 360.0))
    late = session.snap_for(late_name)
    assert late.fovea_xy == pytest.approx((700.0, 350.0))
    expected = series_scale_from_dfd(
        session.scale_um_px, session.disc_xy, session.fovea_xy, late.disc_xy, late.fovea_xy,
    )
    assert late.scale_um_px == pytest.approx(expected)
    assert late.scale_um_px == pytest.approx(own)

    session.place_fovea((520.0, 360.0))
    assert late.fovea_xy == pytest.approx((700.0, 350.0))
    moved = series_scale_from_dfd(
        session.scale_um_px, session.disc_xy, session.fovea_xy, late.disc_xy, late.fovea_xy,
    )
    assert late.scale_um_px == pytest.approx(moved)
    assert moved == pytest.approx(expected)


def test_confirm_refuses_an_empty_mask():
    from optos_ui.session import OptosSession

    img, fovea, disc, scale = _scene()
    session = OptosSession()
    session.gray = img
    session.disc_xy = disc
    session.fovea_xy = fovea
    session.fovea_user_placed = True
    session.scale_um_px = scale
    session.mask = np.zeros(img.shape, dtype=np.uint8)
    assert "領域がない" in session.confirm()
    assert session.confirmed is False


def test_later_visit_records_without_moving_the_proposed_fovea():
    from pathlib import Path

    from optos_ui.session import OptosSession, VisitSnap

    session = OptosSession()
    session.case_id = "OD"
    early = Path("eye_20240101.png")
    late = Path("eye_20240601.png")
    session.files = [early, late]
    session.index = 1
    session.date_iso = "2024-06-01"
    gray = np.full((80, 120), 180, dtype=np.uint8)
    prior = np.zeros(gray.shape, dtype=np.uint8)
    prior[30:40, 20:30] = 255
    session.snaps["OD/eye_20240101.png"] = VisitSnap(
        filename=early.name,
        path=str(early),
        date_iso="2024-01-01",
        date_source="filename",
        gray=gray,
        disc_xy=(90.0, 40.0),
        fovea_xy=(40.0, 40.0),
        scale_um_px=50.0,
        disc_radius_px=12.0,
        mask=prior,
        confirmed=True,
        fovea_user_placed=True,
    )
    session.gray = gray
    session.disc_xy = (92.0, 38.0)
    session.fovea_xy = (42.0, 41.0)
    session.fovea_user_placed = False
    session.fovea_proposal_source = "registration"
    session.fovea_proposal_xy = session.fovea_xy
    session.registration_zoom = 1.0
    session.scale_um_px = 50.0
    session.mask = prior.copy()
    session.mask_stale = False
    session.tool = "brush_add"
    session.edit_at((10.0, 35.0))
    note = session.confirm()
    assert "確定" in note
    assert "写した位置" in note
    assert session.confirmed is True
    assert session.fovea_user_placed is True
    assert session.fovea_xy == (42.0, 41.0)
    saved = session.snap_for(late.name)
    assert saved is not None and saved.confirmed is True
    assert saved.fovea_user_placed is True
    assert int(np.count_nonzero(saved.mask)) > int(np.count_nonzero(prior))


def test_confirm_waits_until_the_user_places_the_fovea():
    from optos_ui.session import OptosSession

    img, fovea, disc, _scale = _scene()
    session = OptosSession()
    session.gray = img
    session.mask = np.zeros(img.shape, dtype=np.uint8)
    session.place_disc(disc)
    assert session.fovea_xy is not None
    assert session.fovea_user_placed is False
    assert "中心窩" in session.confirm()
    assert session.confirmed is False
    session.place_fovea(fovea)
    assert "確定" in session.confirm()
    assert session.confirmed is True


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


def test_csv_exports_visits_and_withholds_years_when_close():
    from pathlib import Path

    from optos_ui.csv_export import SERIES_COLUMNS, VISIT_COLUMNS, csv_bytes
    from optos_ui.session import OptosSession, VisitSnap

    def mask_near():
        out = np.zeros((240, 240), dtype=np.uint8)
        cv2.circle(out, (70, 120), 30, 255, thickness=-1)
        return out

    session = OptosSession()
    session.case_id = "OD"
    session.files = [Path("a.png"), Path("b.png")]
    for name, date in (("a.png", "2020-01-01"), ("b.png", "2021-01-01")):
        session.snaps[f"OD/{name}"] = VisitSnap(
            filename=name, path=name, date_iso=date, date_source="filename",
            gray=np.full((240, 240), 180, dtype=np.uint8),
            disc_xy=(200.0, 120.0), fovea_xy=(120.0, 120.0),
            scale_um_px=10.0, mask=mask_near(), confirmed=True,
        )
    rows = session.visit_rows()
    assert len(rows) == 2
    assert rows[0]["area_mm2"] > 0
    text = csv_bytes(VISIT_COLUMNS, rows).decode("utf-8-sig")
    assert text.startswith("case_id,date,")
    note = session.run_series()
    assert "出しません" in note
    summary = session.series_summary
    assert summary is not None
    assert summary["years_to_fovea"] is None
    assert summary["gap_um_per_year"] is None
    assert summary["area_mm2_per_year"] is not None
    series = csv_bytes(SERIES_COLUMNS, [summary]).decode("utf-8-sig")
    assert "years_to_fovea" in series.splitlines()[0]
    assert series.splitlines()[1].split(",")[8] == ""


def test_browser_close_exits_only_when_no_client_remains(monkeypatch):
    import optos_app

    calls = []
    monkeypatch.setattr(optos_app.os, "_exit", lambda code: calls.append(code))
    optos_app._clients = 1
    optos_app._exit_if_no_clients()
    assert calls == []
    optos_app._clients = 0
    optos_app._exit_if_no_clients()
    assert calls == [0]
    optos_app._clients = 0


def test_series_table_marks_a_backward_step():
    from optos_ui.csv_export import mark_reversals

    rows = mark_reversals([
        {
            "area_mm_sqrt": 2.0,
            "gawain": 40.0,
            "dist_um_nearest": 200.0,
            "gap_um": 180.0,
            "intact_1mm_mm2": 1.5,
        },
        {
            "area_mm_sqrt": 1.5,
            "gawain": 50.0,
            "dist_um_nearest": 150.0,
            "gap_um": 220.0,
            "intact_1mm_mm2": 1.2,
        },
    ])
    assert rows[0]["reversed"] == []
    assert set(rows[1]["reversed"]) == {"area_mm_sqrt", "gawain", "gap_um"}


def test_sqrt_growth_ignores_a_column_that_disagrees_with_area():
    from engine.timeseries import estimate_growth_mm_sqrt_per_year

    results = [
        {"date": "2024-04-13", "stats": {"area_mm2": 0.566, "area_mm_sqrt": 0.795}},
        {"date": "2025-04-13", "stats": {"area_mm2": 0.566, "area_mm_sqrt": 1.795}},
    ]
    assert estimate_growth_mm_sqrt_per_year(results) == pytest.approx(0.0, abs=0.001)


def test_final_export_drops_a_leftover_second_reader_row():
    from optos_ui.session import OptosSession, VisitSnap

    session = OptosSession()
    session.case_id = "最終"
    gray = np.full((40, 40), 180, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    common = dict(
        filename="case1_OD_20260901.png",
        path="case1_OD_20260901.png",
        date_iso="2026-09-01",
        date_source="filename",
        gray=gray,
        disc_xy=(30.0, 20.0),
        fovea_xy=(12.0, 15.0),
        scale_um_px=20.0,
        mask=mask,
        confirmed=True,
        fovea_user_placed=True,
    )
    session.snaps["第2/case1_OD_20260901.png"] = VisitSnap(**common)
    session.snaps["最終/case1_OD_20260901.png"] = VisitSnap(**common)
    rows = session.reading_payload()
    assert [row["case_id"] for row in rows] == ["最終"]


def test_opening_the_final_role_does_not_keep_the_second_reader_image(tmp_path):
    import cv2

    from optos_ui.session import OptosSession

    image = tmp_path / "case1_OD_20260901.png"
    cv2.imwrite(str(image), np.full((32, 32), 180, dtype=np.uint8))
    session = OptosSession()
    session.case_id = "第2"
    session.files = [image]
    session.index = 0
    session.gray = np.full((32, 32), 40, dtype=np.uint8)
    session.mask = np.zeros((32, 32), dtype=np.uint8)
    session.mask[1, 1] = 255
    session.confirmed = True
    session.fovea_user_placed = True
    session.fovea_xy = (4.0, 4.0)
    session.disc_xy = (20.0, 16.0)
    session.scale_um_px = 12.0
    session.disc_radius_px = 6.0
    session.load_paths([image], label="最終")
    assert all(not key.startswith("第2/") for key in session.snaps)
    assert session.case_id == "最終"


def test_folder_images_all_stay_targets_and_exclusions_are_logged(tmp_path, caplog):
    import logging

    import cv2

    from optos_ui.session import OptosSession

    folder = tmp_path / "OD"
    folder.mkdir()
    names = [
        "eye_20260101.png",
        "eye_20260101_b.png",
        "eye_20260102.png",
        "eye_20260601.png",
    ]
    for name in names:
        cv2.imwrite(str(folder / name), np.full((16, 16), 180, dtype=np.uint8))
    (folder / "notes.txt").write_text("memo", encoding="utf-8")
    (folder / "broken.png").write_bytes(b"not-a-png")
    session = OptosSession()
    with caplog.at_level(logging.INFO, logger="ga.optos.reading"):
        status = session.load_folder(str(folder))
    assert [path.name for path in session.files] == ["broken.png", *names]
    assert "notes.txt" in status and "未対応の形式" in status
    assert "画像を読めない" in status
    assert any("notes.txt" in rec.message and "未対応" in rec.message for rec in caplog.records)
    assert any("broken.png" in rec.message and "読めない" in rec.message for rec in caplog.records)
    assert any("読影対象 5 枚" in rec.message and "eye_20260101_b.png" in rec.message for rec in caplog.records)
    gaps = dict(session.measurement_gaps())
    assert gaps["eye_20260101.png"] == "未確定"
    assert gaps["eye_20260101_b.png"] == "未確定"
    assert gaps["eye_20260102.png"] == "未確定"
    assert gaps["eye_20260601.png"] == "未確定"
    assert "読めない" in gaps["broken.png"]
    _excluded, report = session.exclusion_report()
    assert "notes.txt: 未対応の形式" in report
    assert "eye_20260601.png: 未確定" in report


def test_baseline_threshold_is_applied_to_a_later_image():
    from engine.optos import _cores_at_threshold, apply_contrast_gain, baseline_appearance

    baseline = np.full((120, 120), 200, dtype=np.uint8)
    cv2.circle(baseline, (60, 60), 8, 30, thickness=-1)
    gain, percentile = baseline_appearance(baseline, (60.0, 60.0), (110.0, 60.0), 25.0)
    assert 1.0 <= percentile <= 45.0
    later = np.full((120, 120), 230, dtype=np.uint8)
    cv2.circle(later, (60, 60), 8, 120, thickness=-1)
    adjusted = apply_contrast_gain(later, gain)
    disc = np.zeros(later.shape, dtype=bool)
    cores = _cores_at_threshold(adjusted, (60.0, 60.0), 25.0, disc, percentile, 0.001)
    assert cores[60, 60] == 255
    assert cores[8, 8] == 0


def test_threshold_logs_pixel_counts_before_and_after(caplog):
    import logging

    from engine.optos import segment_optos

    image = np.full((120, 120), 200, dtype=np.uint8)
    cv2.circle(image, (60, 60), 8, 30, thickness=-1)
    with caplog.at_level(logging.INFO, logger="ga.optos.segment"):
        segment_optos(image, (60.0, 60.0), (110.0, 60.0), 25.0, hypo_percentile=15.0)
    assert any("しきい値前" in rec.message and "しきい値後" in rec.message for rec in caplog.records)


def test_prior_mask_disc_size_and_small_lesions():
    from engine.optos import (
        align_prior_mask,
        apply_reading_exclusions,
        drop_small_lesions,
    )
    from optos_ui.csv_export import area_change_warnings
    from optos_ui.session import OptosSession, VisitSnap

    mask = np.zeros((220, 360), dtype=np.uint8)
    mask[100, 80] = 255
    moved, _preview = align_prior_mask(
        mask,
        (100.0, 100.0),
        (80.0, 100.0),
        (150.0, 120.0),
        (110.0, 120.0),
        (220, 360),
    )
    assert moved[120, 110] == 255
    turned = np.zeros((200, 200), dtype=np.uint8)
    turned[50, 80] = 255
    rotated, _preview = align_prior_mask(
        turned,
        (50.0, 50.0),
        (80.0, 50.0),
        (100.0, 100.0),
        (100.0, 140.0),
        (200, 200),
    )
    assert rotated[140, 100] == 255

    scale = 10.0
    lesions = np.zeros((80, 80), dtype=np.uint8)
    lesions[10:20, 10:20] = 255
    lesions[40:70, 40:70] = 255
    kept = drop_small_lesions(lesions, scale, 0.05)
    assert kept[15, 15] == 0
    assert kept[50, 50] == 255

    image = np.full((400, 400), 180, dtype=np.uint8)
    painted = np.zeros((400, 400), dtype=np.uint8)
    painted[0, 200] = 255
    painted[200, 310] = 255
    painted[50, 200] = 255
    cleaned = apply_reading_exclusions(painted, image, (200.0, 200.0), scale)
    assert cleaned[0, 200] == 0
    assert cleaned[200, 310] == 0
    assert cleaned[50, 200] == 255

    assert any("減りました" in note for note in area_change_warnings([
        ("2024-01-01", 1.0), ("2024-04-01", 0.8),
    ]))
    assert any("2倍" in note for note in area_change_warnings([
        ("2024-01-01", 1.0), ("2024-06-01", 2.5),
    ]))
    assert area_change_warnings([("2024-01-01", 1.0), ("2026-01-01", 3.0)]) == []

    session = OptosSession()
    session.case_id = "OD"
    early = Path("eye_20240101.png")
    late = Path("eye_20240601.png")
    session.files = [early, late]
    session.index = 1
    session.date_iso = "2024-06-01"
    gray = _disc_image(80, (900, 350), (700, 1400))
    session.gray = gray
    prior = np.zeros(gray.shape, dtype=np.uint8)
    # 1画素だと回転付きの最近傍補間で消えることがあるので、小さなブロブにする。
    cv2.circle(prior, (860, 350), 2, 255, -1)
    session.snaps["OD/eye_20240101.png"] = VisitSnap(
        filename=early.name,
        path=str(early),
        date_iso="2024-01-01",
        date_source="filename",
        gray=gray,
        disc_xy=(900.0, 350.0),
        fovea_xy=(600.0, 350.0),
        scale_um_px=15.0,
        disc_radius_px=40.0,
        mask=prior,
        confirmed=True,
        fovea_user_placed=True,
    )
    note = session.place_disc((900.0, 350.0))
    assert "初期輪郭" in note
    assert session.mask_from_prior is True
    assert session.alignment_gray is None
    origin = session.fovea_xy
    disc = session.disc_xy
    scale0 = session.scale_um_px
    assert origin is not None and disc is not None and scale0 is not None
    d0 = math.hypot(origin[0] - disc[0], origin[1] - disc[1])
    moved = session.place_fovea((origin[0] + 20.0, origin[1] - 6.0))
    d1 = math.hypot(session.fovea_xy[0] - session.disc_xy[0], session.fovea_xy[1] - session.disc_xy[1])
    assert "拡大縮小" in moved
    assert session.scale_um_px == pytest.approx(scale0 * d0 / d1)
    assert session.mask_stale is False
    prior_snap = session.snaps["OD/eye_20240101.png"]
    expected_mask, _preview = align_prior_mask(
        prior_snap.mask,
        prior_snap.disc_xy,
        prior_snap.fovea_xy,
        session.disc_xy,
        session.fovea_xy,
        session.gray.shape[:2],
        prior_gray=prior_snap.gray,
        gray=session.gray,
    )
    assert np.array_equal(session.mask, expected_mask)
    session.place_fovea((origin[0] + 30.0, origin[1] + 4.0))
    assert session.mask_stale is False
    expected_again, _preview = align_prior_mask(
        prior_snap.mask,
        prior_snap.disc_xy,
        prior_snap.fovea_xy,
        session.disc_xy,
        session.fovea_xy,
        session.gray.shape[:2],
        prior_gray=prior_snap.gray,
        gray=session.gray,
    )
    assert np.array_equal(session.mask, expected_again)
    before = int(np.count_nonzero(session.mask))
    session.tool = "brush_add"
    session.edit_at((40.0, 40.0))
    assert int(np.count_nonzero(session.mask)) > before


def test_os_decreases_warn_on_confirm_and_in_the_csv(tmp_path):
    from ga_reading.bundle import write_reader_csv
    from ga_reading.merge import read_csv
    from optos_ui.csv_export import area_change_warnings
    from optos_ui.session import OptosSession, VisitSnap

    visits = [
        ("2024-04-13", 4.356),
        ("2024-08-06", 3.186),
        ("2024-10-15", 4.879),
        ("2024-12-17", 3.977),
        ("2025-04-15", 5.29),
        ("2026-01-27", 4.963),
        ("2026-05-26", 5.083),
        ("2026-07-28", 3.133),
        ("2026-09-01", 6.203),
        ("2026-10-27", 6.141),
    ]
    notes = area_change_warnings(visits)
    warned = {note[:10] for note in notes if "減りました" in note}
    assert warned == {
        "2024-08-06",
        "2024-12-17",
        "2026-01-27",
        "2026-07-28",
        "2026-10-27",
    }

    def _blob(n: int) -> np.ndarray:
        image = np.zeros((40, 40), dtype=np.uint8)
        image[5:5 + n, 5:5 + n] = 255
        return image

    session = OptosSession()
    session.case_id = "OS"
    early = Path("case1_OS_20240413.png")
    late = Path("case1_OS_20240806.png")
    session.files = [early, late]
    session.index = 1
    session.date_iso = "2024-08-06"
    session.gray = np.full((40, 40), 180, dtype=np.uint8)
    session.mask = _blob(4)
    session.disc_xy = (30.0, 20.0)
    session.fovea_xy = (12.0, 15.0)
    session.fovea_user_placed = True
    session.scale_um_px = 20.0
    session.snaps["OS/case1_OS_20240413.png"] = VisitSnap(
        filename=early.name,
        path=str(early),
        date_iso="2024-04-13",
        date_source="filename",
        gray=session.gray,
        disc_xy=(30.0, 20.0),
        fovea_xy=(12.0, 15.0),
        scale_um_px=20.0,
        mask=_blob(12),
        confirmed=True,
        fovea_user_placed=True,
    )
    message = session.confirm()
    assert "減りました" in message
    # 面積減とともに、中心窩が萎縮の中（縁まで 0）から外へ出た逆行も付く。
    assert session.needs_review == "area_decrease;gap_reopened"
    saved = write_reader_csv(tmp_path / "out", session.reading_payload())
    _fields, rows = read_csv(saved)
    by_date = {row["date"]: row for row in rows}
    assert "needs_review" in _fields
    assert by_date["2024-08-06"]["needs_review"] == "area_decrease;gap_reopened"
    assert by_date["2024-04-13"]["needs_review"] == ""


def test_confirm_flags_gap_reopened_when_the_fovea_leaves_the_atrophy():
    from optos_ui.csv_export import gap_reopen_warnings
    from optos_ui.session import OptosSession, VisitSnap

    # OD_reading の実測列。0 → 正 の 2 箇所だけに警告が出る。
    notes = gap_reopen_warnings([
        ("2024-04-13", 119.7), ("2024-08-06", 52.7), ("2024-10-15", 0.0),
        ("2024-12-17", 0.0), ("2025-04-15", 54.5), ("2026-01-27", 56.4),
        ("2026-05-26", 56.4), ("2026-07-28", 0.0), ("2026-09-01", 109.0),
        ("2026-10-27", 0.0),
    ])
    assert [note[:10] for note in notes] == ["2025-04-15", "2026-09-01"]

    def _blob(n: int) -> np.ndarray:
        image = np.zeros((40, 40), dtype=np.uint8)
        image[5:5 + n, 5:5 + n] = 255
        return image

    def _session(prev_fovea, current_fovea) -> "OptosSession":
        session = OptosSession()
        session.case_id = "OD"
        early = Path("case1_OD_20240413.png")
        late = Path("case1_OD_20240806.png")
        session.files = [early, late]
        session.index = 1
        session.date_iso = "2024-08-06"
        session.gray = np.full((40, 40), 180, dtype=np.uint8)
        session.mask = _blob(13)
        session.disc_xy = (30.0, 20.0)
        session.fovea_xy = current_fovea
        session.fovea_user_placed = True
        session.scale_um_px = 20.0
        session.snaps["OD/case1_OD_20240413.png"] = VisitSnap(
            filename=early.name,
            path=str(early),
            date_iso="2024-04-13",
            date_source="filename",
            gray=session.gray,
            disc_xy=(30.0, 20.0),
            fovea_xy=prev_fovea,
            scale_um_px=20.0,
            mask=_blob(12),
            confirmed=True,
            fovea_user_placed=True,
        )
        return session

    # 前回は萎縮の中（縁まで 0）、今回は縁から離れた → 逆行なので要レビュー。
    session = _session(prev_fovea=(12.0, 15.0), current_fovea=(25.0, 15.0))
    message = session.confirm()
    assert "萎縮は戻らない" in message
    assert session.needs_review == "gap_reopened"

    # 逆方向（進行で縁まで 0 になる）は正常なので付かない。
    session = _session(prev_fovea=(25.0, 15.0), current_fovea=(12.0, 15.0))
    message = session.confirm()
    assert "萎縮は戻らない" not in message
    assert session.needs_review == ""


def test_empty_resegment_keeps_the_previous_mask(caplog):
    import logging

    from optos_ui.session import OptosSession

    session = OptosSession()
    session.gray = np.full((80, 80), 180, dtype=np.uint8)
    session.mask = np.zeros((80, 80), dtype=np.uint8)
    session.mask[20:40, 20:40] = 255
    session.disc_xy = (60.0, 40.0)
    session.fovea_xy = (30.0, 40.0)
    session.scale_um_px = 20.0
    kept = session.mask.copy()

    def _empty(crop, *_args, **_kwargs):
        return np.zeros(crop.shape[:2], dtype=np.uint8)

    with caplog.at_level(logging.INFO, logger="ga.optos.reading"):
        import optos_ui.session as session_mod
        original = session_mod.segment_optos
        session_mod.segment_optos = _empty
        try:
            message = session.resegment()
        finally:
            session_mod.segment_optos = original
    assert "空" in message
    assert "除外後 0 px" in message
    assert np.array_equal(session.mask, kept)
    assert any("除外前 0 px、除外後 0 px" in rec.message for rec in caplog.records)


def test_ecc_nudges_the_prior_mask_onto_the_vessel():
    from engine.optos import align_prior_mask

    rng = np.random.default_rng(0)
    prior = rng.integers(140, 200, size=(160, 160), dtype=np.uint8)
    cv2.line(prior, (20, 70), (140, 90), 30, 3)
    cv2.line(prior, (40, 30), (80, 140), 25, 2)
    shift = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 6.0]], dtype=np.float32)
    current = cv2.warpAffine(prior, shift, (160, 160), borderMode=cv2.BORDER_REFLECT)
    mask = np.zeros((160, 160), dtype=np.uint8)
    mask[80, 80] = 255
    moved, preview = align_prior_mask(
        mask,
        (80.0, 80.0),
        (40.0, 80.0),
        (80.0, 80.0),
        (40.0, 80.0),
        (160, 160),
        prior_gray=prior,
        gray=current,
    )
    assert preview is not None
    ys, xs = np.nonzero(moved)
    assert abs(int(xs[0]) - 84) <= 2
    assert abs(int(ys[0]) - 86) <= 2


def _vessel_texture(shape: tuple[int, int], seed: int) -> np.ndarray:
    """血管に見立てた暗い線のある合成眼底。特徴点の位置合わせ用。"""
    rng = np.random.default_rng(seed)
    image = rng.integers(130, 190, size=shape, dtype=np.uint8)
    h, w = shape
    for _ in range(30):
        a = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        b = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(image, a, b, int(rng.integers(20, 60)), int(rng.integers(2, 4)))
    return cv2.GaussianBlur(image, (0, 0), 1.0)


def test_registered_fovea_proposal_follows_rotation():
    from engine.optos import register_fovea_from_prior

    prior = _vessel_texture((400, 400), seed=3)
    warp = cv2.getRotationMatrix2D((200.0, 200.0), 4.0, 1.0)
    warp[:, 2] += (12.0, -8.0)
    current = cv2.warpAffine(prior, warp, (400, 400), borderMode=cv2.BORDER_REFLECT)
    fovea = (150.0, 230.0)
    expected = warp @ np.array([fovea[0], fovea[1], 1.0])
    got = register_fovea_from_prior(prior, fovea, current)
    assert got is not None
    (gx, gy), zoom = got
    assert gx == pytest.approx(expected[0], abs=2.5)
    assert gy == pytest.approx(expected[1], abs=2.5)
    assert zoom == pytest.approx(1.0, abs=0.03)
    # スケールを渡すと中心 8mm のクロップで合わせ直す。結果はほぼ同じ。
    cropped = register_fovea_from_prior(prior, fovea, current, scale_um_px=50.0)
    assert cropped is not None
    assert cropped[0][0] == pytest.approx(expected[0], abs=2.5)
    assert cropped[0][1] == pytest.approx(expected[1], abs=2.5)
    # 特徴がなければ None（呼び出し側が乳頭ベクトルの提案に戻す）。
    flat = np.full((400, 400), 150, dtype=np.uint8)
    assert register_fovea_from_prior(flat, fovea, flat) is None


def test_second_visit_proposes_the_fovea_from_vessel_registration():
    from optos_ui.session import OptosSession, VisitSnap

    shape = (500, 700)
    disc = (600.0, 150.0)
    fovea = (350.0, 250.0)
    base = _vessel_texture(shape, seed=5)
    cv2.circle(base, (int(disc[0]), int(disc[1])), 40, 30, -1)
    warp = cv2.getRotationMatrix2D((350.0, 250.0), 4.0, 1.0)
    warp[:, 2] += (10.0, -6.0)
    current = cv2.warpAffine(base, warp, (shape[1], shape[0]), borderMode=cv2.BORDER_REFLECT)
    disc_now = warp @ np.array([disc[0], disc[1], 1.0])
    fovea_now = warp @ np.array([fovea[0], fovea[1], 1.0])

    session = OptosSession()
    session.case_id = "OD"
    session.files = [Path("a_first.png"), Path("b_later.png")]
    session.index = 1
    session.date_iso = "2024-06-01"
    session.gray = current
    session.mask = np.zeros(shape, dtype=np.uint8)
    session.snaps["OD/a_first.png"] = VisitSnap(
        filename="a_first.png",
        path="a_first.png",
        date_iso="2024-01-01",
        date_source="filename",
        gray=base,
        disc_xy=disc,
        fovea_xy=fovea,
        scale_um_px=50.0,
        disc_radius_px=40.0,
        mask=None,
        confirmed=False,
        fovea_user_placed=True,
    )
    note = session.place_disc((float(disc_now[0]), float(disc_now[1])))
    assert session.fovea_proposal_source == "registration"
    assert "血管" in note
    assert session.fovea_xy[0] == pytest.approx(fovea_now[0], abs=3.0)
    assert session.fovea_xy[1] == pytest.approx(fovea_now[1], abs=3.0)
    # スケールも倍率から（回転のみなので 1枚目の 50 µm/px とほぼ同じ）。
    assert session.registration_zoom == pytest.approx(1.0, abs=0.03)
    assert session.scale_um_px == pytest.approx(50.0, rel=0.03)

    # 提案から大きく離れたクリックは、手動を優先して DFD 比に戻し、確認を促す。
    proposal = session.fovea_proposal_xy
    note = session.place_fovea((proposal[0] + 20.0, proposal[1]))
    assert "離れています" in note
    assert session.registration_zoom is None
    assert session.fovea_xy[0] == pytest.approx(proposal[0] + 20.0)
    # 提案のすぐそばのクリックは「確認」とみなし、写した位置に合わせる。
    note = session.place_fovea((proposal[0] + 1.0, proposal[1]))
    assert "離れています" not in note
    assert "写した位置に置きました" in note
    # クリック中心の再位置合わせなので、元の提案と厳密一致はしない。
    assert session.fovea_xy[0] == pytest.approx(proposal[0], abs=2.0)
    assert session.fovea_xy[1] == pytest.approx(proposal[1], abs=2.0)
    assert session.registration_zoom is not None
    assert session.scale_um_px == pytest.approx(50.0, rel=0.03)


def test_reader_role_does_not_prefix_the_export_filename():
    from optos_ui.session import OptosSession, VisitSnap

    session = OptosSession()
    session.case_id = "第2"
    gray = np.full((40, 40), 180, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    session.snaps["第2/2022-02-25.png"] = VisitSnap(
        filename="2022-02-25.png",
        path="2022-02-25.png",
        date_iso="2022-02-25",
        date_source="filename",
        gray=gray,
        disc_xy=(30.0, 20.0),
        fovea_xy=(12.0, 15.0),
        scale_um_px=20.0,
        mask=mask,
        confirmed=True,
        fovea_user_placed=True,
    )
    rows = session.reading_payload()
    assert len(rows) == 1
    assert rows[0]["filename"] == "2022-02-25.png"


def test_confirm_fovea_returns_the_usual_mask_colors():
    from optos_ui.session import OptosSession

    session = OptosSession()
    session.gray = np.full((80, 80), 180, dtype=np.uint8)
    session.disc_xy = (60.0, 40.0)
    session.fovea_xy = (30.0, 40.0)
    session.scale_um_px = 20.0
    session.mask = np.zeros((80, 80), dtype=np.uint8)
    session.mask[20:40, 20:40] = 255
    session.mask_from_prior = True
    session.alignment_gray = np.full((80, 80), 90, dtype=np.uint8)
    session.fovea_user_placed = False
    message = session.confirm_fovea()
    assert "いつもの色" in message
    assert session.fovea_user_placed is True
    assert session.alignment_gray is None
    assert session.mask_stale is False
    assert int(np.count_nonzero(session.mask)) == 400


def test_arrow_key_moves_the_image_and_fovea_mark_is_one_pixel():
    from optos_ui.app import draw_screen_marks, pan_delta_for_key

    right = pan_delta_for_key("Arrow Right", 2.0)
    left = pan_delta_for_key("Arrow Left", 2.0)
    assert right is not None and left is not None
    assert right[0] < 0
    assert left[0] > 0
    assert pan_delta_for_key("a", 2.0) is None
    shown = np.zeros((80, 80, 3), dtype=np.uint8)
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[30:50, 30:50] = 255
    draw_screen_marks(
        shown,
        fovea_xy=(40.0, 40.0),
        disc_xy=None,
        scale_um_px=None,
        mask=mask,
        origin=(0.0, 0.0),
        zoom=1.0,
        crop_xy=(0.0, 0.0),
        contour_rgb=(0, 255, 80),
    )
    assert shown[40, 24, 1] > 0
    assert shown[39, 24, 1] == 0
    assert shown[41, 24, 1] == 0
    assert shown[30, 40, 1] > 0


def test_cyan_circle_is_six_mm_in_diameter():
    from engine.optos import QUANT_RADIUS_MM, px_from_mm
    from optos_ui.app import draw_screen_marks

    assert QUANT_RADIUS_MM == 3.0
    scale = 50.0
    radius = px_from_mm(QUANT_RADIUS_MM, scale)
    shown = np.zeros((400, 400, 3), dtype=np.uint8)
    draw_screen_marks(
        shown,
        fovea_xy=(200.0, 200.0),
        disc_xy=None,
        scale_um_px=scale,
        mask=None,
        origin=(0.0, 0.0),
        zoom=1.0,
        crop_xy=(0.0, 0.0),
        contour_rgb=(0, 255, 80),
    )
    assert shown[200, 200 + radius, 2] > 200
    assert int(shown[200, 200 + 2 * radius].sum()) == 0


def test_provisional_fovea_opens_the_macula_window():
    from optos_ui.app import uses_macula_window

    assert uses_macula_window(False, (10.0, 10.0), 15.0) is True
    assert uses_macula_window(True, (10.0, 10.0), 15.0) is True
    assert uses_macula_window(False, None, 15.0) is False
    assert uses_macula_window(True, (10.0, 10.0), None) is False


def test_analysis_circle_is_centered_in_the_view():
    from optos_ui.app import VIEW_H, VIEW_W, analysis_display_zoom, centered_crop_origin

    scale = 10.0
    zoom = analysis_display_zoom(scale, 1)
    assert zoom < 1
    diameter = 2 * (6.0 * 1000 / scale) * zoom
    assert diameter <= min(VIEW_W, VIEW_H)
    coarse = analysis_display_zoom(40.0, 1)
    coarse_diameter = 2 * (6.0 * 1000 / 40.0) * coarse
    assert coarse > 1
    assert coarse_diameter == pytest.approx(min(VIEW_W, VIEW_H) / 1.08, rel=0.02)
    center = (800.0, 800.0)
    zw = int(round(1600 * zoom))
    zh = int(round(1600 * zoom))
    x0, y0 = centered_crop_origin(center, (0.0, 0.0), zoom, (zw, zh))
    shown_cx = x0 + min(VIEW_W, zw) / 2
    shown_cy = y0 + min(VIEW_H, zh) / 2
    assert abs(shown_cx - center[0] * zoom) <= 1
    assert abs(shown_cy - center[1] * zoom) <= 1


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
    assert "中心窩まで 1000 µm" in far
    assert "接近速度 100.0 µm/年" in far
    assert "中心窩距離" not in far


def test_visit_switch_keeps_the_series_and_case_switch_clears_it():
    from pathlib import Path

    from optos_ui.session import OptosSession, VisitSnap

    def snap(name: str, date: str) -> VisitSnap:
        return VisitSnap(
            filename=name, path=name, date_iso=date, date_source="filename",
            gray=np.full((80, 80), 180, dtype=np.uint8),
            disc_xy=(60.0, 40.0), fovea_xy=(40.0, 40.0),
            scale_um_px=10.0,
            mask=np.full((80, 80), 255, dtype=np.uint8),
            confirmed=True, fovea_user_placed=True,
        )

    session = OptosSession()
    session.case_id = "OD"
    session.cases = {"OD": [Path("a.png"), Path("b.png")], "OS": [Path("c.png")]}
    session.files = list(session.cases["OD"])
    session.index = 0
    session.snaps = {
        "OD/a.png": snap("a.png", "2020-01-01"),
        "OD/b.png": snap("b.png", "2021-01-01"),
        "OS/c.png": snap("c.png", "2020-06-01"),
    }
    first = session.snaps["OD/a.png"]
    session.gray = first.gray
    session.mask = first.mask.copy()
    session.disc_xy = first.disc_xy
    session.fovea_xy = first.fovea_xy
    session.scale_um_px = first.scale_um_px
    session.confirmed = True
    session.fovea_user_placed = True
    session.date_iso = first.date_iso
    session.date_source = first.date_source
    note = session.run_series()
    assert session.series_summary is not None
    session.open_index(1)
    assert session.series_note == note
    assert session.series_summary is not None
    assert session.series_visit_table
    session.select_case("OS")
    assert session.series_summary is None
    assert session.series_visit_table == []
    assert session.series_note == ""


def test_scale_disagreement_note_only_when_far_apart():
    assert scale_disagreement_note(10.0, 12.0) == ""
    note = scale_disagreement_note(15.8, 10.0)
    assert "15.8" in note and "10.0" in note


def test_tool_colors_mark_only_the_active_tool():
    from optos_ui.app import _tool_colors, _tool_name

    on_fg, on_bg = _tool_colors(True)
    off_fg, off_bg = _tool_colors(False)
    assert on_fg != off_fg
    assert on_bg != off_bg
    assert _tool_name("fovea") == "中心窩"
    assert _tool_name("brush_erase") == "消す"


def test_sync_option_list_updates_labels_without_replacing_controls():
    import flet as ft

    from optos_ui.app import _sync_option_list

    options = [ft.dropdown.Option("0", "a"), ft.dropdown.Option("1", "b")]
    first, second = options
    _sync_option_list(options, [("0", "済 a"), ("1", "b")])
    assert options[0] is first and options[1] is second
    assert options[0].text == "済 a"
    _sync_option_list(options, [("0", "済 a")])
    assert options == [first]
    _sync_option_list(options, [("0", "済 a"), ("2", "c")])
    assert options[0] is first
    assert options[1].key == "2"
    assert options[1].text == "c"


def test_drop_border_keeps_a_central_blob():
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[0:5, 0:8] = 255
    cv2.circle(mask, (40, 30), 6, 255, thickness=-1)
    kept = drop_border_touching(mask)
    assert kept[0, 0] == 0
    assert kept[30, 40] == 255
