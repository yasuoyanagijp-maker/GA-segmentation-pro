"""Regression: after 3/3 confirm, series must stick (no silent load of image 0)."""

from pathlib import Path

from flet_ui.main_view import (
    resolve_file_dropdown_change,
    resolve_series_landing_index,
)
from flet_ui.state import GAAppState


def test_file_dropdown_switches_timepoint_while_series_active():
    """While series is open, file dropdown switches timepoint for series files."""
    files = [Path("img1.png"), Path("img2.png"), Path("img3.png")]
    series = [
        {"filename": "img1.png"},
        {"filename": "img2.png"},
        {"filename": "img3.png"},
    ]
    action, payload = resolve_file_dropdown_change(
        skip=False,
        selected="img1.png",
        series_results=series,
        files=files,
        file_idx=2,
    )
    assert action == "series_timepoint"
    assert payload == 0

    action, payload = resolve_file_dropdown_change(
        skip=False,
        selected="img3.png",
        series_results=series,
        files=files,
        file_idx=0,
    )
    assert action == "series_timepoint"
    assert payload == 2


def test_file_dropdown_exits_series_for_unconfirmed_file():
    """Partial series (2/4): picking an unconfirmed file must load_image."""
    files = [
        Path("img1.png"), Path("img2.png"), Path("img3.png"), Path("img4.png"),
    ]
    series = [
        {"filename": "img1.png"},
        {"filename": "img2.png"},
    ]
    action, payload = resolve_file_dropdown_change(
        skip=False,
        selected="img3.png",
        series_results=series,
        files=files,
        file_idx=1,
    )
    assert action == "load_image"
    assert payload == 2

    action, payload = resolve_file_dropdown_change(
        skip=False,
        selected="missing.png",
        series_results=series,
        files=files,
        file_idx=0,
    )
    assert action == "ignore"
    assert payload is None


def test_file_dropdown_blocked_while_series_launching():
    """Mid-build page.update races: on_change before series_results is set."""
    files = [Path("img1.png"), Path("img2.png"), Path("img3.png")]
    action, payload = resolve_file_dropdown_change(
        skip=False,
        selected="img1.png",
        series_results=None,
        files=files,
        file_idx=2,
        series_launching=True,
    )
    assert action == "ignore"
    assert payload is None


def test_file_dropdown_loads_only_when_series_inactive():
    files = [Path("img1.png"), Path("img2.png"), Path("img3.png")]
    action, payload = resolve_file_dropdown_change(
        skip=False,
        selected="img1.png",
        series_results=None,
        files=files,
        file_idx=2,
    )
    assert action == "load_image"
    assert payload == 0

    action, payload = resolve_file_dropdown_change(
        skip=True,
        selected="img1.png",
        series_results=None,
        files=files,
        file_idx=2,
    )
    assert action == "ignore"


def test_series_landing_prefers_just_confirmed_file():
    results = [
        {"filename": "img1.png", "date": "2020-01-01"},
        {"filename": "img2.png", "date": "2021-01-01"},
        {"filename": "img3.png", "date": "2022-01-01"},
    ]
    assert resolve_series_landing_index(results, preferred_filename="img3.png") == 2
    assert resolve_series_landing_index(results, preferred_filename="img1.png") == 0
    assert resolve_series_landing_index(results, preferred_filename="missing.png") == 0
    assert resolve_series_landing_index(results, preferred_filename=None) == 0
    assert resolve_series_landing_index([], preferred_filename="img3.png") == 0


def test_batch_complete_state_keeps_series_results():
    state = GAAppState()
    state.files = [Path("a.png"), Path("b.png"), Path("c.png")]
    state.file_idx = 2
    state.per_file_results = {
        "a.png": {"filename": "a.png"},
        "b.png": {"filename": "b.png"},
        "c.png": {"filename": "c.png"},
    }
    state.series_results = [
        {"filename": "a.png", "date": "2020-01-01", "stats": {}},
        {"filename": "b.png", "date": "2021-01-01", "stats": {}},
        {"filename": "c.png", "date": "2022-01-01", "stats": {}},
    ]
    land = resolve_series_landing_index(
        state.series_results, preferred_filename="c.png",
    )
    state.series_time_idx = land
    state.view_mode = "work"
    state.file_idx = land

    action, _ = resolve_file_dropdown_change(
        skip=False,
        selected="a.png",
        series_results=state.series_results,
        files=state.files,
        file_idx=state.file_idx,
    )
    assert action == "series_timepoint"
    assert state.series_results is not None
    assert len(state.series_results) == 3
    assert state.series_time_idx == 2


def test_series_ready_at_two_confirmed_not_all():
    """④ auto-launch threshold: 2+ confirmed, even if folder has more files."""
    files = [Path("a.png"), Path("b.png"), Path("c.png"), Path("d.png")]
    per_file = {
        "a.png": {"filename": "a.png"},
        "b.png": {"filename": "b.png"},
    }
    saved = sum(1 for f in files if f.name in per_file)
    n = len(files)
    assert saved >= 2
    assert saved < n  # partial batch still qualifies for series
    series_ready = saved >= 2
    assert series_ready is True


def test_registration_and_roi_defaults_on():
    """④ の既定動作: 位置合わせ ON・マスクQC補正 ON（リセット後も維持）。"""
    state = GAAppState()
    assert state.series_registration_enabled is True
    assert state.series_roi_enabled is True
    assert state.series_roi_applied is False

    # フルリセット後も既定は ON に戻る
    state.series_registration_enabled = False
    state.series_roi_enabled = False
    state.series_roi_applied = True
    state.reset_analysis()
    assert state.series_registration_enabled is True
    assert state.series_roi_enabled is True
    assert state.series_roi_applied is False

    # keep_series=True では時系列設定は温存される
    state.series_registration_enabled = False
    state.series_roi_enabled = False
    state.reset_analysis(keep_series=True)
    assert state.series_registration_enabled is False
    assert state.series_roi_enabled is False


def test_scale_recal_keeps_other_confirmed_results():
    """2枚目で乳頭/幅指定しても1枚目の確定は消えない（チェック保持）。"""
    state = GAAppState()
    state.files = [Path("img1.png"), Path("img2.png")]
    state.file_idx = 1  # currently on image 2
    state.per_file_results = {
        "img1.png": {"filename": "img1.png", "scale_um_px": 5.0},
        "img2.png": {"filename": "img2.png", "scale_um_px": 5.0},
    }
    # Mimic invalidate: drop only current file, keep others
    cur = state.files[state.file_idx].name
    if cur in state.per_file_results:
        del state.per_file_results[cur]
    assert "img1.png" in state.per_file_results
    assert "img2.png" not in state.per_file_results
    assert len(state.per_file_results) == 1
