"""Regression: after 3/3 confirm, series must stick (no silent load of image 0)."""

from pathlib import Path

from flet_ui.main_view import (
    resolve_file_dropdown_change,
    resolve_series_landing_index,
)
from flet_ui.state import GAAppState


def test_file_dropdown_never_loads_while_series_active():
    """Spurious dropdown on_change after auto-launch must not clear / jump."""
    files = [Path("img1.png"), Path("img2.png"), Path("img3.png")]
    series = [
        {"filename": "img1.png"},
        {"filename": "img2.png"},
        {"filename": "img3.png"},
    ]
    # Selecting earliest file while series open must NOT navigate (looks like image 1).
    action, payload = resolve_file_dropdown_change(
        skip=False,
        selected="img1.png",
        series_results=series,
        files=files,
        file_idx=2,
    )
    assert action == "ignore"
    assert payload is None

    action, payload = resolve_file_dropdown_change(
        skip=False,
        selected="img3.png",
        series_results=series,
        files=files,
        file_idx=2,
    )
    assert action == "ignore"
    assert payload is None

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
        series_results=None,  # not yet assigned
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
    """3/3 auto-launch must not snap visually to earliest (folder image 1)."""
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
    """Simulate post-3/3 apply: series_results stays set on landing timepoint."""
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
    assert action == "ignore"
    assert state.series_results is not None
    assert len(state.series_results) == 3
    assert state.series_time_idx == 2
