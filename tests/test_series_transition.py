"""Regression: after 3/3 confirm, series must stick (no silent load of image 0)."""

from pathlib import Path

from flet_ui.main_view import resolve_file_dropdown_change
from flet_ui.state import GAAppState


def test_file_dropdown_never_loads_while_series_active():
    """Spurious dropdown on_change after auto-launch must not clear / load image 0."""
    files = [Path("img1.png"), Path("img2.png"), Path("img3.png")]
    series = [
        {"filename": "img1.png"},
        {"filename": "img2.png"},
        {"filename": "img3.png"},
    ]
    # Same file already showing (earliest date = folder index 0): must ignore.
    action, payload = resolve_file_dropdown_change(
        skip=False,
        selected="img1.png",
        series_results=series,
        files=files,
        file_idx=2,  # user was on image 3/3 when finishing
    )
    assert action == "series_timepoint"
    assert payload == 0

    # Unknown / wrong selection while series active: ignore (do not load_image).
    action, payload = resolve_file_dropdown_change(
        skip=False,
        selected="img1.png",
        series_results=series,
        files=files,
        file_idx=0,
    )
    assert action == "series_timepoint"
    assert payload == 0

    action, payload = resolve_file_dropdown_change(
        skip=False,
        selected="missing.png",
        series_results=series,
        files=files,
        file_idx=0,
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


def test_batch_complete_state_keeps_series_results():
    """Simulate post-3/3 apply: series_results stays set; file_idx may be 0."""
    state = GAAppState()
    state.files = [Path("a.png"), Path("b.png"), Path("c.png")]
    state.file_idx = 2
    state.per_file_results = {
        "a.png": {"filename": "a.png"},
        "b.png": {"filename": "b.png"},
        "c.png": {"filename": "c.png"},
    }
    # Mimic _apply_series_results after build (date-sorted → first file)
    state.series_results = [
        {"filename": "a.png", "date": "2020-01-01", "stats": {}},
        {"filename": "b.png", "date": "2021-01-01", "stats": {}},
        {"filename": "c.png", "date": "2022-01-01", "stats": {}},
    ]
    state.series_overlays = [object(), object(), object()]  # type: ignore[list-item]
    state.series_time_idx = 0
    state.view_mode = "work"
    state.file_idx = 0  # synced to first timepoint

    # Spurious dropdown event selecting image 1 must not imply "exit series"
    action, _ = resolve_file_dropdown_change(
        skip=False,
        selected="a.png",
        series_results=state.series_results,
        files=state.files,
        file_idx=state.file_idx,
    )
    assert action != "load_image"
    assert state.series_results is not None
    assert len(state.series_results) == 3
