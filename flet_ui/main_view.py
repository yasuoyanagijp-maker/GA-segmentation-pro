"""Flet メインビュー — engine を直接呼び出す（FastAPI なし）。"""

from __future__ import annotations

import asyncio
import base64
import copy
from datetime import date, datetime
from pathlib import Path

import cv2
import flet as ft
import numpy as np

from engine import calibration, segmentation
from engine.crop import apply_crop, normalize_box
from engine.io_utils import SUPPORTED_SUFFIXES, list_images, load_gray
from engine.life_table import LIFE_TABLE_SOURCE, format_life_expectancy, life_expectancy_years
from engine.overlay import make_overlay
from engine.region_growing import (
    clear_mask_in_freehand_region,
    erase_oversegmentation_from_mask,
    expand_segmentation_from_mask,
)
from engine.export import (
    build_batch_csv_bytes,
    generate_pdf_report,
    suggested_csv_filename,
    suggested_pdf_filename,
)
from engine.foveal_correction import (
    apply_foveal_block_correction,
    correction_effect_stats,
    gray_for_segmentation,
)
from engine.gawain import GAWAIN_LABEL, format_gawain_display
from engine.stats import ga_stats
from engine.timeseries import (
    build_series_from_saved,
    estimate_growth_mm2_per_year,
    estimate_growth_mm_sqrt_per_year,
    estimate_progression_rate_um_per_year,
    guess_image_date,
    overlays_to_gif_bytes,
)
from flet_ui.state import GAAppState
from flet_ui import theme as T
from flet_ui.theme import (
    apply_page_theme,
    btn_draw,
    btn_erase,
    btn_ok,
    btn_outlined,
    btn_primary,
    btn_sidebar,
    btn_toolbar,
)

DISPLAY_WIDTH = 520
# 未読込時のプレースホルダ（Flet Image は src か src_base64 が必須）
EMPTY_PX = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


def _b64_jpeg(rgb: np.ndarray, quality: int = 85) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("画像エンコードに失敗しました")
    return base64.b64encode(buf).decode("ascii")


def _safe_text(text: str) -> str:
    """ブラウザで文字化けしやすい記号を ASCII 相当に置換。"""
    return (
        text.replace("\u2014", " - ")
        .replace("\u2013", "-")
        .replace("\u2026", "...")
        .replace("mm\u00b2", "mm2")
        .replace("mm²", "mm2")
        .replace("\u03bcm", "um")
        .replace("μm", "um")
    )


def resolve_file_dropdown_change(
    *,
    skip: bool,
    selected: str | None,
    series_results: list[dict] | None,
    files: list[Path],
    file_idx: int,
    series_launching: bool = False,
) -> tuple[str, int | None]:
    """Decide how the image-file dropdown should react.

    Returns (action, payload) where action is one of:
      - "ignore": do nothing (sync skip, empty, keep series, or launch guard)
      - "series_timepoint": reserved (unused; timepoints use series_date_dd)
      - "load_image": leave series (if any) and load files[payload]

    Critical: while series_results is set OR series is launching, never return
    "load_image" and never navigate via this dropdown. Spurious Flet on_change
    after 3/3 used to clear the series / jump to folder index 0 (often during
    page.update before series_results was assigned, or by selecting the earliest
    dated file which looks like “back to image 1”).
    """
    if skip or series_launching or not selected:
        return "ignore", None
    if series_results is not None:
        # Timepoint navigation is only via series_date_dd while series is open.
        return "ignore", None
    for i, f in enumerate(files):
        if f.name == selected:
            if i == file_idx:
                return "ignore", None
            return "load_image", i
    return "ignore", None


def resolve_series_landing_index(
    results: list[dict],
    *,
    preferred_filename: str | None,
) -> int:
    """Pick which timepoint to show after series build.

    Prefer the just-confirmed file so 3/3 does not visually snap back to image 1
    (earliest date / series index 0).
    """
    if not results:
        return 0
    if preferred_filename:
        for i, r in enumerate(results):
            if r.get("filename") == preferred_filename:
                return i
    return 0


def _card(label: str, value: str, color: str = T.TEXT) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                ft.Text(label, size=12, color=T.TEXT_MUTED),
                ft.Text(value, size=20, weight=ft.FontWeight.W_600, color=color),
            ],
            spacing=2,
        ),
        bgcolor=T.PANEL_BG,
        border=ft.border.all(1, "#374151"),
        border_radius=10,
        padding=ft.padding.symmetric(horizontal=16, vertical=12),
    )


def _gawain_card(stats: dict | None) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                ft.Text(GAWAIN_LABEL, size=11, color=T.TEXT_MUTED),
                ft.Text(
                    format_gawain_display(stats.get("gawain") if stats else None),
                    size=20,
                    weight=ft.FontWeight.W_600,
                    color=T.ACCENT_AMBER,
                ),
            ],
            spacing=2,
        ),
        bgcolor=T.PANEL_BG,
        border=ft.border.all(1, "#374151"),
        border_radius=10,
        padding=ft.padding.symmetric(horizontal=16, vertical=12),
    )


def _ga_area_card(stats: dict | None) -> ft.Container:
    """Raw mm² and square-root–transformed mm shown side-by-side (cohort baseline style)."""
    if not stats:
        return _card("GA Area", "-", T.ACCENT_RED)
    return ft.Container(
        content=ft.Column(
            [
                ft.Text("GA Area", size=12, color=T.TEXT_MUTED),
                ft.Row(
                    [
                        ft.Text(
                            f"{stats['area_mm2']} mm2",
                            size=20,
                            weight=ft.FontWeight.W_600,
                            color=T.ACCENT_RED,
                        ),
                        ft.Text(
                            f"{stats['area_mm_sqrt']} mm [SQRT]",
                            size=20,
                            weight=ft.FontWeight.W_600,
                            color=T.ACCENT_RED,
                        ),
                    ],
                    spacing=16,
                    wrap=True,
                ),
            ],
            spacing=2,
        ),
        bgcolor=T.PANEL_BG,
        border=ft.border.all(1, "#374151"),
        border_radius=10,
        padding=ft.padding.symmetric(horizontal=16, vertical=12),
    )


def _growth_card(mm2_per_year: float | None, mm_sqrt_per_year: float | None) -> ft.Container:
    """Raw mm²/year and square-root–transformed mm/year shown side-by-side."""
    if mm2_per_year is None and mm_sqrt_per_year is None:
        return _card("GA Growth", "-", T.ACCENT_AMBER)
    parts: list[ft.Control] = []
    if mm2_per_year is not None:
        parts.append(ft.Text(
            f"{mm2_per_year} mm2/year",
            size=20,
            weight=ft.FontWeight.W_600,
            color=T.ACCENT_AMBER,
        ))
    if mm_sqrt_per_year is not None:
        parts.append(ft.Text(
            f"{mm_sqrt_per_year} mm/year [SQRT]",
            size=20,
            weight=ft.FontWeight.W_600,
            color=T.ACCENT_AMBER,
        ))
    return ft.Container(
        content=ft.Column(
            [
                ft.Text("GA Growth", size=12, color=T.TEXT_MUTED),
                ft.Row(parts, spacing=16, wrap=True),
            ],
            spacing=2,
        ),
        bgcolor=T.PANEL_BG,
        border=ft.border.all(1, "#374151"),
        border_radius=10,
        padding=ft.padding.symmetric(horizontal=16, vertical=12),
    )


async def build_main_view(page: ft.Page, state: GAAppState) -> None:
    status = ft.Text("", color=T.ACCENT_AMBER, size=13)
    hint = ft.Text("", color=T.TEXT_MUTED, size=13)
    img_ctrl = ft.Image(
        src_base64=EMPTY_PX, fit=ft.ImageFit.CONTAIN, width=DISPLAY_WIDTH, height=200,
    )
    results_col = ft.Column(spacing=8, expand=True, scroll=ft.ScrollMode.AUTO)
    file_dd = ft.Dropdown(label="画像選択", options=[], width=320, text_size=13)
    input_field = ft.TextField(
        label="入力フォルダ", expand=True, value=state.input_dir or str(Path.home() / "Downloads"),
    )
    output_field = ft.TextField(
        label="出力フォルダ", expand=True,
        value=state.output_dir or str(Path.home() / "Downloads" / "ga_output"),
    )
    age_field = ft.TextField(label="年齢", value=str(state.age), width=100, keyboard_type=ft.KeyboardType.NUMBER)
    sex_dd = ft.Dropdown(
        label="性別", width=120,
        options=[ft.dropdown.Option("Male"), ft.dropdown.Option("Female")],
        value=state.sex,
    )
    disc_field = ft.TextField(
        label="乳頭径 (μm)", value=str(int(state.disc_diameter_um)), width=140,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    field_width_field = ft.TextField(
        label="画像幅 (mm)", value="", width=180,
        keyboard_type=ft.KeyboardType.NUMBER,
        hint_text="例: 55",
    )
    image_width_label = ft.Text("", size=11, color=T.TEXT_MUTED)
    scale_info = ft.Text("", size=11, color=T.TEXT_MUTED)
    disc_info = ft.Text("", size=11, color=T.ACCENT_AMBER)
    modality_dd = ft.Dropdown(
        label="画像モダリティ",
        width=320,
        options=[
            ft.dropdown.Option(key="optos", text="Optos"),
            ft.dropdown.Option(key="hra", text="HRA (Spectralis)"),
        ],
        value=state.image_modality,
    )
    prog_field = ft.TextField(
        label="進行速度 (μm/年) — 時系列解析で自動算出", value="", width=220,
        keyboard_type=ft.KeyboardType.NUMBER,
        hint_text="④ 時系列で自動算出",
        read_only=True,
    )
    view_mode = ft.RadioGroup(
        content=ft.Row([
            ft.Radio(value="work", label="作業", label_style=ft.TextStyle(color=T.TEXT)),
            ft.Radio(value="gif", label="GIF", label_style=ft.TextStyle(color=T.TEXT)),
            ft.Radio(value="graph", label="グラフ", label_style=ft.TextStyle(color=T.TEXT)),
        ]),
        value=state.view_mode,
        visible=False,
    )
    series_date_dd = ft.Dropdown(label="撮影日", width=360, visible=False)
    series_dates_col = ft.Column(spacing=6)
    series_dates_section = ft.Column(
        [
            ft.Text("時系列（撮影日）", size=12, weight=ft.FontWeight.W_600, color=T.PRIMARY),
            ft.Text(
                "EXIF → ファイル名 → 更新日時の順で自動推定。誤りがあれば修正してください。",
                size=10,
                color=T.TEXT_MUTED,
            ),
            series_dates_col,
        ],
        spacing=6,
        visible=False,
    )
    date_picker_target: dict[str, str | None] = {"fname": None}
    date_picker = ft.DatePicker(
        first_date=datetime(1990, 1, 1),
        last_date=datetime(2035, 12, 31),
    )
    foveal_strength_slider = ft.Slider(
        min=0, max=100, value=50, divisions=20,
        label="{value}%",
        width=280,
    )
    high_strength_slider = ft.Slider(
        min=80, max=100, value=85, divisions=4,
        label="{value}%",
        width=220,
    )
    use_clahe_sw = ft.Switch(label="CLAHE", value=True)
    clahe_clip_dd = ft.Dropdown(
        label="CLAHE clip",
        width=120,
        value="2.0",
        options=[
            ft.dropdown.Option(key="2.0", text="2.0"),
            ft.dropdown.Option(key="4.0", text="4.0"),
            ft.dropdown.Option(key="6.0", text="6.0"),
        ],
        text_size=12,
    )
    gif_img = ft.Image(visible=False, fit=ft.ImageFit.CONTAIN, width=DISPLAY_WIDTH)
    graph_area = ft.Column(visible=False)

    disp_scale = {"v": 1.0}
    disp_size = {"w": DISPLAY_WIDTH, "h": 200}
    drag = {"active": False, "start": (0.0, 0.0)}
    edit_state = {
        "pressing": False,
        "preview_mask": None,
        "changed": False,
        "undo_stack": [],
        "session_base": None,
    }
    batch_progress = ft.Text("", size=12, color=T.TEXT_MUTED, visible=False)
    # Dropdown sync: generation counter avoids asyncio.sleep(0) races where Flet
    # delivers on_change after skip was already cleared.
    dd_sync = {"skip": False, "gen": 0}
    # Blocks load_image while series is being built (covers page.update races
    # before series_results is assigned).
    series_guard = {"launching": False, "gen": 0}
    seg_correction_stats: dict | None = None

    def _set_status(msg: str) -> None:
        status.value = _safe_text(msg)

    def _batch_saved_count() -> int:
        if not state.files:
            return 0
        return sum(1 for f in state.files if f.name in state.per_file_results)

    def _batch_position_label() -> str:
        """現在の画像番号（1始まり）/ 総枚数。"""
        n = len(state.files)
        if n == 0:
            return "0/0"
        return f"{state.file_idx + 1}/{n}"

    def _batch_progress_label() -> str:
        n = len(state.files)
        saved = _batch_saved_count()
        return _safe_text(f"画像 {_batch_position_label()} · 確定済み {saved}/{n}")

    def _prune_stale_batch_results() -> None:
        names = {f.name for f in state.files}
        for stale in [k for k in state.per_file_results if k not in names]:
            del state.per_file_results[stale]

    def _begin_dropdown_sync() -> int:
        """Start ignoring programmatic dropdown changes; return a generation token."""
        dd_sync["gen"] += 1
        dd_sync["skip"] = True
        return dd_sync["gen"]

    def _release_dropdown_sync(gen: int, *, hold_s: float = 0.35) -> None:
        """Release skip only if no newer sync started (avoids sleep(0) races)."""

        async def _release() -> None:
            # Hold long enough for deferred Flet web on_change after page.update.
            await asyncio.sleep(0)
            await asyncio.sleep(hold_s)
            if dd_sync["gen"] == gen:
                dd_sync["skip"] = False

        page.run_task(_release)

    def _begin_series_launch_guard() -> int:
        series_guard["gen"] += 1
        series_guard["launching"] = True
        # Also arm dropdown skip for the whole build so mid-build updates cannot
        # load_image_at(0) before series_results exists.
        return _begin_dropdown_sync()

    def _end_series_launch_guard(dd_gen: int) -> None:
        series_guard["launching"] = False
        _release_dropdown_sync(dd_gen, hold_s=0.6)

    def _refresh_file_dropdown() -> None:
        if not state.files:
            file_dd.options = []
            return
        # Never rebuild the file dropdown while series is showing: replacing
        # options/value fires Flet on_change and used to clear series + load image 0.
        if state.series_results is not None:
            _refresh_file_dropdown_labels()
            return
        gen = _begin_dropdown_sync()
        file_dd.options = [
            ft.dropdown.Option(
                key=f.name,
                text=(
                    f"[済] {f.name}"
                    if f.name in state.per_file_results
                    else f.name
                ),
            )
            for f in state.files
        ]
        file_dd.value = state.files[state.file_idx].name
        _release_dropdown_sync(gen)

    def _refresh_file_dropdown_labels() -> None:
        """[済] ラベルのみ更新する（value は変更しない）。"""
        if not state.files:
            return
        if len(file_dd.options) != len(state.files):
            # Options missing/desynced: rebuild under sync guard without exiting series.
            gen = _begin_dropdown_sync()
            file_dd.options = [
                ft.dropdown.Option(
                    key=f.name,
                    text=(
                        f"[済] {f.name}"
                        if f.name in state.per_file_results
                        else f.name
                    ),
                )
                for f in state.files
            ]
            # While series is active / launching, leave value alone so Flet cannot
            # fire a deferred on_change that reloads folder index 0.
            if state.series_results is None and not series_guard["launching"] and state.files:
                file_dd.value = state.files[state.file_idx].name
            _release_dropdown_sync(gen)
            return
        for opt, f in zip(file_dd.options, state.files):
            opt.text = f"[済] {f.name}" if f.name in state.per_file_results else f.name

    def _sync_file_idx_to_series_timepoint(*, update_dropdown: bool = False) -> None:
        """時系列表示中: file_idx を現在の時点のファイルに合わせる。

        By default do NOT mutate file_dd.value — changing the dropdown value is what
        previously raced with on_file_change and wiped series_results after 3/3.
        """
        if not state.series_results or not state.files:
            return
        idx = min(state.series_time_idx, len(state.series_results) - 1)
        fname = state.series_results[idx]["filename"]
        for i, f in enumerate(state.files):
            if f.name == fname:
                state.file_idx = i
                if update_dropdown and file_dd.value != fname:
                    gen = _begin_dropdown_sync()
                    file_dd.value = fname
                    _release_dropdown_sync(gen)
                break

    def _refresh_seg_ok_label() -> None:
        seg_ok_btn.text = "✓ 確定" if state.batch_mode else "OK"

    def _ensure_output_dir() -> Path:
        out = Path(output_field.value.strip() or state.output_dir or ".")
        out.mkdir(parents=True, exist_ok=True)
        state.output_dir = str(out)
        return out

    def _clear_series_display() -> None:
        state.series_results = None
        state.series_overlays = None
        state.series_gif = None
        state.estimated_prog = None
        state.estimated_growth_mm2_per_year = None
        state.estimated_growth_mm_sqrt_per_year = None
        state.series_time_idx = 0
        state.view_mode = "work"
        view_mode.value = "work"
        view_mode.visible = False
        series_date_dd.visible = False
        prog_field.value = ""

    def _invalidate_saved_results() -> None:
        """スケール変更などで確定済み結果が無効になったとき。"""
        if not state.per_file_results and not state.series_results:
            return
        state.per_file_results = {}
        _clear_series_display()
        _update_batch_ui()

    def _update_batch_ui(*, refresh_dropdown: bool = True) -> None:
        n = len(state.files)
        saved = _batch_saved_count()
        state.batch_mode = n >= 2
        batch_progress.visible = state.batch_mode
        ready = saved >= 2
        if state.batch_mode:
            batch_progress.value = _batch_progress_label()
        series_btn.visible = ready
        series_launch_row.visible = ready
        if ready:
            label = _safe_text(f"時系列で表示 ({saved}/{n})")
            series_launch_btn.text = label
            series_btn.text = label
        _refresh_seg_ok_label()
        if refresh_dropdown:
            _refresh_file_dropdown()

    def _snapshot_current() -> dict:
        path = state.files[state.file_idx]
        return {
            "path": path,
            "filename": path.name,
            "date": state.series_dates.get(path.name, guess_image_date(path).isoformat()),
            "image": state.img_gray.copy(),
            "mask": state.mask.copy(),
            "stats": copy.deepcopy(state.stats),
            "fovea_xy": state.fovea_xy,
            "scale_um_px": state.scale_um_px,
        }

    def _apply_series_results(
        results: list[dict],
        *,
        landing_filename: str | None = None,
    ) -> None:
        overlays = []
        for row in results:
            s = row["stats"]
            fxy = row.get("fovea_xy") or state.fovea_xy
            scale = row.get("scale_um_px") or state.scale_um_px
            overlays.append(make_overlay(
                row["image"], row["mask"],
                fovea_xy=fxy,
                nearest_xy=s.get("nearest_xy"),
                disc_pts=state.disc_pts,
                disc_diameter_um=state.disc_diameter_um,
                show_macula=True,
                scale_um_px=scale,
            ))
        state.series_results = results
        state.series_overlays = overlays
        state.series_gif = overlays_to_gif_bytes(overlays)
        state.estimated_prog = estimate_progression_rate_um_per_year(results)
        state.estimated_growth_mm2_per_year = estimate_growth_mm2_per_year(results)
        state.estimated_growth_mm_sqrt_per_year = estimate_growth_mm_sqrt_per_year(results)
        prog_field.value = str(int(state.estimated_prog)) if state.estimated_prog else ""
        view_mode.visible = True
        series_date_dd.visible = True
        series_date_dd.options = [
            ft.dropdown.Option(
                key=str(i),
                text=_safe_text(f"{r['date']} - {r['filename']}"),
            )
            for i, r in enumerate(results)
        ]
        land = resolve_series_landing_index(results, preferred_filename=landing_filename)
        series_date_dd.value = str(land)
        state.series_time_idx = land
        # Keep file_idx aligned for status labels, but never mutate file_dd here —
        # that mutation was racing with on_file_change after 3/3 auto-launch.
        _sync_file_idx_to_series_timepoint(update_dropdown=False)

    def _next_batch_idx(after: int) -> int | None:
        """確定後: 現在の次の画像インデックス（順番に進む）。"""
        nxt = after + 1
        if nxt < len(state.files):
            return nxt
        return None

    def _first_unsaved_idx() -> int | None:
        for i, f in enumerate(state.files):
            if f.name not in state.per_file_results:
                return i
        return None

    def init_series_dates(files: list[Path]) -> None:
        """既存の手動編集日付を保持しつつ、新規ファイルのみ自動推定。"""
        dates: dict[str, str] = {}
        for f in files:
            key = f.name
            if key in state.series_dates:
                dates[key] = state.series_dates[key]
            else:
                dates[key] = guess_image_date(f).isoformat()
        state.series_dates = dates

    def _invalidate_series_after_date_edit() -> None:
        if not state.series_results:
            return
        _clear_series_display()
        status.value = "撮影日を変更しました — ④ 時系列を再実行してください"

    def _apply_series_date(fname: str, picked: date) -> None:
        state.series_dates[fname] = picked.isoformat()
        _invalidate_series_after_date_edit()
        refresh_series_dates_ui()

    def _on_date_picker_change(e: ft.ControlEvent) -> None:
        fname = date_picker_target["fname"]
        val = e.control.value
        if fname and val is not None:
            picked = val.date() if isinstance(val, datetime) else val
            _apply_series_date(fname, picked)
        date_picker_target["fname"] = None
        page.update()

    def _open_date_picker(fname: str) -> None:
        date_picker_target["fname"] = fname
        iso = state.series_dates.get(fname)
        if iso:
            date_picker.value = datetime.fromisoformat(iso).date()
        page.open(date_picker)

    def _bind_date_field_submit(fname: str):
        def _handler(e: ft.ControlEvent) -> None:
            raw = (e.control.value or "").strip()
            try:
                picked = datetime.strptime(raw, "%Y-%m-%d").date()
            except ValueError:
                e.control.error_text = "YYYY-MM-DD 形式"
                page.update()
                return
            e.control.error_text = None
            _apply_series_date(fname, picked)
            page.update()
        return _handler

    def _bind_date_picker_open(fname: str):
        def _handler(_e: ft.ControlEvent) -> None:
            _open_date_picker(fname)
        return _handler

    def refresh_series_dates_ui() -> None:
        series_dates_col.controls.clear()
        show = len(state.files) >= 2
        series_dates_section.visible = show
        if not show:
            return
        init_series_dates(state.files)
        for f in state.files:
            fname = f.name
            iso = state.series_dates[fname]
            series_dates_col.controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                fname,
                                size=10,
                                color=T.TEXT_MUTED,
                                max_lines=2,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Row(
                                [
                                    ft.TextField(
                                        value=iso,
                                        width=130,
                                        text_size=12,
                                        hint_text="YYYY-MM-DD",
                                        on_submit=_bind_date_field_submit(fname),
                                    ),
                                    ft.IconButton(
                                        icon=ft.Icons.CALENDAR_MONTH,
                                        icon_size=18,
                                        tooltip="カレンダー",
                                        on_click=_bind_date_picker_open(fname),
                                    ),
                                ],
                                spacing=0,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                        ],
                        spacing=2,
                        tight=True,
                    ),
                    padding=ft.padding.only(bottom=4),
                )
            )

    date_picker.on_change = _on_date_picker_change

    def _refresh_scale_info() -> None:
        if state.scale_um_px is None:
            scale_info.value = ""
            return
        src = "乳頭" if state.calibration_source == "disc" else "幅指定"
        scale_info.value = _safe_text(f"スケール ({src}): {state.scale_um_px:.2f} um/px")

    def _update_disc_info() -> None:
        pts = state.disc_draft_pts if state.mode in ("disc_line", "disc_confirm") else state.disc_pts
        if len(pts) != 2:
            disc_info.value = ""
            return
        p1, p2 = pts
        d_px = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        try:
            disc_um = float(disc_field.value or state.disc_diameter_um)
        except ValueError:
            disc_um = state.disc_diameter_um
        if d_px > 0:
            preview = disc_um / d_px
            disc_info.value = _safe_text(
                f"乳頭計測: {d_px:.0f} px = {disc_um:.0f} um ({preview:.2f} um/px)"
            )
        else:
            disc_info.value = ""

    def _refresh_image_width_label() -> None:
        if state.img_gray is None:
            image_width_label.value = ""
            return
        _, w = state.img_gray.shape
        image_width_label.value = f"現在の画像: {w} px（水平幅）"

    def _to_img_xy(local_x: float, local_y: float) -> tuple[float, float]:
        h, w = state.img_gray.shape
        ox = max(0.0, min(w - 1, local_x / disp_scale["v"]))
        oy = max(0.0, min(h - 1, local_y / disp_scale["v"]))
        return ox, oy

    def _display_mask() -> np.ndarray:
        if edit_state["preview_mask"] is not None:
            return edit_state["preview_mask"]
        if state.mask is not None:
            return state.mask
        return np.zeros_like(state.img_gray) if state.img_gray is not None else np.array([])

    def _mask_style() -> str:
        if state.mode in ("seg_review", "seg_erase", "seg_draw", "seg_clear", "seg_clear_confirm"):
            return "blend"
        return "solid"

    def _overlay_disc_pts() -> list[tuple[float, float]] | None:
        if state.mode in ("disc_line", "disc_confirm") and len(state.disc_draft_pts) == 2:
            return state.disc_draft_pts
        return state.disc_pts or None

    def _try_apply_width_scale() -> bool:
        """HRA 幅指定フィールドから scale_um_px を算出（未設定時）。"""
        if state.scale_um_px is not None:
            return True
        if state.img_gray is None:
            return False
        raw = (field_width_field.value or "").strip()
        if not raw and state.field_width_mm is not None:
            raw = str(state.field_width_mm)
        if not raw:
            return False
        try:
            field_mm = float(raw)
            if field_mm <= 0:
                return False
        except ValueError:
            return False
        _, w_px = state.img_gray.shape
        try:
            state.scale_um_px = calibration.compute_scale_um_per_px_from_field_width(w_px, field_mm)
        except ValueError:
            return False
        state.calibration_source = "width"
        state.field_width_mm = field_mm
        _refresh_scale_info()
        return True

    def _foveal_prereq_error() -> str | None:
        if state.fovea_xy is None:
            return "中心窩が未指定です"
        raw = state.img_gray_raw if state.img_gray_raw is not None else state.img_gray
        if raw is None:
            return "画像が読み込まれていません"
        if state.scale_um_px is None and not _try_apply_width_scale():
            if modality_dd.value == "hra":
                return "先に「①' 幅指定」で画像幅 (mm) を設定してください"
            return "先にスケール較正（乳頭計測または幅指定）を完了してください"
        return None

    def _foveal_strength() -> float:
        return float(foveal_strength_slider.value or 0) / 100.0

    def _uncorrected_gray() -> np.ndarray | None:
        """補正前の原画像（中心窩タップ時に固定。補正後で上書きしない）。"""
        if state.img_gray_raw is not None:
            return state.img_gray_raw
        return state.img_gray

    def _restore_display_gray() -> None:
        """表示用画像を補正前の原画像に戻す（セグメント確認キャンセル時）。"""
        raw = _uncorrected_gray()
        if raw is not None:
            state.img_gray = raw.copy()

    def _enter_foveal_confirm_preserving_fovea() -> None:
        """中心窩座標を維持したまま補正確認パネルへ進める。"""
        if state.fovea_xy is None:
            return
        state.foveal_correction_applied = False
        state.foveal_correction_skipped = False
        state.mode = "foveal_confirm"
        foveal_strength_slider.value = int(state.foveal_correction_strength * 100)

    def _corrected_gray_preview() -> np.ndarray | None:
        err = _foveal_prereq_error()
        if err:
            return None
        raw = _uncorrected_gray()
        if raw is None or state.fovea_xy is None or state.scale_um_px is None:
            return None
        return apply_foveal_block_correction(
            raw, state.fovea_xy, state.scale_um_px, _foveal_strength(),
        )

    def _gray_for_segmentation() -> np.ndarray | None:
        """セグメンテーション入力: 原画像 + 保存済み補正設定から毎回算出。"""
        working = state.img_gray
        if working is None:
            return None
        original = _uncorrected_gray()
        if original is None:
            return None
        return gray_for_segmentation(
            working,
            original,
            state.fovea_xy,
            state.scale_um_px,
            correction_applied=state.foveal_correction_applied,
            correction_skipped=state.foveal_correction_skipped,
            correction_strength=state.foveal_correction_strength,
        )

    def _compute_seg_correction_stats(seg_gray: np.ndarray) -> dict | None:
        raw = _uncorrected_gray()
        if (
            raw is None
            or state.fovea_xy is None
            or state.scale_um_px is None
            or not state.foveal_correction_applied
            or state.foveal_correction_skipped
            or state.foveal_correction_strength <= 0
        ):
            return None
        mask_raw = segmentation.segment_ga(
            raw, use_clahe=state.use_clahe, clahe_clip=state.clahe_clip,
        )
        mask_corr = segmentation.segment_ga(
            seg_gray, use_clahe=state.use_clahe, clahe_clip=state.clahe_clip,
        )
        return correction_effect_stats(
            raw, seg_gray, state.fovea_xy, state.scale_um_px,
            mask_raw=mask_raw, mask_corr=mask_corr,
        )

    def _segmentation_source_label() -> str:
        if (
            state.foveal_correction_applied
            and state.foveal_correction_strength > 0
            and not state.foveal_correction_skipped
            and state.fovea_xy is not None
        ):
            pct = int(state.foveal_correction_strength * 100)
            parts = [f"補正 ({pct}%)"]
            if seg_correction_stats:
                arp = seg_correction_stats.get("area_reduction_pct")
                if arp is not None:
                    parts.append(f"面積 {arp:+.1f}%")
                cl = seg_correction_stats.get("central_lift_mean", 0.0)
                if cl > 0.5:
                    parts.append(f"中心 +{cl:.1f}")
                elif not seg_correction_stats.get("mask_effective"):
                    parts.append("輝度変化小")
                if not seg_correction_stats.get("effective"):
                    parts.append("効果小")
            return " - ".join(parts)
        return "原画像 (中心窩補正なし)"

    def _run_segmentation(seg_gray: np.ndarray) -> np.ndarray:
        return segmentation.segment_ga(
            seg_gray,
            use_clahe=state.use_clahe,
            clahe_clip=state.clahe_clip,
        )

    def _overlay_base_gray() -> np.ndarray | None:
        """オーバーレイ/Erase/Draw と同じグレー画像を返す。"""
        if state.mask is not None or state.mode in ("seg_review", "seg_erase", "seg_draw"):
            seg = _gray_for_segmentation()
            if seg is not None:
                return seg
        return state.img_gray

    def _apply_pending_foveal_correction() -> bool:
        """中心窩補正パネル中のスライダー設定を反映（原画像は保持）。"""
        err = _foveal_prereq_error()
        if err:
            _set_status(err)
            return False
        raw = _uncorrected_gray()
        if raw is None:
            _set_status("画像が読み込まれていません")
            return False
        strength = _foveal_strength()
        state.img_gray = apply_foveal_block_correction(
            raw, state.fovea_xy, state.scale_um_px, strength,
        )
        state.foveal_correction_applied = strength > 0
        state.foveal_correction_skipped = False
        state.foveal_correction_strength = strength
        return True

    def _draw_fovea_crosshair(rgb: np.ndarray) -> np.ndarray:
        if state.fovea_xy is None:
            return rgb
        fx, fy = int(state.fovea_xy[0]), int(state.fovea_xy[1])
        r = 8
        cyan = (34, 211, 238)
        cv2.circle(rgb, (fx, fy), r, cyan, 1)
        cv2.line(rgb, (fx - 18, fy), (fx - r, fy), cyan, 1)
        cv2.line(rgb, (fx + r, fy), (fx + 18, fy), cyan, 1)
        cv2.line(rgb, (fx, fy - 18), (fx, fy - r), cyan, 1)
        cv2.line(rgb, (fx, fy + r), (fx, fy + 18), cyan, 1)
        cv2.circle(rgb, (fx, fy), 2, cyan, -1)
        return rgb

    def _build_rgb_frame(self_mask: np.ndarray | None = None) -> np.ndarray:
        """表示用 RGB フレームを生成（erase プレビュー用に mask を差し替え可能）。"""
        if state.mode == "foveal_confirm":
            corrected = _corrected_gray_preview()
            if corrected is not None:
                return _draw_fovea_crosshair(cv2.cvtColor(corrected, cv2.COLOR_GRAY2RGB))

        mask = self_mask if self_mask is not None else _display_mask()
        stats = state.stats
        base_gray = _overlay_base_gray()
        if base_gray is None:
            base_gray = state.img_gray
        return make_overlay(
            base_gray,
            mask,
            fovea_xy=state.fovea_xy,
            nearest_xy=stats["nearest_xy"] if stats else None,
            disc_pts=_overlay_disc_pts(),
            disc_diameter_um=state.disc_diameter_um,
            crop_pts=(
                state.clear_pts if state.mode in ("seg_clear", "seg_clear_confirm")
                else (state.crop_pts if state.mode in ("crop_drag", "crop_confirm") else None)
            ),
            show_macula=state.stats is not None and state.mode not in (
                "seg_review", "seg_erase", "seg_draw", "seg_clear", "seg_clear_confirm",
            ),
            scale_um_px=state.scale_um_px,
            mask_style=_mask_style(),
        )

    def _push_image_rgb(rgb: np.ndarray, *, do_update: bool = True, jpeg_quality: int = 85) -> None:
        h, w = rgb.shape[:2]
        disp_scale["v"] = DISPLAY_WIDTH / w
        disp_h = int(h * disp_scale["v"])
        disp = cv2.resize(rgb, (DISPLAY_WIDTH, disp_h))
        disp_size["h"] = disp_h
        img_ctrl.width = DISPLAY_WIDTH
        img_ctrl.height = disp_h
        gesture_stack.height = disp_h
        img_ctrl.src_base64 = _b64_jpeg(disp, quality=jpeg_quality)
        if do_update:
            page.update()

    def load_image_at(idx: int, *, new_folder: bool = False) -> None:
        if not state.files:
            return
        # Never replace the working image while a series view is active or
        # mid-launch — that is how 3/3 confirmation snapped back to image 1.
        if state.series_results is not None or series_guard["launching"]:
            return
        state.file_idx = max(0, min(idx, len(state.files) - 1))
        path = state.files[state.file_idx]
        gray = load_gray(path)
        state.img_full = gray
        state.img_gray = gray.copy()
        state.crop_box = None
        state.crop_pts = []
        if new_folder:
            state.reset_analysis(keep_series=True)
        else:
            state.reset_per_image()
        _refresh_image_width_label()
        _refresh_scale_info()
        _refresh_file_dropdown()
        refresh_image()
        refresh_results()
        _update_batch_ui()

    def _save_batch_result() -> None:
        """バッチモード: 現在画像の確定結果をメモリへ保存（画面は現在画像のまま）。"""
        if state.stats is None or state.mask is None or not state.files:
            return
        fname = state.files[state.file_idx].name
        state.per_file_results[fname] = _snapshot_current()
        _prune_stale_batch_results()
        _clear_series_display()

    def on_next_image(_=None) -> None:
        """確定後、次の画像（なければ最初の未解析）へ進む。"""
        if not state.files:
            return
        nxt = _next_batch_idx(state.file_idx)
        if nxt is None:
            nxt = _first_unsaved_idx()
        if nxt is None:
            saved = _batch_saved_count()
            _set_status(f"全画像を確認済み ({saved}/{len(state.files)})")
            page.update()
            return
        load_image_at(nxt)
        saved = _batch_saved_count()
        n = len(state.files)
        _set_status(
            f"画像 {nxt + 1}/{n} · 確定済み {saved}/{n}"
            f" — {state.files[nxt].name}"
        )
        _update_batch_ui()
        page.update()

    def on_show_saved(_=None) -> None:
        if not state.files:
            return
        fname = state.files[state.file_idx].name
        snap = state.per_file_results.get(fname)
        if not snap:
            status.value = "この画像の確定結果がありません"
            page.update()
            return
        _clear_series_display()
        state.img_gray = snap["image"].copy()
        state.mask = snap["mask"].copy()
        state.stats = copy.deepcopy(snap["stats"])
        state.fovea_xy = snap.get("fovea_xy")
        if snap.get("scale_um_px") is not None:
            state.scale_um_px = snap["scale_um_px"]
        state.mode = None
        status.value = f"確定済み結果を表示: {fname}"
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    def on_export_csv(_=None) -> None:
        if not state.series_results:
            status.value = "先に④ 時系列を実行してください"
            page.update()
            return
        out = _ensure_output_dir()
        path = out / suggested_csv_filename()
        path.write_bytes(build_batch_csv_bytes(state.series_results))
        status.value = f"CSV 保存: {path}"
        page.update()

    def on_export_pdf(_=None) -> None:
        if state.series_results and state.series_overlays:
            row = state.series_results[state.series_time_idx]
            overlay = state.series_overlays[state.series_time_idx]
            stats = row["stats"]
            fname = row["filename"]
            date_str = row["date"]
        elif state.stats is not None and state.mask is not None and state.img_gray is not None:
            overlay = _build_rgb_frame()
            stats = state.stats
            fname = state.files[state.file_idx].name if state.files else "image"
            date_str = state.series_dates.get(fname, "")
        else:
            status.value = "保存する解析結果がありません"
            page.update()
            return
        out = _ensure_output_dir()
        path = out / suggested_pdf_filename(fname)
        generate_pdf_report(
            filename=fname,
            date=date_str,
            stats=stats,
            overlay_rgb=overlay,
            output_path=path,
        )
        status.value = f"PDF 保存: {path}"
        page.update()

    def on_reanalyze_timepoint(_=None) -> None:
        if not state.series_results:
            return
        row = state.series_results[state.series_time_idx]
        fname = row["filename"]
        state.per_file_results.pop(fname, None)
        _clear_series_display()
        for i, f in enumerate(state.files):
            if f.name == fname:
                load_image_at(i)
                break
        status.value = f"{fname} — 再解析してください（③ 解析）"
        on_analyze()
        _update_batch_ui()

    def refresh_image(*, do_update: bool = True) -> None:
        if state.view_mode == "gif" and state.series_gif:
            gif_img.src_base64 = base64.b64encode(state.series_gif).decode("ascii")
            img_ctrl.visible = False
            gif_img.visible = True
            graph_area.visible = False
            return
        if state.view_mode == "graph" and state.series_results:
            img_ctrl.visible = False
            gif_img.visible = False
            graph_area.visible = True
            _refresh_graph()
            return

        img_ctrl.visible = True
        gif_img.visible = False
        graph_area.visible = False

        if state.series_results and state.view_mode == "work":
            rgb = state.series_overlays[state.series_time_idx]
        elif state.img_gray is not None:
            rgb = _build_rgb_frame()
        else:
            img_ctrl.src_base64 = EMPTY_PX
            return

        _push_image_rgb(rgb, do_update=do_update)

    def _refresh_graph() -> None:
        graph_area.controls.clear()
        if not state.series_results:
            return
        rows = []
        for r in state.series_results:
            s = r["stats"]
            rows.append(
                ft.DataRow(cells=[
                    ft.DataCell(ft.Text(r["date"])),
                    ft.DataCell(ft.Text(f"{s['area_mm2']}")),
                    ft.DataCell(ft.Text(f"{s['area_mm_sqrt']}")),
                    ft.DataCell(ft.Text(format_gawain_display(s["gawain"]))),
                    ft.DataCell(ft.Text(
                        f"{s['dist_um']:.0f}" if s["dist_um"] is not None else "-")),
                ])
            )
        graph_area.controls.append(
            ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Date")),
                    ft.DataColumn(ft.Text("GA mm2")),
                    ft.DataColumn(ft.Text("GA mm [SQRT]")),
                    ft.DataColumn(ft.Text(GAWAIN_LABEL, size=10)),
                    ft.DataColumn(ft.Text("Dist um")),
                ],
                rows=rows,
                border=ft.border.all(1, "#374151"),
            )
        )

    def refresh_results() -> None:
        results_col.controls.clear()
        stats = state.stats
        if state.series_results and state.view_mode == "work":
            stats = state.series_results[state.series_time_idx]["stats"]

        age = int(age_field.value or state.age)
        sex = sex_dd.value or state.sex
        life = format_life_expectancy(age, sex)
        life_y = life_expectancy_years(age, sex)
        dist = stats["dist_um"] if stats else None

        # 進行速度・Cross Point は時系列解析（④）でのみ算出・表示する。
        is_series = state.series_results is not None
        cross = None
        if is_series:
            try:
                prog = float(prog_field.value) if prog_field.value else state.estimated_prog
            except ValueError:
                prog = state.estimated_prog
            if dist is not None and prog and prog > 0:
                cross = round(dist / prog, 1)

        cards = [
            _ga_area_card(stats),
            _growth_card(
                state.estimated_growth_mm2_per_year if is_series else None,
                state.estimated_growth_mm_sqrt_per_year if is_series else None,
            ),
            _gawain_card(stats),
            _card("Dist to fovea", f"{dist:.0f} um" if dist is not None else "-", T.ACCENT_CYAN),
            _card("平均余命", life),
        ]
        if is_series:
            cards.append(
                _card("Cross Point", f"{cross} years" if cross is not None else "-", "#10B981"),
            )
        results_col.controls.extend(cards)
        if cross is not None:
            if cross < life_y:
                results_col.controls.append(ft.Text(
                    f"⚠ Cross Point ({cross}年) < 余命 ({life_y:.1f}年)",
                    color=T.ACCENT_AMBER, size=12,
                ))
            else:
                results_col.controls.append(ft.Text(
                    f"Cross Point ({cross}年) ≥ 余命 ({life_y:.1f}年)",
                    color="#10B981", size=12,
                ))
        if stats and stats.get("lesions"):
            results_col.controls.append(ft.Text("GA病変", weight=ft.FontWeight.BOLD, color=T.PRIMARY))
            for l in stats["lesions"]:
                results_col.controls.append(ft.Text(
                    f"#{l['id']}: {l['area_mm2']} mm2, dist {l['dist_um']} um",
                    size=12, color=T.TEXT_MUTED,
                ))
        if stats and state.view_mode == "work" and not state.series_results:
            results_col.controls.append(
                btn_toolbar("↻ 再解析", on_click=on_reanalyze),
            )
            results_col.controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("高強度再解析 (80-100%)", size=11, color=T.TEXT_MUTED),
                            high_strength_slider,
                            btn_toolbar("↻ 高強度で再解析", on_click=on_reanalyze_high_strength),
                        ],
                        spacing=4,
                        tight=True,
                    ),
                    bgcolor=T.PANEL_BG,
                    border=ft.border.all(1, "#374151"),
                    border_radius=8,
                    padding=8,
                )
            )
        fname = state.files[state.file_idx].name if state.files else ""
        if (
            state.batch_mode
            and fname
            and fname in state.per_file_results
            and not state.series_results
            and (_next_batch_idx(state.file_idx) is not None or _first_unsaved_idx() is not None)
        ):
            results_col.controls.append(
                btn_primary("次へ ▶", on_click=on_next_image),
            )
        if fname and fname in state.per_file_results:
            results_col.controls.append(
                btn_toolbar("個別結果", on_click=on_show_saved),
            )
        if state.series_results:
            results_col.controls.append(
                ft.Row(
                    [
                        btn_toolbar("CSV 保存", on_click=on_export_csv),
                        btn_toolbar("PDF 保存", on_click=on_export_pdf),
                    ],
                    spacing=8,
                    wrap=True,
                )
            )
            results_col.controls.append(
                btn_toolbar("↻ この時点を再解析", on_click=on_reanalyze_timepoint),
            )
        elif stats and state.mode is None and not state.series_results:
            results_col.controls.append(
                btn_toolbar("PDF 保存", on_click=on_export_pdf),
            )
        if fname and fname in state.per_file_results and state.batch_mode:
            results_col.controls.append(ft.Text(
                f"[済] {fname}",
                size=11,
                color=T.ACCENT_GREEN,
            ))
        if seg_correction_stats and state.foveal_correction_applied:
            results_col.controls.append(ft.Text(
                _safe_text(
                    f"セグメント入力: {_segmentation_source_label()}"
                ),
                size=11,
                color=(
                    T.TEXT_MUTED if seg_correction_stats.get("effective")
                    else T.ACCENT_AMBER
                ),
            ))
        elif state.fovea_xy and stats and state.mode in (None, "seg_review", "seg_erase", "seg_draw"):
            results_col.controls.append(ft.Text(
                "セグメント入力: 原画像 (中心窩補正なし)",
                size=11, color=T.ACCENT_AMBER,
            ))
        page.update()

    def refresh_hint() -> None:
        msgs = {
            "crop_drag": "クロップ: ドラッグで矩形を描画",
            "crop_confirm": "クロップ範囲を確認して OK または やり直し",
            "disc_line": "乳頭計測: 乳頭の両端を結ぶ直線をドラッグ",
            "disc_confirm": "直線を確認して OK または やり直し",
            "width_cal": "画像の水平方向の実長 (mm) を入力して OK",
            "fovea": "中心窩をタップ",
            "foveal_confirm": "黄斑色素ブロック補正 - 強度を調整して適用または補正しない（自動で解析へ）",
            "seg_review": "セグメンテーション確認 - Erase / Draw / 範囲消去で修正、OK で確定",
            "seg_erase": "長押しで削除 (緑の内外どちらでも可) - やり直しで取消",
            "seg_draw": "長押しで暗部を追加 (緑の内外どちらでも可) - やり直しで取消",
            "seg_clear": "範囲消去: フリーハンドで囲んで指定",
            "seg_clear_confirm": "範囲内のマスクを消去します - OK または やり直し",
        }
        hint.value = _safe_text(msgs.get(state.mode or "", ""))
        if not state.mode and modality_dd.value == "hra" and state.scale_um_px is None:
            hint.value = "HRA: 乳頭が見えない場合は「幅指定」でスケール設定"
        hint.color = T.PRIMARY if state.mode or hint.value else T.TEXT_MUTED

    def on_load_files(_=None) -> None:
        state.input_dir = input_field.value.strip()
        state.output_dir = output_field.value.strip()
        folder = Path(state.input_dir)
        state.files = list_images(folder)
        init_series_dates(state.files)
        refresh_series_dates_ui()
        if state.files:
            load_image_at(0, new_folder=True)
            _set_status(
                f"{len(state.files)} 件読み込み ({', '.join(sorted(SUPPORTED_SUFFIXES))})"
            )
        else:
            _set_status(f"画像なし: {folder}")
            state.img_gray = None
            refresh_image()
        _update_batch_ui()
        page.update()

    def _freehand_path_length(pts: list[tuple[float, float]]) -> float:
        if len(pts) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            total += (dx * dx + dy * dy) ** 0.5
        return total

    def _append_clear_point(p: tuple[float, float]) -> None:
        if not state.clear_pts:
            state.clear_pts.append(p)
            return
        lx, ly = state.clear_pts[-1]
        if (p[0] - lx) ** 2 + (p[1] - ly) ** 2 >= 4.0:
            state.clear_pts.append(p)

    def on_file_change(e) -> None:
        action, payload = resolve_file_dropdown_change(
            skip=dd_sync["skip"],
            selected=e.control.value,
            series_results=state.series_results,
            files=state.files,
            file_idx=state.file_idx,
            series_launching=series_guard["launching"],
        )
        if action != "load_image" or payload is None:
            return
        if state.series_results is not None or series_guard["launching"]:
            return
        i = payload
        load_image_at(i)
        if state.batch_mode:
            saved = _batch_saved_count()
            n = len(state.files)
            fname = state.files[i].name
            tag = "[済] " if fname in state.per_file_results else ""
            _set_status(
                f"画像 {i + 1}/{n} · 確定済み {saved}/{n} — {tag}{fname}"
            )
            page.update()

    file_dd.on_change = on_file_change

    def on_pan_start(e: ft.DragStartEvent) -> None:
        if state.img_gray is None or state.mode in ("seg_review", "seg_erase", "seg_draw"):
            return
        if state.mode == "disc_line":
            drag["active"] = True
            drag["start"] = (e.local_x, e.local_y)
            p = _to_img_xy(e.local_x, e.local_y)
            state.disc_draft_pts = [p, p]
            refresh_image(do_update=False)
        elif state.mode in ("crop_drag", "seg_clear"):
            drag["active"] = True
            drag["start"] = (e.local_x, e.local_y)
            p = _to_img_xy(e.local_x, e.local_y)
            if state.mode == "crop_drag":
                state.crop_pts = [p, p]
            else:
                state.clear_pts = [p]
            refresh_image(do_update=False)

    def on_pan_update(e: ft.DragUpdateEvent) -> None:
        if not drag["active"] or state.img_gray is None:
            return
        if state.mode == "disc_line":
            p1 = _to_img_xy(*drag["start"])
            p2 = _to_img_xy(e.local_x, e.local_y)
            state.disc_draft_pts = [p1, p2]
            _update_disc_info()
            refresh_image()
        elif state.mode == "crop_drag":
            p1 = _to_img_xy(*drag["start"])
            p2 = _to_img_xy(e.local_x, e.local_y)
            state.crop_pts = [p1, p2]
            refresh_image()
        elif state.mode == "seg_clear":
            _append_clear_point(_to_img_xy(e.local_x, e.local_y))
            refresh_image()

    def on_pan_end(_e: ft.DragEndEvent) -> None:
        if not drag["active"]:
            return
        drag["active"] = False
        if state.mode == "disc_line" and len(state.disc_draft_pts) == 2:
            p1, p2 = state.disc_draft_pts
            d_px = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
            if d_px >= 5:
                state.mode = "disc_confirm"
                status.value = f"乳頭径ライン: {d_px:.0f} px - OK で確定"
                _update_disc_info()
            else:
                state.disc_draft_pts = []
                status.value = "直線が短すぎます。もう一度ドラッグしてください"
        elif state.mode == "crop_drag" and len(state.crop_pts) == 2:
            p1, p2 = state.crop_pts
            w = abs(p2[0] - p1[0])
            h = abs(p2[1] - p1[1])
            if w >= 5 and h >= 5:
                state.mode = "crop_confirm"
                status.value = "クロップ範囲を確認して OK で確定"
            else:
                state.crop_pts = []
                status.value = "矩形が小さすぎます。もう一度ドラッグしてください"
        elif state.mode == "seg_clear" and len(state.clear_pts) >= 3:
            path_len = _freehand_path_length(state.clear_pts)
            if path_len >= 12.0:
                state.mode = "seg_clear_confirm"
                status.value = "範囲内のマスクを消去します - OK で実行"
            else:
                state.clear_pts = []
                status.value = "範囲が短すぎます。もう一度フリーハンドで囲んでください"
        update_confirm_visibility()
        refresh_hint()
        refresh_image()
        page.update()

    async def continuous_edit(ix: int, iy: int) -> None:
        """長押し中、パワー拡大に合わせてプレビューマスクを毎フレーム更新。"""
        a_power = 0.5
        is_erase = state.mode == "seg_erase"
        base_gray = _overlay_base_gray()
        while edit_state["pressing"]:
            if base_gray is None or state.mask is None:
                break
            if is_erase:
                preview, delta = erase_oversegmentation_from_mask(
                    base_gray, state.mask, (ix, iy), a_power,
                )
                action = "削除"
            else:
                preview, delta = expand_segmentation_from_mask(
                    base_gray, state.mask, (ix, iy), a_power,
                )
                action = "追加"
            edit_state["preview_mask"] = preview
            if delta > 0:
                edit_state["changed"] = True
            rgb = _build_rgb_frame(preview)
            _push_image_rgb(rgb, do_update=True, jpeg_quality=72)
            status.value = (
                f"{'Erase' if is_erase else 'Draw'}中… パワー {a_power:.1f}（{action} {delta}px）"
                if delta else f"{'Erase' if is_erase else 'Draw'}中… パワー {a_power:.1f}（拡大中）"
            )
            a_power = min(6.0, a_power + 0.2)
            await asyncio.sleep(0.04)

    def _commit_edit_stroke() -> None:
        if edit_state["changed"] and edit_state["preview_mask"] is not None:
            edit_state["undo_stack"].append(state.mask.copy())
            state.mask = edit_state["preview_mask"].copy()
            status.value = (
                "削除を適用 — 続けて編集するか ✓ OK で確定"
                if state.mode == "seg_erase"
                else "追加を適用 — 続けて編集するか ✓ OK で確定"
            )
        else:
            status.value = "変化なし — 別の箇所を長押しするか ✓ OK で確定"
        edit_state["preview_mask"] = None
        edit_state["changed"] = False

    def on_tap_down(e: ft.TapEvent) -> None:
        if state.img_gray is None:
            return
        if state.mode in ("seg_erase", "seg_draw"):
            ox, oy = _to_img_xy(e.local_x, e.local_y)
            ix, iy = int(ox), int(oy)
            edit_state["pressing"] = True
            edit_state["preview_mask"] = None
            edit_state["changed"] = False
            edit_state["seed"] = (ix, iy)
            is_erase = state.mode == "seg_erase"
            status.value = (
                "Erase中…（緑の内外どちらでも可・明るい縁を削除）"
                if is_erase
                else "Draw中…（緑の内外どちらでも可・暗部を追加）"
            )
            base_gray = _overlay_base_gray()
            if base_gray is None:
                return
            if is_erase:
                preview, delta = erase_oversegmentation_from_mask(
                    base_gray, state.mask, (ix, iy), 0.5,
                )
            else:
                preview, delta = expand_segmentation_from_mask(
                    base_gray, state.mask, (ix, iy), 0.5,
                )
            edit_state["preview_mask"] = preview
            if delta > 0:
                edit_state["changed"] = True
                _push_image_rgb(_build_rgb_frame(preview), do_update=True, jpeg_quality=72)

            async def _run() -> None:
                await continuous_edit(ix, iy)

            page.run_task(_run)
            return
        if state.mode != "fovea":
            return
        ox, oy = _to_img_xy(e.local_x, e.local_y)
        state.fovea_xy = (ox, oy)
        state.img_gray_raw = state.img_gray.copy()
        state.foveal_correction_applied = False
        state.foveal_correction_skipped = False
        if not _try_apply_width_scale() and state.scale_um_px is None:
            status.value = (
                "中心窩を指定しました — HRA の場合は先に「①' 幅指定」で mm を設定してください"
                if modality_dd.value == "hra"
                else "中心窩を指定しました — 先にスケール較正を完了してください"
            )
            state.mode = "foveal_confirm"
        else:
            state.mode = "foveal_confirm"
            status.value = f"中心窩: ({ox:.0f}, {oy:.0f}) — 補正強度を調整"
        state.series_results = None
        state.mask = None
        state.stats = None
        foveal_strength_slider.value = int(state.foveal_correction_strength * 100)
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    def on_tap_up(_e: ft.TapEvent) -> None:
        if state.mode not in ("seg_erase", "seg_draw"):
            return
        edit_state["pressing"] = False
        _commit_edit_stroke()
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        page.update()

    def on_foveal_slider_change(_e) -> None:
        if state.mode != "foveal_confirm":
            return
        refresh_image()

    def on_foveal_ok(_=None) -> None:
        err = _foveal_prereq_error()
        if err:
            status.value = err
            page.update()
            return
        if _uncorrected_gray() is None:
            status.value = "画像が読み込まれていません"
            page.update()
            return
        strength = _foveal_strength()
        state.foveal_correction_applied = strength > 0
        state.foveal_correction_skipped = False
        state.foveal_correction_strength = strength
        state.series_results = None
        _run_segmentation_review()
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    def on_foveal_skip(_=None) -> None:
        raw = _uncorrected_gray()
        if raw is not None:
            state.img_gray = raw.copy()
        state.foveal_correction_applied = False
        state.foveal_correction_skipped = True
        state.series_results = None
        _run_segmentation_review()
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    def on_foveal_redo(_=None) -> None:
        state.fovea_xy = None
        state.img_gray_raw = None
        state.foveal_correction_applied = False
        state.foveal_correction_skipped = False
        state.mode = "fovea"
        status.value = "中心窩の位置をもう一度タップしてください"
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    def on_reanalyze(_=None) -> None:
        nonlocal seg_correction_stats
        if state.img_gray is None:
            status.value = "画像を読み込んでください"
        elif state.scale_um_px is None:
            status.value = "先にスケール設定を完了してください"
        elif state.fovea_xy is None:
            status.value = "先に中心窩を指定してください"
        else:
            seg_gray = _gray_for_segmentation()
            if seg_gray is None:
                status.value = "画像を読み込んでください"
            else:
                seg_correction_stats = _compute_seg_correction_stats(seg_gray)
                state.img_gray = seg_gray.copy()
                state.mask = _run_segmentation(seg_gray)
                state.stats = None
                state.mode = "seg_review"
                edit_state["pressing"] = False
                edit_state["preview_mask"] = None
                edit_state["undo_stack"] = []
                edit_state["session_base"] = None
                status.value = _safe_text(
                    f"再解析 ({_segmentation_source_label()})"
                    " - Erase / Draw / 範囲消去で修正、OK で確定"
                )
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    def on_reanalyze_high_strength(_=None) -> None:
        """補正強度を 80-100% に上げて再解析。"""
        nonlocal seg_correction_stats
        if state.fovea_xy is None or state.scale_um_px is None:
            _set_status("中心窩とスケールが必要です")
            page.update()
            return
        if state.img_gray is None:
            _set_status("画像を読み込んでください")
            page.update()
            return
        strength = float(high_strength_slider.value or 85) / 100.0
        strength = max(0.8, min(1.0, strength))
        state.foveal_correction_strength = strength
        state.foveal_correction_applied = True
        state.foveal_correction_skipped = False
        foveal_strength_slider.value = int(strength * 100)
        high_strength_slider.value = int(strength * 100)
        raw = _uncorrected_gray()
        if raw is None:
            _set_status("原画像がありません")
            page.update()
            return
        state.img_gray = apply_foveal_block_correction(
            raw, state.fovea_xy, state.scale_um_px, strength,
        )
        seg_gray = _gray_for_segmentation()
        if seg_gray is None:
            _set_status("画像を読み込んでください")
            page.update()
            return
        seg_correction_stats = _compute_seg_correction_stats(seg_gray)
        state.img_gray = seg_gray.copy()
        state.mask = _run_segmentation(seg_gray)
        state.stats = None
        state.mode = "seg_review"
        edit_state["pressing"] = False
        edit_state["preview_mask"] = None
        edit_state["undo_stack"] = []
        edit_state["session_base"] = None
        _set_status(
            f"高強度再解析 ({_segmentation_source_label()})"
            " - Erase / Draw / 範囲消去で修正、OK で確定"
        )
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    def on_clahe_settings_change(_=None) -> None:
        state.use_clahe = bool(use_clahe_sw.value)
        try:
            state.clahe_clip = float(clahe_clip_dd.value or 2.0)
        except ValueError:
            state.clahe_clip = 2.0
        clahe_clip_dd.disabled = not state.use_clahe
        if state.mask is not None and state.img_gray is not None and state.fovea_xy is not None:
            on_reanalyze()
        else:
            page.update()

    def on_seg_clear_start(_=None) -> None:
        if state.mask is None:
            _set_status("先に解析を実行してください")
            page.update()
            return
        if state.mode == "seg_review" and edit_state["session_base"] is None:
            edit_state["session_base"] = state.mask.copy()
            edit_state["undo_stack"] = []
        state.clear_pts = []
        state.mode = "seg_clear"
        edit_state["preview_mask"] = None
        edit_state["changed"] = False
        edit_state["pressing"] = False
        _set_status("フリーハンドで消去範囲を囲んでください")
        refresh_hint()
        update_confirm_visibility()
        page.update()

    def on_seg_clear_ok(_=None) -> None:
        if state.mask is None or len(state.clear_pts) < 3:
            _set_status("フリーハンドで消去範囲を囲んでください")
            page.update()
            return
        edit_state["undo_stack"].append(state.mask.copy())
        new_mask, removed = clear_mask_in_freehand_region(state.mask, state.clear_pts)
        if removed == 0:
            edit_state["undo_stack"].pop()
        else:
            state.mask = new_mask
        state.clear_pts = []
        state.mode = "seg_review"
        _set_status(
            f"範囲内マスクを消去 ({removed} px) - 続けて編集するか OK"
            if removed else "範囲内にマスクがありません - やり直しまたは続行"
        )
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        page.update()

    def on_seg_clear_redo(_=None) -> None:
        state.clear_pts = []
        state.mode = "seg_clear"
        _set_status("フリーハンドで消去範囲を囲んでください")
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        page.update()

    def on_clear_auto_segmentation(_=None) -> None:
        """自動セグメンテーション結果をすべて消去（マスクを空にする）。"""
        if state.mask is None or state.mode not in ("seg_review", "seg_erase", "seg_draw"):
            return
        if state.mode == "seg_review" and edit_state["session_base"] is None:
            edit_state["session_base"] = state.mask.copy()
            edit_state["undo_stack"] = []
        if np.any(state.mask > 0):
            edit_state["undo_stack"].append(state.mask.copy())
        state.mask = np.zeros_like(state.mask)
        edit_state["preview_mask"] = None
        edit_state["changed"] = False
        edit_state["pressing"] = False
        state.mode = "seg_review"
        _set_status("自動セグメンテーションを消去 — Draw で追加するか OK で確定")
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        page.update()

    def on_disc_ok(_=None) -> None:
        if len(state.disc_draft_pts) != 2:
            status.value = "直線を描いてください"
            page.update()
            return
        p1, p2 = state.disc_draft_pts
        try:
            state.disc_pts = [p1, p2]
            state.scale_um_px = calibration.compute_scale_um_per_px(
                p1, p2, float(disc_field.value or state.disc_diameter_um))
            state.calibration_source = "disc"
        except ValueError as exc:
            status.value = str(exc)
            page.update()
            return
        state.disc_draft_pts = []
        state.mode = None
        _invalidate_saved_results()
        status.value = f"乳頭計測完了: {state.scale_um_px:.2f} μm/px — 確定済み結果をクリアしました"
        disc_info.value = ""
        _refresh_scale_info()
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    def on_disc_redo(_=None) -> None:
        state.disc_draft_pts = []
        state.mode = "disc_line"
        disc_info.value = ""
        status.value = "乳頭の直線をもう一度ドラッグしてください"
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        page.update()

    def on_crop_ok(_=None) -> None:
        if state.img_gray is None or len(state.crop_pts) != 2:
            status.value = "クロップ範囲を描いてください"
            page.update()
            return
        h, w = state.img_gray.shape
        p1, p2 = state.crop_pts
        try:
            box = normalize_box(p1[0], p1[1], p2[0], p2[1], w, h)
            if state.crop_box is None:
                state.crop_box = box
            else:
                ox0, oy0, _, _ = state.crop_box
                bx0, by0, bx1, by1 = box
                state.crop_box = (ox0 + bx0, oy0 + by0, ox0 + bx1, oy0 + by1)
            state.img_gray = apply_crop(state.img_gray, box)
            if state.img_gray_raw is not None:
                state.img_gray_raw = apply_crop(state.img_gray_raw, box)
            state.crop_pts = []
            state.mode = None
            state.reset_per_image()
            status.value = "クロップ完了"
        except ValueError as exc:
            status.value = str(exc)
            state.crop_pts = []
            state.mode = "crop_drag"
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        page.update()

    def on_crop_redo(_=None) -> None:
        state.crop_pts = []
        state.mode = "crop_drag"
        status.value = "クロップ矩形をもう一度ドラッグしてください"
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        page.update()

    def on_width_start(_=None) -> None:
        if state.img_gray is None:
            status.value = "画像を読み込んでください"
            page.update()
            return
        state.mode = "width_cal"
        if state.field_width_mm is not None:
            field_width_field.value = str(state.field_width_mm)
        _refresh_image_width_label()
        status.value = ""
        refresh_hint()
        update_confirm_visibility()
        page.update()

    def on_width_ok(_=None) -> None:
        if state.img_gray is None:
            status.value = "画像を読み込んでください"
            page.update()
            return
        try:
            field_mm = float(field_width_field.value)
            if field_mm <= 0:
                raise ValueError
        except ValueError:
            status.value = "画像幅 (mm) を正しく入力してください"
            page.update()
            return
        _, w_px = state.img_gray.shape
        try:
            state.scale_um_px = calibration.compute_scale_um_per_px_from_field_width(w_px, field_mm)
        except ValueError as exc:
            status.value = str(exc)
            page.update()
            return
        state.calibration_source = "width"
        state.field_width_mm = field_mm
        state.disc_pts = []
        state.disc_draft_pts = []
        state.mode = None
        _invalidate_saved_results()
        disc_info.value = ""
        status.value = (
            f"幅指定完了: {w_px} px = {field_mm} mm → {state.scale_um_px:.2f} μm/px"
            " — 確定済み結果をクリアしました"
        )
        _refresh_scale_info()
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    def on_width_cancel(_=None) -> None:
        state.mode = None
        status.value = ""
        refresh_hint()
        update_confirm_visibility()
        page.update()

    def on_modality_change(e) -> None:
        state.image_modality = modality_dd.value or "optos"
        refresh_hint()
        page.update()

    def set_mode(mode: str) -> None:
        state.mode = mode
        drag["active"] = False
        if mode == "disc_line":
            state.disc_pts = []
            state.disc_draft_pts = []
            state.scale_um_px = None
            state.calibration_source = None
            disc_info.value = ""
            _refresh_scale_info()
            status.value = "乳頭の両端を結ぶ直線をドラッグしてください"
        elif mode == "crop_drag":
            state.crop_pts = []
            status.value = "クロップ範囲をドラッグしてください"
        elif mode == "fovea":
            if state.scale_um_px is None and not _try_apply_width_scale():
                if modality_dd.value == "hra":
                    status.value = "HRA: 先に「①' 幅指定」で画像幅 (mm) を設定してください"
                else:
                    status.value = "先にスケール較正を完了してください"
            else:
                status.value = "中心窩の位置をタップしてください"
        refresh_hint()
        update_confirm_visibility()
        page.update()

    def _run_segmentation_review(*, status_prefix: str = "セグメンテーション") -> bool:
        """③ 解析と同じセグメント確認画面へ進める。"""
        nonlocal seg_correction_stats
        seg_gray = _gray_for_segmentation()
        if seg_gray is None:
            _set_status("画像を読み込んでください")
            return False
        seg_correction_stats = _compute_seg_correction_stats(seg_gray)
        state.img_gray = seg_gray.copy()
        state.mask = _run_segmentation(seg_gray)
        state.stats = None
        state.mode = "seg_review"
        edit_state["pressing"] = False
        edit_state["preview_mask"] = None
        edit_state["undo_stack"] = []
        edit_state["session_base"] = None
        _set_status(
            f"{status_prefix}完了 ({_segmentation_source_label()})"
            " - Erase / Draw / 範囲消去で修正、OK で確定"
        )
        return True

    def on_analyze(_=None) -> None:
        nonlocal seg_correction_stats
        if state.scale_um_px is None:
            if modality_dd.value == "hra":
                _set_status("先にスケール設定 (幅指定または乳頭計測) を完了してください")
            else:
                _set_status("先に乳頭計測を完了してください")
        elif state.fovea_xy is None:
            _set_status("先に中心窩を指定してください")
        elif state.img_gray is None:
            _set_status("画像を読み込んでください")
        else:
            if state.mode == "foveal_confirm":
                if not _apply_pending_foveal_correction():
                    page.update()
                    return
            _run_segmentation_review()
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    def on_seg_ok(_=None) -> None:
        if state.mask is None:
            _set_status("マスクがありません")
            page.update()
            return
        edit_state["pressing"] = False
        edit_state["preview_mask"] = None
        edit_state["changed"] = False
        edit_state["undo_stack"] = []
        edit_state["session_base"] = None
        state.stats = ga_stats(state.mask, state.scale_um_px, state.fovea_xy)
        state.mode = None
        saved = 0
        n = len(state.files)
        batch_all_done = False
        if state.batch_mode:
            _save_batch_result()
            saved = _batch_saved_count()
            pos = state.file_idx + 1
            batch_all_done = saved >= n and n >= 2
            if saved >= n:
                _set_status(
                    f"確定済み {saved}/{n} (画像 {pos}/{n})"
                    " - 全画像完了。時系列表示を開始します…"
                )
            else:
                _set_status(
                    f"確定済み {saved}/{n} (画像 {pos}/{n})"
                    " - 結果を確認し「次へ ▶」で次画像"
                )
        else:
            _set_status("解析完了")
        if state.batch_mode and batch_all_done:
            land_name = state.files[state.file_idx].name if state.files else None
            _launch_series_display(auto=True, landing_filename=land_name)
            return
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        _update_batch_ui()
        page.update()

    def _begin_edit_mode(new_mode: str) -> None:
        if state.mask is None:
            status.value = "先に解析を実行してください"
            page.update()
            return
        if state.mode == "seg_review":
            edit_state["session_base"] = state.mask.copy()
            edit_state["undo_stack"] = []
        edit_state["preview_mask"] = None
        edit_state["changed"] = False
        edit_state["pressing"] = False
        state.mode = new_mode
        status.value = (
            "長押しで削除（複数箇所可）— ↩ やり直しで取消"
            if new_mode == "seg_erase"
            else "長押しで暗部を追加（複数箇所可）— ↩ やり直しで取消"
        )
        refresh_hint()
        update_confirm_visibility()
        page.update()

    def on_edit_redo(_=None) -> None:
        if state.mode not in ("seg_erase", "seg_draw"):
            return
        edit_state["pressing"] = False
        edit_state["preview_mask"] = None
        edit_state["changed"] = False
        if edit_state["undo_stack"]:
            state.mask = edit_state["undo_stack"].pop()
            status.value = "直前の編集を取り消しました"
        elif edit_state["session_base"] is not None:
            state.mask = edit_state["session_base"].copy()
            edit_state["undo_stack"] = []
            status.value = "編集開始前のマスクに戻しました"
        else:
            status.value = "取り消す編集がありません"
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        page.update()

    def on_seg_erase_click(_=None) -> None:
        _begin_edit_mode("seg_erase")

    def on_seg_draw_click(_=None) -> None:
        _begin_edit_mode("seg_draw")

    def _launch_series_display(
        *,
        auto: bool = False,
        landing_filename: str | None = None,
    ) -> bool:
        """確定済み結果から時系列表示を構築する。"""
        saved = _batch_saved_count()
        n = len(state.files)
        if saved < 2:
            _set_status(f"時系列には2枚以上の確定結果が必要です (現在 {saved}/{n} 枚)")
            return False
        prefer = landing_filename
        if prefer is None and state.files:
            prefer = state.files[state.file_idx].name
        dd_gen = _begin_series_launch_guard()
        _set_status("時系列表示を構築中…")
        # Do NOT page.update() here: flushing UI mid-build lets deferred
        # file_dd on_change fire while series_results is still None.
        try:
            results = build_series_from_saved(
                state.per_file_results,
                state.files,
                state.series_dates,
                register=True,
            )
        except ValueError as exc:
            _end_series_launch_guard(dd_gen)
            _set_status(str(exc))
            page.update()
            return False
        except Exception as exc:
            _end_series_launch_guard(dd_gen)
            _set_status(f"時系列エラー: {exc}")
            page.update()
            return False
        state.disc_diameter_um = float(disc_field.value or 1800)
        _apply_series_results(results, landing_filename=prefer)
        prefix = "時系列表示を開始" if auto else "時系列表示"
        _set_status(f"{prefix} ({len(results)} 時点) — GIF / グラフで確認できます")
        state.view_mode = "work"
        view_mode.value = "work"
        view_mode.visible = True
        series_date_dd.visible = True
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        _update_batch_ui(refresh_dropdown=False)
        _refresh_file_dropdown_labels()
        page.update()
        _end_series_launch_guard(dd_gen)
        return True

    def on_series(_=None) -> None:
        _launch_series_display(auto=False)

    def on_pick_folder(_=None) -> None:
        async def _pick():
            path = await folder_picker.get_directory_path(dialog_title="入力フォルダを選択")
            if path:
                input_field.value = path
                on_load_files()
        page.run_task(_pick)

    folder_picker = ft.FilePicker(on_result=lambda _: None)
    page.overlay.extend([folder_picker, date_picker])

    def on_view_change(e) -> None:
        state.view_mode = e.control.value
        refresh_image()
        page.update()

    view_mode.on_change = on_view_change

    def on_series_date(e) -> None:
        if e.control.value is not None:
            state.series_time_idx = int(e.control.value)
            _sync_file_idx_to_series_timepoint(update_dropdown=False)
            refresh_image()
            refresh_results()

    series_date_dd.on_change = on_series_date

    gesture_stack = ft.Stack(
        [img_ctrl, gif_img],
        width=DISPLAY_WIDTH,
        height=200,
    )
    img_stack = ft.GestureDetector(
        content=gesture_stack,
        on_pan_start=on_pan_start,
        on_pan_update=on_pan_update,
        on_pan_end=on_pan_end,
        on_tap_down=on_tap_down,
        on_tap_up=on_tap_up,
        mouse_cursor=ft.MouseCursor.PRECISE,
    )

    series_btn = btn_toolbar("④ 時系列", on_click=on_series)
    series_btn.visible = False

    series_launch_btn = btn_primary("④ 時系列で表示", on_click=on_series)
    series_launch_btn.visible = False
    series_launch_row = ft.Row([series_launch_btn], visible=False)

    toolbar = ft.Row(
        [
            btn_toolbar("⓪ クロップ", on_click=lambda _: set_mode("crop_drag")),
            btn_toolbar("① 乳頭", on_click=lambda _: set_mode("disc_line")),
            btn_toolbar("①' 幅指定", on_click=on_width_start),
            btn_toolbar("② 中心窩", on_click=lambda _: set_mode("fovea")),
            btn_primary("③ 解析", on_click=on_analyze),
            series_btn,
        ],
        wrap=True,
        spacing=8,
    )

    def update_confirm_visibility() -> None:
        cancel_row.visible = state.mode in (
            "disc_line", "crop_drag", "fovea", "seg_review", "seg_erase", "seg_draw",
            "seg_clear",
        )
        disc_confirm_row.visible = state.mode == "disc_confirm"
        crop_confirm_row.visible = state.mode == "crop_confirm"
        width_panel.visible = state.mode == "width_cal"
        foveal_confirm_panel.visible = state.mode == "foveal_confirm"
        seg_confirm_row.visible = state.mode in (
            "seg_review", "seg_erase", "seg_draw", "seg_clear",
        )
        seg_edit_redo_btn.visible = state.mode in ("seg_erase", "seg_draw")
        seg_clear_confirm_row.visible = state.mode == "seg_clear_confirm"
        seg_auto_clear_btn.visible = state.mode in (
            "seg_review", "seg_erase", "seg_draw",
        )

    def on_cancel(_=None) -> None:
        nonlocal seg_correction_stats
        if state.mode in ("seg_clear", "seg_clear_confirm"):
            state.clear_pts = []
            state.mode = "seg_review"
            status.value = "範囲消去をキャンセル"
        elif state.mode in ("seg_erase", "seg_draw"):
            if edit_state["session_base"] is not None:
                state.mask = edit_state["session_base"].copy()
            edit_state["pressing"] = False
            edit_state["preview_mask"] = None
            edit_state["undo_stack"] = []
            state.mode = "seg_review"
            status.value = "編集をキャンセル — ✓ OK で確定"
        elif state.mode == "seg_review":
            state.mask = None
            state.stats = None
            _restore_display_gray()
            seg_correction_stats = None
            edit_state["pressing"] = False
            edit_state["preview_mask"] = None
            edit_state["undo_stack"] = []
            edit_state["session_base"] = None
            if state.fovea_xy is not None:
                _enter_foveal_confirm_preserving_fovea()
                fx, fy = state.fovea_xy
                err = _foveal_prereq_error()
                if err:
                    status.value = (
                        f"解析をキャンセル — 原画像。"
                        f"中心窩 ({fx:.0f}, {fy:.0f}) — {err}"
                    )
                else:
                    status.value = (
                        f"解析をキャンセル — 原画像。"
                        f"中心窩 ({fx:.0f}, {fy:.0f}) — 補正強度を調整"
                    )
            else:
                state.mode = None
                status.value = "解析をキャンセル — 原画像に戻しました"
        else:
            state.cancel_mode()
        drag["active"] = False
        if state.mode not in ("seg_review",):
            if (
                status.value.startswith("編集をキャンセル")
                or status.value.startswith("範囲消去")
                or status.value.startswith("解析をキャンセル")
                or status.value.startswith("画像 ")
                or "確定済み" in status.value
            ):
                pass
            elif state.mode is None:
                status.value = ""
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    seg_auto_clear_btn = btn_outlined(
        "自動セグメンテーションキャンセル",
        on_click=on_clear_auto_segmentation,
    )
    seg_auto_clear_btn.visible = False

    cancel_row = ft.Row([
        btn_outlined("✕ キャンセル", on_click=on_cancel),
        seg_auto_clear_btn,
    ], visible=False)

    disc_confirm_row = ft.Row([
        btn_ok("✓ OK", on_click=on_disc_ok),
        btn_outlined("↩ やり直し", on_click=on_disc_redo),
    ], visible=False)

    crop_confirm_row = ft.Row([
        btn_ok("✓ OK", on_click=on_crop_ok),
        btn_outlined("↩ やり直し", on_click=on_crop_redo),
    ], visible=False)

    seg_edit_redo_btn = btn_outlined("↩ やり直し", on_click=on_edit_redo)
    seg_edit_redo_btn.visible = False

    seg_ok_btn = btn_ok("OK", on_click=on_seg_ok)

    seg_confirm_row = ft.Row([
        seg_ok_btn,
        btn_erase("Erase 白領域（長押し）", on_click=on_seg_erase_click),
        btn_draw("Draw 暗部（長押し）", on_click=on_seg_draw_click),
        btn_outlined("範囲消去", on_click=on_seg_clear_start),
        seg_edit_redo_btn,
    ], visible=False, wrap=True)

    seg_clear_confirm_row = ft.Row([
        btn_ok("✓ 範囲を消去", on_click=on_seg_clear_ok),
        btn_outlined("↩ やり直し", on_click=on_seg_clear_redo),
    ], visible=False)

    width_panel = ft.Container(
        visible=False,
        bgcolor=T.PANEL_BG,
        border=ft.border.all(1, "#374151"),
        border_radius=8,
        padding=ft.padding.all(12),
        content=ft.Column(
            [
                ft.Text("幅指定スケール", size=13, weight=ft.FontWeight.W_600, color=T.PRIMARY),
                ft.Text(
                    "画像の水平方向の実長 (mm) を入力してください",
                    size=11,
                    color=T.TEXT_MUTED,
                ),
                ft.Row(
                    [
                        field_width_field,
                        ft.Text("mm", size=14, color=T.TEXT),
                    ],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.END,
                ),
                image_width_label,
                ft.Row(
                    [
                        btn_ok("✓ OK", on_click=on_width_ok),
                        btn_outlined("✕ キャンセル", on_click=on_width_cancel),
                    ],
                    spacing=8,
                ),
            ],
            spacing=8,
            tight=True,
        ),
    )
    modality_dd.on_change = on_modality_change
    use_clahe_sw.on_change = on_clahe_settings_change
    clahe_clip_dd.on_change = on_clahe_settings_change
    use_clahe_sw.value = state.use_clahe
    clahe_clip_dd.value = str(state.clahe_clip)
    clahe_clip_dd.disabled = not state.use_clahe

    foveal_strength_slider.on_change = on_foveal_slider_change

    foveal_confirm_panel = ft.Container(
        visible=False,
        bgcolor=T.PANEL_BG,
        border=ft.border.all(1, "#374151"),
        border_radius=8,
        padding=ft.padding.all(12),
        content=ft.Column(
            [
                ft.Text("中心窩ブロック補正", size=13, weight=ft.FontWeight.W_600, color=T.PRIMARY),
                ft.Text(
                    "黄斑色素による滑らかな同心円状の暗化のみ補正（完全な GA 低蛍光は保護）",
                    size=11,
                    color=T.TEXT_MUTED,
                ),
                ft.Text("補正強度", size=11, color=T.TEXT_MUTED),
                foveal_strength_slider,
                ft.Row(
                    [
                        btn_ok("✓ 補正を適用", on_click=on_foveal_ok),
                        btn_outlined("補正しない", on_click=on_foveal_skip),
                        btn_outlined("↩ やり直し", on_click=on_foveal_redo),
                    ],
                    spacing=8,
                    wrap=True,
                ),
            ],
            spacing=8,
            tight=True,
        ),
    )

    left = ft.Column(
        [
            ft.Text("GA Segmentation Pro", size=22, weight=ft.FontWeight.BOLD, color=T.PRIMARY),
            ft.Text("研究用", size=11, color=T.TEXT_MUTED),
            toolbar,
            batch_progress,
            cancel_row,
            disc_confirm_row,
            crop_confirm_row,
            width_panel,
            foveal_confirm_panel,
            seg_confirm_row,
            seg_clear_confirm_row,
            hint,
            disc_info,
            scale_info,
            status,
            series_launch_row,
            view_mode,
            series_date_dd,
            img_stack,
            graph_area,
        ],
        spacing=8,
        scroll=ft.ScrollMode.AUTO,
    )

    sidebar = ft.Column(
        [
            ft.Text("フォルダ", weight=ft.FontWeight.BOLD, color=T.PRIMARY),
            ft.Row([input_field, ft.IconButton(ft.Icons.FOLDER_OPEN, on_click=on_pick_folder, tooltip="フォルダ選択")]),
            output_field,
            btn_sidebar("画像リストを読込", on_click=on_load_files, expand=True),
            file_dd,
            series_dates_section,
            ft.Divider(color="#374151"),
            modality_dd,
            disc_field,
            ft.Divider(color="#374151"),
            ft.Text("セグメンテーション", size=12, weight=ft.FontWeight.W_600, color=T.PRIMARY),
            use_clahe_sw,
            clahe_clip_dd,
            ft.Text(
                "CLAHE OFF は過大マスクになりやすい。clip を上げると面積が縮小しやすい。",
                size=10,
                color=T.TEXT_MUTED,
            ),
            ft.Divider(color="#374151"),
            age_field,
            sex_dd,
            prog_field,
            ft.Text(f"余命: {LIFE_TABLE_SOURCE}", size=10, color=T.TEXT_MUTED),
        ],
        width=340,
        spacing=10,
        scroll=ft.ScrollMode.AUTO,
    )

    page.add(
        ft.Container(
            content=ft.Row(
                [
                    sidebar,
                    ft.VerticalDivider(width=1, color="#374151"),
                    ft.Container(content=left, expand=3, padding=ft.padding.only(left=8, right=8)),
                    ft.VerticalDivider(width=1, color="#374151"),
                    ft.Container(content=results_col, width=280, padding=10),
                ],
                expand=True,
                spacing=0,
            ),
            expand=True,
        )
    )
    page.update()

    refresh_hint()
    update_confirm_visibility()
    _refresh_image_width_label()
    _refresh_scale_info()
    refresh_series_dates_ui()
    _update_batch_ui()

    async def _deferred_auto_load() -> None:
        import asyncio
        import os
        await asyncio.sleep(0.05)
        auto_dir = os.environ.get("GA_INPUT_DIR", "").strip()
        if auto_dir:
            input_field.value = auto_dir
            on_load_files()

    page.run_task(_deferred_auto_load)
