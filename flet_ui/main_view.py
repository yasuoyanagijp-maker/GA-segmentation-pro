"""Flet メインビュー — engine を直接呼び出す（FastAPI なし）。"""

from __future__ import annotations

import asyncio
import base64
import copy
import logging
import math
from datetime import date, datetime
from pathlib import Path

import cv2
import flet as ft
import numpy as np

from engine import calibration, segmentation
from engine.crop import apply_crop, normalize_box
from engine.cases import (
    UNCLASSIFIED_CASE_ID,
    create_case,
    discover_cases,
    move_files_to_case,
    ordered_case_ids,
)
from engine.io_utils import SUPPORTED_SUFFIXES, load_gray
from engine.life_table import LIFE_TABLE_SOURCE, format_life_expectancy, life_expectancy_years
from engine.overlay import make_overlay
from engine.region_growing import (
    clear_mask_in_freehand_region,
    erase_oversegmentation_from_mask,
    expand_segmentation_from_mask,
)
from engine.export import (
    build_batch_csv_bytes,
    gawain_figure_png_from_context,
    generate_pdf_report_bytes,
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
from engine.series_registration import REGISTRATION_METHODS
from engine.series_roi_correction import apply_roi_correction_to_series
from engine.timeseries import (
    build_series_from_saved,
    datetime_to_plot_x,
    disc_xy_from_pts,
    estimate_growth_mm2_per_year,
    estimate_growth_mm_sqrt_per_year,
    estimate_progression_rate_um_per_year,
    extract_image_datetime,
    guess_image_datetime,
    datetime_to_iso,
    is_datetime_source_confirmed,
    parse_series_datetime,
    parse_user_datetime_input,
    refresh_series_datetimes_for_files,
    segment_aligned_series,
    series_metric_regressions,
    sync_date_to_snapshots,
    landmarks_from_snapshot,
    landmarks_ready_for_registration,
    overlays_to_gif_bytes,
    reregister_timepoint,
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
APP_VERSION = "datetime-landmark-gate-2026-07-24"  # 画面右下に表示（コード反映確認用）
# 未読込時のプレースホルダ（Flet Image は src か src_base64 が必須）
EMPTY_PX = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

_log = logging.getLogger("ga.batch")
if not _log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[GA-batch] %(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)
    _log.propagate = False


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
      - "ignore": do nothing (sync skip, empty, or launch guard)
      - "series_timepoint": switch series time index (payload = series index)
      - "load_image": leave series (if any) and load files[payload]

    While series is open:
      - selecting a file already in the series → switch timepoint
      - selecting a file NOT in the series (typically unconfirmed) → exit
        series and load that image so the user can continue batch analysis
    During series_launching, ignore all dropdown events (race guard).
    """
    if skip or series_launching or not selected:
        return "ignore", None
    if series_results is not None:
        for i, r in enumerate(series_results):
            if r["filename"] == selected:
                return "series_timepoint", i
        # Not in series → exit series and open that file for analysis.
        for i, f in enumerate(files):
            if f.name == selected:
                return "load_image", i
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
    # Yield once so Flet web can finish the websocket handshake before heavy UI build.
    await asyncio.sleep(0)
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
    scale_method_dd = ft.Dropdown(
        label="スケール基準 (Optos)",
        width=320,
        options=[
            ft.dropdown.Option(
                key="dfd",
                text="DFD 4.76mm（文献平均・推奨）",
            ),
            ft.dropdown.Option(
                key="disc",
                text="乳頭径固定（1800μm）",
            ),
        ],
        value=state.scale_method or "dfd",
        tooltip=(
            "DFD: 乳頭中心–中心窩距離を 4.76mm（Beijing Eye Study）として較正。"
            "乳頭径: 従来どおり乳頭端2点＝指定μm。"
        ),
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
    series_mode_sw = ft.Switch(
        label="時系列モード（Registration）",
        value=state.series_mode,
        tooltip=(
            "OFF（既定）: 独立解析 — 各画像で landmark→scale→seg→測定。"
            "ON: 症例内のみ時系列 — landmark→日時→Registration→整列QC→seg→測定。"
        ),
    )
    registration_sw = ft.Switch(
        label="位置合わせ (Registration)",
        value=state.series_registration_enabled,
        tooltip="時系列モード ON 時は常に有効（症例をまたぐ registration は不可）",
        visible=False,
    )
    reseg_after_reg_sw = ft.Switch(
        label="整列後にセグメンテーション（本線）",
        value=state.series_resegment_enabled,
        tooltip=(
            "ON（既定・本線）: 整列QCの後、整列空間で GA を segmentation。"
            "OFF: 先に確定したマスクを warp のみ（失敗時フォールバック）。"
        ),
    )
    roi_correction_sw = ft.Switch(
        label="マスクQC補正",
        value=state.series_roi_enabled,
        tooltip=(
            "ON（既定）: 整列・seg 後、面積が時系列で異常な時点のマスクを"
            "QC補正し統計を再計算。OFF: 補正しない"
        ),
    )
    reg_apply_btn = btn_toolbar("Registration", on_click=lambda _: None)
    reg_apply_btn.visible = False
    reg_progress_bar = ft.ProgressBar(
        value=0,
        bar_height=8,
        color=T.PRIMARY,
        bgcolor="#374151",
        visible=False,
    )
    reg_progress_label = ft.Text("", size=11, color=T.TEXT_MUTED, visible=False)
    case_dd = ft.Dropdown(label="症例", options=[], width=320, text_size=13)
    new_case_field = ft.TextField(
        label="新規症例名", width=200, text_size=12, hint_text="例: patient3",
    )
    create_case_btn = btn_sidebar("症例を作成", on_click=lambda _: None, expand=True)
    move_case_dd = ft.Dropdown(
        label="移動先の症例", options=[], width=320, text_size=12,
    )
    case_move_list_col = ft.Column(spacing=2, scroll=ft.ScrollMode.AUTO, height=140)
    case_select_all_btn = btn_outlined("すべて選択", on_click=lambda _: None)
    case_deselect_all_btn = btn_outlined("選択解除", on_click=lambda _: None)
    move_selected_btn = btn_sidebar(
        "選択画像を症例へ移動", on_click=lambda _: None, expand=True,
    )
    case_move_section = ft.Column(
        [
            ft.Text(
                "症例へ振り分け（複数選択可）",
                size=12,
                weight=ft.FontWeight.W_600,
                color=T.PRIMARY,
            ),
            ft.Row([case_select_all_btn, case_deselect_all_btn], spacing=6),
            case_move_list_col,
            move_case_dd,
            move_selected_btn,
        ],
        spacing=6,
    )
    prep_confirm_btn = btn_toolbar(
        "時系列準備として確定", on_click=lambda _: None,
    )
    prep_confirm_btn.visible = False
    series_dates_section = ft.Column(
        [
            ft.Text("時系列（検査日時）", size=12, weight=ft.FontWeight.W_600, color=T.PRIMARY),
            ft.Text(
                "手順: ①各画像の日時を入力し「確定」→ ②各画像で乳頭＋中心窩を確認し"
                "「時系列準備として確定」→ ③ Registration。"
                " 日時は EXIF/ファイル名。未取得は手動必須。",
                size=10,
                color=T.TEXT_MUTED,
            ),
            series_dates_col,
        ],
        spacing=6,
        visible=False,
    )
    date_picker_target: dict[str, str | None] = {"fname": None}
    # ignore=True while setting DatePicker.value programmatically (avoids apply loop)
    date_picker_guard = {"ignore": False}
    # skip=True while rebuilding dates Column (Checkbox on_change must not re-enter)
    dates_ui_guard = {"skip": False, "gen": 0}
    date_picker = ft.DatePicker(
        first_date=datetime(1990, 1, 1),
        last_date=datetime(2035, 12, 31),
    )
    foveal_strength_slider = ft.Slider(
        min=0, max=100, value=0, divisions=20,
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
    graph_metric_dd = ft.Dropdown(
        label="表示する指標",
        width=280,
        text_size=13,
        value=state.graph_metric,
        options=[
            ft.dropdown.Option(key="area_mm2", text="GA Area (mm2)"),
            ft.dropdown.Option(key="area_mm_sqrt", text="SQRT GA Area (mm)"),
            ft.dropdown.Option(key="gawain", text=_safe_text(GAWAIN_LABEL)),
            ft.dropdown.Option(key="dist_um", text="Distance to fovea (um)"),
        ],
    )

    disp_scale = {"v": 1.0}
    disp_size = {"w": DISPLAY_WIDTH, "h": 200}
    drag = {"active": False, "start": (0.0, 0.0)}
    edit_state = {
        "pressing": False,
        "preview_mask": None,
        "changed": False,
        "undo_stack": [],
        "session_base": None,
        "clear_outside": False,
    }
    batch_progress = ft.Text("", size=12, color=T.TEXT_MUTED, visible=False)
    # Dropdown sync: generation counter avoids asyncio.sleep(0) races where Flet
    # delivers on_change after skip was already cleared.
    dd_sync = {"skip": False, "gen": 0}
    # Blocks load_image while series is being built (covers page.update races
    # before series_results is assigned).
    series_guard = {"launching": False, "gen": 0, "roi_running": False}
    # gen: invalidate late call_soon_threadsafe progress callbacks after busy clears
    registration_guard = {"running": False, "gen": 0}
    _registration_controls: dict = {"redo_btn": None}
    seg_correction_stats: dict | None = None
    # 遅延生成されるコントロール（registration 確認パネル等）への参照置き場。
    _ui: dict = {}

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

    def _stash_reg_cache_from_series() -> None:
        """位置合わせ済み（または確認中）の行列をメモリに残す。"""
        results = state.series_results
        if not results:
            return
        if not (state.series_registered or state.series_review_active):
            return
        by_file: dict = {}
        for r in results:
            if r.get("is_reference"):
                continue
            m = r.get("reg_matrix")
            if m is None:
                continue
            by_file[r["filename"]] = {
                "matrix": np.asarray(m, dtype=np.float32).copy(),
                "method": r.get("reg_method", "affine"),
            }
        if not by_file:
            return
        state.series_reg_cache = {
            "selected": tuple(sorted(r["filename"] for r in results)),
            "ref": results[0]["filename"],
            "confirmed": bool(state.series_registered),
            "by_file": by_file,
        }

    def _clear_reg_cache() -> None:
        state.series_reg_cache = None

    def _prior_matrices_from_cache(
        selected_set: set[str],
        series_dates: dict[str, str],
        series_methods: dict[str, str],
    ) -> tuple[dict[str, np.ndarray] | None, bool]:
        """再利用可能な prior_matrices と、以前確認完了済みか。"""
        cache = state.series_reg_cache
        if not cache:
            return None, False
        selected_key = tuple(sorted(selected_set))
        if cache.get("selected") != selected_key:
            return None, False
        ordered = sorted(selected_set, key=lambda n: series_dates.get(n, ""))
        if not ordered or ordered[0] != cache.get("ref"):
            return None, False
        by_file = cache.get("by_file") or {}
        for fname, entry in by_file.items():
            if fname not in selected_set:
                return None, False
            cached_method = entry.get("method", "affine")
            wanted = series_methods.get(fname, cached_method)
            if wanted != cached_method:
                return None, False
        prior = {
            fname: entry["matrix"]
            for fname, entry in by_file.items()
            if fname in selected_set
        }
        return prior, bool(cache.get("confirmed"))

    def _clear_series_display() -> None:
        _stash_reg_cache_from_series()
        state.series_results = None
        state.series_overlays = None
        state.series_gif = None
        state.estimated_prog = None
        state.estimated_growth_mm2_per_year = None
        state.estimated_growth_mm_sqrt_per_year = None
        state.series_time_idx = 0
        state.series_review_active = False
        state.series_roi_review_active = False
        state.series_review_idx = 0
        state.series_registered = False
        state.series_roi_applied = False
        state.series_roi_qc = None
        state.view_mode = "work"
        view_mode.value = "work"
        view_mode.visible = False
        series_date_dd.visible = False
        prog_field.value = ""
        reg_apply_btn.visible = False
        # Case-move / series exit: drop progress chrome; invalidate late ticks.
        # In-flight registration finally still owns registration_guard["running"].
        registration_guard["gen"] += 1
        reg_progress_bar.visible = False
        reg_progress_label.visible = False
        reg_progress_bar.value = 0
        reg_progress_label.value = ""
        if "reg_review_panel" in _ui:
            _ui["reg_review_panel"].visible = False
        if "roi_review_panel" in _ui:
            _ui["roi_review_panel"].visible = False

    def _selected_dates_ready() -> bool:
        """選択対象の全画像に確定済み検査日時があるか。"""
        for fname in _selected_confirmed():
            src = state.series_date_source.get(fname)
            if state.series_dates_manual.get(fname):
                src = "manual"
            if not state.series_dates.get(fname):
                return False
            if not is_datetime_source_confirmed(src):
                return False
        return True

    def _selected_landmarks_ready() -> bool:
        """選択対象の全画像で乳頭＋中心窩が準備確定済みか。"""
        for fname in _selected_confirmed():
            if not landmarks_ready_for_registration(state.per_file_results.get(fname)):
                return False
        return True

    def _datetime_next_step_message(just_fname: str | None = None) -> str:
        """日時入力直後の案内（次に何を押すか）。"""
        n = len(state.files)
        dates_pending = 0
        lm_pending = 0
        for f in state.files:
            fname = f.name
            src = state.series_date_source.get(fname)
            if state.series_dates_manual.get(fname):
                src = "manual"
            if not state.series_dates.get(fname) or not is_datetime_source_confirmed(src):
                dates_pending += 1
            snap = state.per_file_results.get(fname)
            if not landmarks_ready_for_registration(snap):
                lm_pending += 1
        prefix = f"{just_fname} の日時を確定。 " if just_fname else "日時を確定。 "
        if dates_pending > 0:
            return _safe_text(
                f"{prefix}残り {dates_pending}/{n} 枚の日時を入力し「確定」"
                " → 続けて各画像で乳頭＋中心窩を確認"
            )
        if lm_pending > 0:
            return _safe_text(
                f"{prefix}次: 各画像で①乳頭→②中心窩→「時系列準備として確定」"
                f"（未準備 {lm_pending}/{n} 枚）"
            )
        if _selected_count() >= 2:
            return _safe_text(
                f"{prefix}全画像の日時・乳頭・中心窩が揃いました — "
                "④ Registration を押してください"
            )
        return _safe_text(
            f"{prefix}対象を2枚以上選択し、④ Registration へ"
        )

    def _update_registration_controls() -> None:
        """時系列モード・Registration・QC トグルの表示を同期。"""
        series_mode_sw.value = state.series_mode
        # 時系列モード ON なら registration 本線
        state.series_registration_enabled = bool(state.series_mode)
        registration_sw.value = state.series_registration_enabled
        reseg_after_reg_sw.value = state.series_resegment_enabled
        roi_correction_sw.value = state.series_roi_enabled
        dates_ok = _selected_dates_ready()
        landmarks_ok = _selected_landmarks_ready()
        ready = (
            state.series_mode
            and _selected_count() >= 2
            and dates_ok
            and landmarks_ok
            and not state.series_registered
            and not state.series_review_active
            and not state.series_roi_review_active
            and not registration_guard["running"]
        )
        reg_apply_btn.visible = state.series_mode and _selected_count() >= 2
        reg_apply_btn.disabled = not ready
        if state.series_mode and _selected_count() >= 2 and not dates_ok:
            reg_apply_btn.tooltip = (
                "全対象画像の検査日時を確定してください（EXIF/ファイル名/手動）"
            )
        elif state.series_mode and _selected_count() >= 2 and not landmarks_ok:
            reg_apply_btn.tooltip = (
                "全対象画像で乳頭（disc）と中心窩を確認し「時系列準備として確定」してください"
            )
        else:
            reg_apply_btn.tooltip = (
                "ランドマーク準備済みの症例内画像を Registration → 整列QC → seg"
            )

    def _invalidate_saved_results(*, reason: str = "scale_change") -> None:
        """スケール変更時: 時系列表示は破棄。他画像の確定は保持する。

        以前は per_file_results を全消去していたため、2枚目で乳頭/幅指定を
        やり直すと1枚目の ✓ が消えていた。各スナップショットは独自の
        scale_um_px を持つので、他画像の確定は残してよい。
        """
        before = list(state.per_file_results.keys())
        cur = state.files[state.file_idx].name if state.files else None
        # 現在画像だけ再確定が必要（スケールが変わった）
        removed_cur = False
        if cur and cur in state.per_file_results:
            del state.per_file_results[cur]
            removed_cur = True
        had_series = state.series_results is not None
        if had_series:
            _clear_series_display()
        after = list(state.per_file_results.keys())
        _log.info(
            "invalidate reason=%s cur=%s removed_cur=%s had_series=%s "
            "before=%s after=%s",
            reason, cur, removed_cur, had_series, before, after,
        )
        if before and not after and removed_cur:
            # only current was saved — fine
            pass
        _update_batch_ui()

    def _update_batch_ui(*, refresh_dropdown: bool = True) -> None:
        n = len(state.files)
        saved = _batch_saved_count()
        selected = _selected_count()
        state.batch_mode = n >= 2
        batch_progress.visible = state.batch_mode
        series_ready = state.series_mode and selected >= 2
        if state.batch_mode:
            batch_progress.value = _batch_progress_label()
        # 独立解析では時系列起動ボタンを出さない
        series_btn.visible = series_ready
        series_launch_row.visible = state.series_mode and saved >= 1
        if series_ready:
            label = _safe_text(
                f"Registration (選択 {selected}/{saved})"
            )
            series_launch_btn.text = label
            series_btn.text = label
        _refresh_seg_ok_label()
        if refresh_dropdown:
            _refresh_file_dropdown()
        refresh_series_dates_ui()
        _refresh_case_ui()
        _update_registration_controls()
        prep_confirm_btn.visible = (
            state.series_mode
            and state.files
            and state.scale_um_px is not None
            and state.fovea_xy is not None
            and state.disc_pts
            and len(state.disc_pts) == 2
            and state.mode is None
        )

    def _is_selected(fname: str) -> bool:
        """時系列対象か。既定では準備/解析確定済み画像を対象にする。"""
        return state.series_selected.get(fname, fname in state.per_file_results)

    def _selected_confirmed() -> list[str]:
        return [
            f.name for f in state.files
            if f.name in state.per_file_results and _is_selected(f.name)
        ]

    def _selected_count() -> int:
        return len(_selected_confirmed())

    def _bind_select_toggle(fname: str):
        def _handler(e: ft.ControlEvent) -> None:
            # Rebuilding the dates Column remounts Checkboxes; ignore those events
            # or _update_batch_ui → refresh_series_dates_ui loops (UI 暴走).
            if dates_ui_guard["skip"]:
                return
            state.series_selected[fname] = bool(e.control.value)
            if state.series_results is not None or state.series_review_active:
                _clear_series_display()
                status.value = _safe_text(
                    "対象画像を変更しました — ④ 時系列を再実行してください"
                )
            _clear_reg_cache()
            # Update labels/buttons only — do NOT rebuild dates Column here.
            series_ready = state.series_mode and _selected_count() >= 2
            series_btn.visible = series_ready
            series_launch_row.visible = state.series_mode and _batch_saved_count() >= 1
            if series_ready:
                label = _safe_text(
                    f"Registration (選択 {_selected_count()}/{_batch_saved_count()})"
                )
                series_launch_btn.text = label
                series_btn.text = label
            _update_registration_controls()
            page.update()
        return _handler

    def _snapshot_current(*, prep_only: bool = False) -> dict:
        """現在画像の解析結果と、registration 用の画像固有固定点を保存する。

        prep_only=True: 時系列本線用。mask/stats なしで landmark+scale のみ確定。
        """
        path = state.files[state.file_idx]
        fovea = None
        if state.fovea_xy is not None:
            fovea = (float(state.fovea_xy[0]), float(state.fovea_xy[1]))
        disc_pts = None
        disc_xy = None
        if state.disc_pts and len(state.disc_pts) == 2:
            disc_pts = [(float(p[0]), float(p[1])) for p in state.disc_pts]
            disc_xy = disc_xy_from_pts(disc_pts)
        mask = None
        stats = None
        if not prep_only and state.mask is not None:
            mask = state.mask.copy()
            stats = copy.deepcopy(state.stats)
        return {
            "path": path,
            "filename": path.name,
            "case_id": state.case_id,
            "date": state.series_dates.get(
                path.name, datetime_to_iso(guess_image_datetime(path)),
            ),
            "exam_datetime": state.series_dates.get(path.name),
            "image": state.img_gray.copy(),
            "mask": mask,
            "stats": stats,
            "prep_only": prep_only or (mask is None),
            "fovea_xy": fovea,
            "disc_pts": disc_pts,
            "disc_xy": disc_xy,
            "scale_um_px": state.scale_um_px,
            "calibration_source": state.calibration_source,
            "field_width_mm": state.field_width_mm,
            "foveal_correction_applied": state.foveal_correction_applied,
            "foveal_correction_skipped": state.foveal_correction_skipped,
            "foveal_correction_strength": state.foveal_correction_strength,
        }

    def _row_overlay(row: dict) -> np.ndarray:
        """時系列用: 位置合わせ後 FAF + GA マスク。乳頭計測線は描画しない。"""
        s = row.get("stats") or {}
        fxy = row.get("fovea_xy")
        scale = row.get("scale_um_px")
        mask = row.get("mask")
        img = row["image"]
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        return make_overlay(
            img, mask,
            fovea_xy=fxy,
            nearest_xy=s.get("nearest_xy"),
            disc_pts=None,
            disc_diameter_um=state.disc_diameter_um,
            show_macula=True,
            scale_um_px=scale,
        )

    def _recompute_series_estimates() -> None:
        """進行速度3種（線形回帰）を state.series_results から再計算する。"""
        results = state.series_results or []
        state.estimated_prog = estimate_progression_rate_um_per_year(results)
        state.estimated_growth_mm2_per_year = estimate_growth_mm2_per_year(results)
        state.estimated_growth_mm_sqrt_per_year = estimate_growth_mm_sqrt_per_year(results)
        prog_field.value = str(int(state.estimated_prog)) if state.estimated_prog else ""

    def _rebuild_series_derived(
        *,
        overlays: list[np.ndarray] | None = None,
    ) -> None:
        """state.series_results から overlay / 進行速度を再構築する。

        overlays を渡すと再生成しない（ワーカースレッドで事前計算した結果を
        受け取る用）。GIF はここでは作らず、GIF ビュー表示時に遅延生成する。
        """
        results = state.series_results or []
        if overlays is None:
            overlays = [_row_overlay(row) for row in results]
        state.series_overlays = overlays
        state.series_gif = None  # 次回 GIF ビュー表示時に再生成
        _recompute_series_estimates()

    def _series_gif_bytes() -> bytes | None:
        """GIF ビュー用アニメGIF（表示幅で遅延生成し、キャッシュする）。

        原寸ではなく表示幅（DISPLAY_WIDTH）で生成する: エンコードが
        約9倍速く、UI へ送るペイロードも約1/7 になる。原寸が必要な
        PDF 出力は overlay（原寸）を使うため影響しない。
        """
        if state.series_gif is not None:
            return state.series_gif
        overlays = state.series_overlays
        if not overlays:
            return None
        frames = [
            cv2.resize(
                o,
                (DISPLAY_WIDTH, max(1, int(o.shape[0] * DISPLAY_WIDTH / o.shape[1]))),
                interpolation=cv2.INTER_AREA,
            )
            for o in overlays
        ]
        state.series_gif = overlays_to_gif_bytes(frames)
        return state.series_gif

    def _apply_series_results(
        results: list[dict],
        *,
        landing_filename: str | None = None,
        overlays: list[np.ndarray] | None = None,
    ) -> None:
        state.series_results = results
        _rebuild_series_derived(overlays=overlays)
        series_date_dd.visible = True
        series_date_dd.options = [
            ft.dropdown.Option(
                key=str(i),
                text=_safe_text(
                    f"{_format_series_datetime_label(r['date'])} - {r['filename']}"
                ),
            )
            for i, r in enumerate(results)
        ]
        land = resolve_series_landing_index(results, preferred_filename=landing_filename)
        series_date_dd.value = str(land)
        state.series_time_idx = land
        # Align file_idx; update file_dd under sync guard so users can jump
        # timepoints from the sidebar without exiting series.
        _sync_file_idx_to_series_timepoint(update_dropdown=True)

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
        sources: dict[str, str] = {}
        for f in files:
            key = f.name
            if state.series_dates_manual.get(key) and key in state.series_dates:
                dates[key] = state.series_dates[key]
                sources[key] = "manual"
            elif key in state.series_dates and key in state.series_date_source:
                dates[key] = state.series_dates[key]
                sources[key] = state.series_date_source[key]
            else:
                dt, source = extract_image_datetime(f)
                if dt is not None:
                    dates[key] = datetime_to_iso(dt)
                    sources[key] = source
                else:
                    dates[key] = ""
                    sources[key] = "missing"
        state.series_dates = dates
        state.series_date_source = sources

    def _apply_series_datetime(
        fname: str,
        dt: datetime,
        *,
        manual: bool,
    ) -> None:
        """検査日時を state / 確定スナップショットへ記録。"""
        iso = datetime_to_iso(dt)
        state.series_dates[fname] = iso
        if manual:
            state.series_dates_manual[fname] = True
            state.series_date_source[fname] = "manual"
        sync_date_to_snapshots(state.per_file_results, fname, iso)
        snap = state.per_file_results.get(fname)
        if snap is not None:
            snap["exam_datetime"] = iso
            snap["case_id"] = state.case_id
        _invalidate_series_after_date_edit()
        refresh_series_dates_ui()
        _update_registration_controls()
        # Explicit next-step guidance (日時入力だけでは Registration に進まない)
        if not state.series_results:
            _set_status(_datetime_next_step_message(just_fname=fname))

    def _ensure_selected_series_dates(selected_fnames: set[str]) -> None:
        """Registration 前: 自動推定は手動未設定のみ。全対象は series_dates を同期。"""
        manual_skip = {
            fname for fname in selected_fnames
            if state.series_dates_manual.get(fname)
        }
        refreshed = refresh_series_datetimes_for_files(
            state.files, selected_fnames, skip_manual=manual_skip,
        )
        for fname, (iso, source) in refreshed.items():
            state.series_dates[fname] = iso
            state.series_date_source[fname] = source
            sync_date_to_snapshots(state.per_file_results, fname, iso)
            snap = state.per_file_results.get(fname)
            if snap is not None:
                snap["exam_datetime"] = iso
                snap["case_id"] = state.case_id
        for fname in selected_fnames:
            iso = state.series_dates.get(fname)
            if iso:
                sync_date_to_snapshots(state.per_file_results, fname, iso)

    def _invalidate_series_after_date_edit() -> None:
        if not state.series_results:
            _clear_reg_cache()
            return
        _clear_series_display()
        _clear_reg_cache()
        status.value = "撮影日を変更しました — ④ 時系列を再実行してください"

    def _apply_series_date(fname: str, picked: date) -> None:
        """カレンダーから日付のみ選択（時刻は 00:00）。"""
        _apply_series_datetime(
            fname, datetime.combine(picked, datetime.min.time()), manual=True,
        )

    def _apply_series_datetime_from_text(fname: str, raw: str) -> None:
        dt = parse_user_datetime_input(raw)
        _apply_series_datetime(fname, dt, manual=True)

    def _on_date_picker_change(e: ft.ControlEvent) -> None:
        if date_picker_guard["ignore"]:
            return
        try:
            fname = date_picker_target["fname"]
            val = e.control.value
            if fname and val is not None:
                picked = val.date() if isinstance(val, datetime) else val
                _apply_series_date(fname, picked)
        except Exception as exc:
            _log.exception("date picker apply failed")
            _set_status(f"日時設定エラー: {exc}")
        finally:
            date_picker_target["fname"] = None
            try:
                page.close(date_picker)
            except Exception:
                pass
            page.update()

    def _open_date_picker(fname: str) -> None:
        date_picker_target["fname"] = fname
        iso = state.series_dates.get(fname)
        date_picker_guard["ignore"] = True
        try:
            if iso:
                try:
                    date_picker.value = datetime.fromisoformat(iso).date()
                except ValueError:
                    date_picker.value = None
            else:
                date_picker.value = None
        finally:
            date_picker_guard["ignore"] = False
        page.open(date_picker)

    def _bind_date_field_submit(fname: str):
        def _handler(e: ft.ControlEvent) -> None:
            raw = (e.control.value or "").strip()
            try:
                _apply_series_datetime_from_text(fname, raw)
            except ValueError:
                e.control.error_text = "YYYY-MM-DD または YYYY-MM-DD HH:MM"
                page.update()
                return
            e.control.error_text = None
            page.update()
        return _handler

    def _bind_date_field_confirm(fname: str, field: ft.TextField):
        """Enter 以外でも明示「確定」できるボタン用。"""
        def _handler(_e: ft.ControlEvent) -> None:
            raw = (field.value or "").strip()
            try:
                _apply_series_datetime_from_text(fname, raw)
                field.error_text = None
            except ValueError:
                field.error_text = "YYYY-MM-DD または YYYY-MM-DD HH:MM"
            page.update()
        return _handler

    def _bind_date_picker_open(fname: str):
        def _handler(_e: ft.ControlEvent) -> None:
            _open_date_picker(fname)
        return _handler

    def refresh_series_dates_ui() -> None:
        dates_ui_guard["gen"] += 1
        gen = dates_ui_guard["gen"]
        dates_ui_guard["skip"] = True
        series_dates_col.controls.clear()
        show = state.series_mode and len(state.files) >= 2
        series_dates_section.visible = show
        if not show:
            async def _release_hidden() -> None:
                await asyncio.sleep(0)
                if dates_ui_guard["gen"] == gen:
                    dates_ui_guard["skip"] = False
            page.run_task(_release_hidden)
            return
        init_series_dates(state.files)
        for f in state.files:
            fname = f.name
            iso = state.series_dates.get(fname, "")
            src = state.series_date_source.get(fname, "")
            if state.series_dates_manual.get(fname):
                src = "manual"
            date_ok = is_datetime_source_confirmed(src)
            done = fname in state.per_file_results
            lm_ready = landmarks_ready_for_registration(
                state.per_file_results.get(fname) if done else None
            )
            prep = bool(
                done and (state.per_file_results[fname].get("prep_only")
                          or state.per_file_results[fname].get("mask") is None)
            )
            lm_note = ""
            if done:
                lm = landmarks_from_snapshot(state.per_file_results[fname])
                if lm.get("fovea_xy"):
                    fx, fy = lm["fovea_xy"]
                    lm_note = f"中心窩 ({fx:.0f},{fy:.0f})"
                    if lm.get("disc_xy"):
                        dx, dy = lm["disc_xy"]
                        lm_note += f" · 乳頭 ({dx:.0f},{dy:.0f})"
                if not lm_ready:
                    lm_note = (lm_note + " — " if lm_note else "") + "乳頭+中心窩の確認が必要"
            elif date_ok:
                lm_note = "次: ①乳頭 → ②中心窩 → 準備確定"
            src_label = {
                "exif": "EXIF",
                "filename": "ファイル名",
                "manual": "手動",
                "mtime": "要確認(mtime)",
                "missing": "未入力",
            }.get(src, src or "未入力")
            date_field = ft.TextField(
                value=iso,
                width=150,
                text_size=12,
                hint_text="YYYY-MM-DD HH:MM",
                error_text=(
                    None if date_ok else "日時を入力し「確定」"
                ),
                on_submit=_bind_date_field_submit(fname),
            )
            series_dates_col.controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Row(
                                [
                                    ft.Checkbox(
                                        value=_is_selected(fname),
                                        disabled=not lm_ready,
                                        tooltip=(
                                            "時系列解析の対象に含める"
                                            if lm_ready
                                            else "先に乳頭＋中心窩を確認し準備確定"
                                        ),
                                        on_change=_bind_select_toggle(fname),
                                        scale=0.85,
                                    ),
                                    ft.Icon(
                                        ft.Icons.CHECK_CIRCLE if lm_ready else ft.Icons.RADIO_BUTTON_UNCHECKED,
                                        size=16,
                                        color=T.ACCENT_GREEN if lm_ready else T.TEXT_MUTED,
                                    ),
                                    ft.Text(
                                        fname,
                                        size=10,
                                        color=T.ACCENT_GREEN if lm_ready else T.TEXT_MUTED,
                                        max_lines=2,
                                        overflow=ft.TextOverflow.ELLIPSIS,
                                        expand=True,
                                    ),
                                    ft.Text(
                                        "準備" if prep else ("確定" if done else ""),
                                        size=9,
                                        color=T.ACCENT_CYAN if prep else T.ACCENT_GREEN,
                                    ) if done else ft.Container(width=0),
                                    ft.Text(
                                        src_label,
                                        size=9,
                                        color=T.ACCENT_GREEN if date_ok else T.ACCENT_AMBER,
                                    ),
                                ],
                                spacing=4,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                            ft.Text(
                                lm_note,
                                size=9,
                                color=T.ACCENT_CYAN if lm_note else T.TEXT_MUTED,
                            ) if lm_note else ft.Container(height=0),
                            ft.Row(
                                [
                                    date_field,
                                    ft.TextButton(
                                        "確定",
                                        tooltip="日時を確定（次のステップへ）",
                                        on_click=_bind_date_field_confirm(fname, date_field),
                                        style=ft.ButtonStyle(padding=4),
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

        async def _release_dates_guard() -> None:
            # Hold through deferred Flet web on_change after page.update.
            await asyncio.sleep(0)
            await asyncio.sleep(0.35)
            if dates_ui_guard["gen"] == gen:
                dates_ui_guard["skip"] = False

        page.run_task(_release_dates_guard)

    date_picker.on_change = _on_date_picker_change

    def _refresh_scale_info() -> None:
        if state.scale_um_px is None:
            scale_info.value = ""
            return
        src = calibration.calibration_source_label(state.calibration_source)
        scale_info.value = _safe_text(
            f"スケール ({src} / {state.calibration_source or '-'}): "
            f"{state.scale_um_px:.2f} um/px"
        )

    def _current_scale_method() -> str:
        return scale_method_dd.value or state.scale_method or "dfd"

    def _disc_diameter_um() -> float:
        try:
            return float(disc_field.value or state.disc_diameter_um)
        except ValueError:
            return float(state.disc_diameter_um)

    def _try_apply_dfd_scale(*, allow_disc_fallback: bool = True) -> bool:
        """乳頭2点＋中心窩から DFD スケールを設定。失敗時は乳頭径へフォールバック可。"""
        if state.fovea_xy is None or not state.disc_pts or len(state.disc_pts) != 2:
            return False
        disc_xy = disc_xy_from_pts(state.disc_pts)
        if disc_xy is None:
            return False
        try:
            state.scale_um_px = calibration.compute_scale_um_per_px_from_dfd(
                state.fovea_xy, disc_xy,
            )
            state.calibration_source = calibration.source_id_dfd()
            state.scale_method = "dfd"
            return True
        except ValueError:
            if not allow_disc_fallback:
                return False
            try:
                p1, p2 = state.disc_pts
                disc_um = _disc_diameter_um()
                state.disc_diameter_um = disc_um
                state.scale_um_px = calibration.compute_scale_um_per_px(
                    p1, p2, disc_um,
                )
                state.calibration_source = calibration.source_id_disc(disc_um)
                state.scale_method = "disc"
                scale_method_dd.value = "disc"
                return True
            except ValueError:
                return False

    def _apply_disc_scale_from_pts() -> bool:
        if not state.disc_pts or len(state.disc_pts) != 2:
            return False
        p1, p2 = state.disc_pts
        try:
            disc_um = _disc_diameter_um()
            state.disc_diameter_um = disc_um
            state.scale_um_px = calibration.compute_scale_um_per_px(p1, p2, disc_um)
            state.calibration_source = calibration.source_id_disc(disc_um)
            state.scale_method = "disc"
            return True
        except ValueError:
            return False

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
            if _current_scale_method() == "dfd":
                disc_info.value = _safe_text(
                    f"乳頭線: {d_px:.0f} px（中点→固定点）· "
                    f"スケールは中心窩指定後に DFD 4.76mm"
                )
            else:
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
        state.calibration_source = calibration.SOURCE_WIDTH
        state.field_width_mm = field_mm
        _refresh_scale_info()
        return True

    def _foveal_prereq_error() -> str | None:
        if state.fovea_xy is None:
            return "中心窩が未指定です"
        raw = state.img_gray_raw if state.img_gray_raw is not None else state.img_gray
        if raw is None:
            return "画像が読み込まれていません"
        if state.scale_um_px is None:
            if (
                _current_scale_method() == "dfd"
                and _try_apply_dfd_scale()
            ):
                _refresh_scale_info()
            elif not _try_apply_width_scale():
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
            raw,
            use_clahe=state.use_clahe,
            clahe_clip=state.clahe_clip,
            fovea_xy=state.fovea_xy,
            scale_um_px=state.scale_um_px,
        )
        mask_corr = segmentation.segment_ga(
            seg_gray,
            use_clahe=state.use_clahe,
            clahe_clip=state.clahe_clip,
            fovea_xy=state.fovea_xy,
            scale_um_px=state.scale_um_px,
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
            fovea_xy=state.fovea_xy,
            scale_um_px=state.scale_um_px,
        )

    def _finalize_mask(mask: np.ndarray) -> np.ndarray:
        """編集後マスクから孤立微小粒子を除去。"""
        return segmentation.cleanup_mask_after_edit(mask)

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

    def _restore_from_snapshot(snap: dict) -> None:
        """確定スナップショットを作業画面へ復元（中心窩・乳頭含む）。"""
        state.img_gray = snap["image"].copy()
        mask = snap.get("mask")
        state.mask = mask.copy() if mask is not None else None
        state.stats = copy.deepcopy(snap.get("stats"))
        lm = landmarks_from_snapshot(snap)
        state.fovea_xy = lm["fovea_xy"]
        state.disc_pts = list(lm["disc_pts"]) if lm["disc_pts"] else []
        if snap.get("scale_um_px") is not None:
            state.scale_um_px = snap["scale_um_px"]
        if snap.get("calibration_source") is not None:
            state.calibration_source = snap["calibration_source"]
            src = state.calibration_source or ""
            if src.startswith("dfd"):
                state.scale_method = "dfd"
                scale_method_dd.value = "dfd"
            elif src == "disc" or src.startswith("disc_"):
                state.scale_method = "disc"
                scale_method_dd.value = "disc"
        if snap.get("field_width_mm") is not None:
            state.field_width_mm = snap["field_width_mm"]
        if "foveal_correction_applied" in snap:
            state.foveal_correction_applied = snap["foveal_correction_applied"]
        if "foveal_correction_skipped" in snap:
            state.foveal_correction_skipped = snap["foveal_correction_skipped"]
        if snap.get("foveal_correction_strength") is not None:
            state.foveal_correction_strength = snap["foveal_correction_strength"]
        state.mode = None

    def load_image_at(
        idx: int,
        *,
        new_folder: bool = False,
        force: bool = False,
    ) -> None:
        if not state.files:
            return
        # Never replace the working image while a series view is active or
        # mid-launch — that is how 3/3 confirmation snapped back to image 1.
        # force=True is for intentional exits (e.g. reanalyze this timepoint).
        if not force and (state.series_results is not None or series_guard["launching"]):
            return
        state.file_idx = max(0, min(idx, len(state.files) - 1))
        path = state.files[state.file_idx]
        fname = path.name
        snap = state.per_file_results.get(fname)
        if new_folder:
            state.reset_analysis(keep_series=True)
            _clear_series_display()
            _clear_reg_cache()
        else:
            state.reset_per_image()
        if snap and not force and not new_folder:
            # 確定済み画像: 保存時の中心窩・乳頭・マスクを復元して表示
            _restore_from_snapshot(snap)
            state.img_full = state.img_gray.copy()
        else:
            gray = load_gray(path)
            state.img_full = gray
            state.img_gray = gray.copy()
            state.crop_box = None
            state.crop_pts = []
        _refresh_image_width_label()
        _refresh_scale_info()
        _refresh_file_dropdown()
        refresh_image()
        refresh_results()
        _update_batch_ui()

    def _save_batch_result(*, prep_only: bool = False) -> None:
        """現在画像をメモリへ保存。prep_only なら landmark+scale のみ（時系列本線）。"""
        if not state.files:
            return
        if state.fovea_xy is None or state.scale_um_px is None:
            _log.warning(
                "save skipped: fovea=%s scale=%s for %s",
                state.fovea_xy is not None, state.scale_um_px is not None,
                state.files[state.file_idx].name,
            )
            return
        if not prep_only and (state.stats is None or state.mask is None):
            _log.warning(
                "save skipped: stats=%s mask=%s files=%s",
                state.stats is not None, state.mask is not None, bool(state.files),
            )
            return
        fname = state.files[state.file_idx].name
        snap = _snapshot_current(prep_only=prep_only)
        state.per_file_results[fname] = snap
        state.stash_current_case_results()
        _prune_stale_batch_results()
        _clear_series_display()
        lm = landmarks_from_snapshot(snap)
        _log.info(
            "saved %s case=%s prep=%s fovea=%s disc=%s → confirmed=%s",
            fname, state.case_id, prep_only, lm["fovea_xy"], lm["disc_xy"],
            list(state.per_file_results.keys()),
        )

    def on_prep_confirm(_=None) -> None:
        """時系列モード: ランドマーク+スケールを準備確定（seg は Registration 後）。"""
        if not state.series_mode:
            _set_status("時系列モードを ON にしてください")
            page.update()
            return
        if state.scale_um_px is None or state.fovea_xy is None:
            _set_status("先に乳頭/スケールと中心窩を完了してください")
            page.update()
            return
        if not state.disc_pts or len(state.disc_pts) != 2:
            _set_status(
                "Registration には各画像の乳頭確認が必要です — 「① 乳頭」で端点を指定"
            )
            page.update()
            return
        if state.img_gray is None:
            _set_status("画像を読み込んでください")
            page.update()
            return
        fname = state.files[state.file_idx].name
        src = state.series_date_source.get(fname)
        if state.series_dates_manual.get(fname):
            src = "manual"
        if not is_datetime_source_confirmed(src):
            _set_status(
                "検査日時が未確定です — サイドバーで日時を入力し「確定」してから準備確定"
            )
            page.update()
            return
        _save_batch_result(prep_only=True)
        state.series_selected.setdefault(fname, True)
        saved = _batch_saved_count()
        selected = _selected_count()
        n = len(state.files)
        lm_left = sum(
            1 for f in state.files
            if not landmarks_ready_for_registration(state.per_file_results.get(f.name))
        )
        if lm_left > 0:
            next_msg = f" — 「次へ ▶」で未準備 {lm_left}/{n} 枚の乳頭＋中心窩を確認"
        elif not _selected_dates_ready():
            next_msg = " — 残り画像の日時を確定してから Registration"
        elif selected >= 2:
            next_msg = " — ④ Registration を押してください"
        else:
            next_msg = " — 対象を2枚以上選択してください"
        _set_status(
            _safe_text(
                f"準備確定 {saved}/{n} · 対象 {selected} 枚（乳頭＋中心窩確認済）"
                f"{next_msg}"
            )
        )
        _update_batch_ui()
        page.update()

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
        _restore_from_snapshot(snap)
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
        rows = []
        for row in state.series_results:
            r = dict(row)
            r.setdefault("case_id", state.case_id)
            r.setdefault("exam_datetime", r.get("date"))
            rows.append(r)
        path.write_bytes(build_batch_csv_bytes(
            rows,
            roi_qc=state.series_roi_qc,
        ))
        status.value = f"CSV 保存: {path}"
        page.update()

    def _roi_qc_note_for_row(row: dict) -> str | None:
        """PDF 用のマスクQC補正1行サマリ（未適用時は None）。"""
        if "roi_raw_area" not in row:
            return None
        method = row.get("roi_refine_method")
        if method:
            note = (
                f"corrected ({method}): {row['roi_raw_area']}"
                f" -> {row['roi_corrected_area']} px"
            )
        else:
            note = "no anomaly (mask unchanged)"
        if row.get("roi_needs_review"):
            note += f"; needs review: {row.get('roi_reason') or ''}"
        return note

    def _resolve_pdf_export_context() -> dict | None:
        """PDF 出力用の filename / stats / overlay / GAWAIN 生成コンテキスト。"""
        if state.series_results and state.series_overlays:
            row = state.series_results[state.series_time_idx]
            overlay = state.series_overlays[state.series_time_idx]
            snap = state.per_file_results.get(row["filename"], row)
            return {
                "roi_qc_note": _roi_qc_note_for_row(row),
                "filename": row["filename"],
                "date": row["date"],
                "exam_datetime": row.get("exam_datetime") or row.get("date"),
                "case_id": row.get("case_id") or state.case_id,
                "stats": row["stats"],
                "overlay_rgb": overlay,
                "image": row.get("image") or snap.get("image"),
                "mask": row.get("mask") or snap.get("mask"),
                "fovea_xy": row.get("fovea_xy") or snap.get("fovea_xy"),
                "scale_um_px": row.get("scale_um_px") or snap.get("scale_um_px"),
                "calibration_source": snap.get("calibration_source"),
                "field_width_mm": snap.get("field_width_mm"),
                "foveal_correction_applied": snap.get("foveal_correction_applied"),
                "foveal_correction_skipped": snap.get("foveal_correction_skipped"),
                "foveal_correction_strength": snap.get("foveal_correction_strength"),
            }
        if state.stats is not None and state.mask is not None and state.img_gray is not None:
            fname = state.files[state.file_idx].name if state.files else "image"
            snap = state.per_file_results.get(fname)
            exam_dt = (
                (snap or {}).get("exam_datetime")
                or state.series_dates.get(fname, "")
            )
            if snap:
                return {
                    "filename": fname,
                    "date": snap.get("date") or exam_dt,
                    "exam_datetime": exam_dt,
                    "case_id": snap.get("case_id") or state.case_id,
                    "stats": state.stats,
                    "overlay_rgb": _build_rgb_frame(),
                    "image": snap.get("image", state.img_gray),
                    "mask": snap.get("mask", state.mask),
                    "fovea_xy": snap.get("fovea_xy", state.fovea_xy),
                    "scale_um_px": snap.get("scale_um_px", state.scale_um_px),
                    "calibration_source": snap.get("calibration_source", state.calibration_source),
                    "field_width_mm": snap.get("field_width_mm", state.field_width_mm),
                    "foveal_correction_applied": snap.get(
                        "foveal_correction_applied", state.foveal_correction_applied,
                    ),
                    "foveal_correction_skipped": snap.get(
                        "foveal_correction_skipped", state.foveal_correction_skipped,
                    ),
                    "foveal_correction_strength": snap.get(
                        "foveal_correction_strength", state.foveal_correction_strength,
                    ),
                }
            return {
                "filename": fname,
                "date": exam_dt,
                "exam_datetime": exam_dt,
                "case_id": state.case_id,
                "stats": state.stats,
                "overlay_rgb": _build_rgb_frame(),
                "image": state.img_gray,
                "mask": state.mask,
                "fovea_xy": state.fovea_xy,
                "scale_um_px": state.scale_um_px,
                "calibration_source": state.calibration_source,
                "field_width_mm": state.field_width_mm,
                "foveal_correction_applied": state.foveal_correction_applied,
                "foveal_correction_skipped": state.foveal_correction_skipped,
                "foveal_correction_strength": state.foveal_correction_strength,
            }
        return None

    def on_export_pdf(_=None) -> None:
        ctx = _resolve_pdf_export_context()
        if ctx is None:
            status.value = "保存する解析結果がありません"
            page.update()
            return

        async def _export_pdf_task() -> None:
            status.value = "PDF 作成中…"
            page.update()
            try:
                gawain_png = await asyncio.to_thread(
                    gawain_figure_png_from_context, ctx,
                )
                pdf_bytes = await asyncio.to_thread(
                    generate_pdf_report_bytes,
                    filename=ctx["filename"],
                    date=ctx["date"],
                    stats=ctx["stats"],
                    overlay_rgb=ctx["overlay_rgb"],
                    gawain_figure_png=gawain_png,
                    scale_um_px=ctx.get("scale_um_px"),
                    calibration_source=ctx.get("calibration_source"),
                    field_width_mm=ctx.get("field_width_mm"),
                    fovea_xy=ctx.get("fovea_xy"),
                    foveal_correction_applied=ctx.get("foveal_correction_applied"),
                    foveal_correction_skipped=ctx.get("foveal_correction_skipped"),
                    foveal_correction_strength=ctx.get("foveal_correction_strength"),
                    roi_qc_note=ctx.get("roi_qc_note"),
                    case_id=ctx.get("case_id"),
                    exam_datetime=ctx.get("exam_datetime"),
                )
                out = _ensure_output_dir()
                path = out / suggested_pdf_filename(ctx["filename"])
                path.write_bytes(pdf_bytes)
                if page.web:
                    status.value = f"PDF ダウンロード: {path}"
                else:
                    pdf_export_pending["bytes"] = pdf_bytes
                    pdf_picker.save_file(
                        dialog_title="PDF を保存",
                        file_name=path.name,
                        file_type=ft.FilePickerFileType.CUSTOM,
                        allowed_extensions=["pdf"],
                        initial_directory=str(out),
                    )
                    status.value = f"PDF 保存: {path}"
            except Exception as exc:
                _log.exception("PDF export failed")
                status.value = f"PDF エラー: {exc}"
            finally:
                page.update()

        page.run_task(_export_pdf_task)

    def on_reanalyze_timepoint(_=None) -> None:
        if not state.series_results:
            return
        row = state.series_results[state.series_time_idx]
        fname = row["filename"]
        state.per_file_results.pop(fname, None)
        _clear_series_display()
        series_guard["launching"] = False
        for i, f in enumerate(state.files):
            if f.name == fname:
                load_image_at(i, force=True)
                break
        status.value = f"{fname} — 再解析してください（③ 解析）"
        on_analyze()
        _update_batch_ui()
        page.update()

    def refresh_image(*, do_update: bool = True) -> None:
        if state.series_review_active and state.series_results:
            rgb = _review_display_rgb()
            img_ctrl.visible = True
            gif_img.visible = False
            graph_area.visible = False
            if rgb is not None:
                _push_image_rgb(rgb, do_update=do_update)
            return
        if state.series_roi_review_active and state.series_results:
            results = state.series_results
            idx = max(0, min(state.series_review_idx, len(results) - 1))
            rgb = (
                state.series_overlays[idx]
                if state.series_overlays
                else _row_overlay(results[idx])
            )
            img_ctrl.visible = True
            gif_img.visible = False
            graph_area.visible = False
            _push_image_rgb(rgb, do_update=do_update)
            return
        if state.view_mode == "gif" and state.series_results:
            gif = _series_gif_bytes()
            if gif:
                gif_img.src_base64 = base64.b64encode(gif).decode("ascii")
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

    _GRAPH_METRICS = {
        "area_mm2": ("GA Area (mm2)", lambda s: s.get("area_mm2"), T.ACCENT_RED),
        "area_mm_sqrt": ("SQRT GA Area (mm)", lambda s: s.get("area_mm_sqrt"), "#F59E0B"),
        "gawain": (_safe_text(GAWAIN_LABEL), lambda s: s.get("gawain"), T.ACCENT_AMBER),
        "dist_um": ("Distance to fovea (um)", lambda s: s.get("dist_um"), T.ACCENT_CYAN),
    }

    def _format_series_datetime_label(iso: str) -> str:
        """グラフ・表用の日時ラベル。"""
        try:
            dt = parse_series_datetime(iso)
        except ValueError:
            return iso
        if dt.hour or dt.minute or dt.second:
            return dt.strftime("%Y-%m-%d %H:%M")
        return dt.strftime("%Y-%m-%d")

    def _series_x_values(results: list[dict]) -> list[float]:
        """各時点の X 座標（1970-01-01 からの経過日数）。"""
        xs: list[float] = []
        for r in results:
            try:
                xs.append(datetime_to_plot_x(parse_series_datetime(r["date"])))
            except ValueError:
                xs.append(float(len(xs)))
        return xs

    def _plot_x_to_datetime(x: float) -> datetime:
        return datetime.fromtimestamp(x * 86400.0)

    def _nice_date_axis_ticks(
        x_min: float,
        x_max: float,
        *,
        max_ticks: int = 6,
        show_time: bool = False,
    ) -> list[tuple[float, str]]:
        """X 軸目盛を日時ラベル付きで生成（連続スケール）。"""
        if x_min == x_max:
            dt = _plot_x_to_datetime(x_min)
            return [(x_min, dt.strftime("%Y-%m-%d %H:%M" if show_time else "%Y-%m-%d"))]
        span = x_max - x_min
        raw_step = span / max(max_ticks - 1, 1)
        mag = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 1.0
        step = raw_step
        for mult in (1, 2, 2.5, 5, 10):
            candidate = mult * mag
            if candidate >= raw_step:
                step = candidate
                break
        lo = math.floor(x_min / step) * step
        fmt = "%Y-%m-%d %H:%M" if show_time or span < 2.0 else "%Y-%m-%d"
        ticks: list[tuple[float, str]] = []
        x = lo
        while x <= x_max + step * 0.5:
            if x >= x_min - step * 0.01:
                ticks.append((x, _plot_x_to_datetime(x).strftime(fmt)))
            x += step
        if len(ticks) < 2:
            ticks = [
                (x_min, _plot_x_to_datetime(x_min).strftime(fmt)),
                (x_max, _plot_x_to_datetime(x_max).strftime(fmt)),
            ]
        return ticks

    def _graph_show_time(results: list[dict]) -> bool:
        for r in results:
            try:
                dt = parse_series_datetime(r["date"])
                if dt.hour or dt.minute or dt.second:
                    return True
            except ValueError:
                continue
        return False

    def _nice_axis_range(vmin: float, vmax: float) -> tuple[float, float, float]:
        """Y軸の min/max/目盛間隔を値に応じて決める（0 起点にしない）。"""
        if vmin == vmax:
            pad = abs(vmin) * 0.1 or 1.0
            vmin, vmax = vmin - pad, vmax + pad
        else:
            pad = (vmax - vmin) * 0.12
            vmin, vmax = vmin - pad, vmax + pad
        span = vmax - vmin
        raw_step = span / 4.0
        mag = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 1.0
        for mult in (1, 2, 2.5, 5, 10):
            step = mult * mag
            if step >= raw_step:
                break
        lo = math.floor(vmin / step) * step
        hi = math.ceil(vmax / step) * step
        return lo, hi, step

    def _refresh_graph() -> None:
        graph_area.controls.clear()
        results = state.series_results or []
        graph_area.controls.append(graph_metric_dd)
        if not results:
            graph_area.controls.append(
                ft.Text("時系列データがありません", size=12, color=T.TEXT_MUTED)
            )
            return

        metric = state.graph_metric
        label, getter, color = _GRAPH_METRICS.get(metric, _GRAPH_METRICS["area_mm2"])
        xs = _series_x_values(results)

        pts, table_rows = [], []
        for x, r in zip(xs, results):
            v = getter(r["stats"])
            if v is not None:
                pts.append((x, float(v)))
            table_rows.append((_format_series_datetime_label(r["date"]), v))

        if len(pts) < 1:
            graph_area.controls.append(
                ft.Text(f"{label}: 有効な値がありません", size=12, color=T.TEXT_MUTED)
            )
        else:
            ys = [p[1] for p in pts]
            lo, hi, step = _nice_axis_range(min(ys), max(ys))
            x_min = min(x for x, _ in pts)
            x_max = max(x for x, _ in pts)
            if x_min == x_max:
                x_min -= 0.5
                x_max += 0.5
            data = ft.LineChartData(
                data_points=[ft.LineChartDataPoint(x, y) for x, y in pts],
                stroke_width=2,
                color=color,
                curved=False,
                stroke_cap_round=True,
                point=True,
            )
            y_labels = []
            v = lo
            while v <= hi + step * 0.5:
                y_labels.append(ft.ChartAxisLabel(
                    value=v,
                    label=ft.Text(f"{v:.2f}".rstrip("0").rstrip("."), size=10, color=T.TEXT_MUTED),
                ))
                v += step
            x_ticks = _nice_date_axis_ticks(
                x_min, x_max, show_time=_graph_show_time(results),
            )
            x_labels = [
                ft.ChartAxisLabel(
                    value=x,
                    label=ft.Text(_safe_text(lbl), size=9, color=T.TEXT_MUTED),
                )
                for x, lbl in x_ticks
            ]
            # 各データ点にも日時ラベル（目盛と重ならない位置）
            for x, r in zip(xs, results):
                x_labels.append(ft.ChartAxisLabel(
                    value=x,
                    label=ft.Text(
                        _safe_text(_format_series_datetime_label(r["date"])),
                        size=8,
                        color=color,
                    ),
                ))
            chart = ft.LineChart(
                data_series=[data],
                min_y=lo, max_y=hi,
                min_x=x_min, max_x=x_max,
                left_axis=ft.ChartAxis(labels=y_labels, labels_size=44, title=ft.Text(label, size=11, color=T.TEXT), title_size=18),
                bottom_axis=ft.ChartAxis(
                    labels=x_labels,
                    labels_size=36,
                    title=ft.Text("撮影日時", size=11, color=T.TEXT),
                    title_size=18,
                ),
                tooltip_bgcolor=T.PANEL_BG,
                border=ft.border.all(1, "#374151"),
                horizontal_grid_lines=ft.ChartGridLines(interval=step, color="#374151", width=1),
                expand=False,
                height=320,
                width=DISPLAY_WIDTH,
            )
            graph_area.controls.append(ft.Text(label, size=13, weight=ft.FontWeight.W_600, color=color))
            graph_area.controls.append(chart)

        fmt = "{:.0f}" if metric in ("dist_um", "gawain") else "{:.3f}"
        rows = [
            ft.DataRow(cells=[
                ft.DataCell(ft.Text(_safe_text(d))),
                ft.DataCell(ft.Text(fmt.format(v) if v is not None else "-")),
            ])
            for d, v in table_rows
        ]
        graph_area.controls.append(
            ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Date")),
                    ft.DataColumn(ft.Text(_safe_text(label))),
                ],
                rows=rows,
                border=ft.border.all(1, "#374151"),
            )
        )

    def refresh_results() -> None:
        results_col.controls.clear()
        stats = state.stats
        if state.series_results and (
            state.series_review_active or state.series_roi_review_active
        ):
            idx = max(0, min(state.series_review_idx, len(state.series_results) - 1))
            stats = state.series_results[idx]["stats"]
        elif state.series_results and state.view_mode == "work":
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
        if is_series and state.series_results:
            for alert in series_metric_regressions(state.series_results):
                results_col.controls.append(ft.Text(
                    _safe_text(f"⚠ {alert}"),
                    color=T.ACCENT_AMBER,
                    size=12,
                ))
        if is_series and state.series_roi_applied and state.series_results:
            corrected = [
                r for r in state.series_results if r.get("roi_refine_method")
            ]
            if corrected:
                labels = ", ".join(
                    f"{_format_series_datetime_label(r['date'])}"
                    f" ({r['roi_refine_method']})"
                    for r in corrected
                )
                results_col.controls.append(ft.Text(
                    _safe_text(f"マスクQC補正済み: {labels}"),
                    color=T.ACCENT_CYAN,
                    size=11,
                ))
            for r in state.series_results:
                if r.get("roi_needs_review"):
                    reason = r.get("roi_reason") or "面積異常"
                    results_col.controls.append(ft.Text(
                        _safe_text(
                            f"⚠ QC要確認 "
                            f"{_format_series_datetime_label(r['date'])}: {reason}"
                        ),
                        color=T.ACCENT_AMBER,
                        size=12,
                    ))
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
        if (
            state.series_results
            and not state.series_review_active
            and not state.series_roi_review_active
        ):
            n_tp = len(state.series_results)
            tp = min(state.series_time_idx, n_tp - 1) + 1
            n_files = len(state.files)
            saved = _batch_saved_count()
            cur = state.series_results[min(state.series_time_idx, n_tp - 1)]
            reg_note = (
                "位置合わせ済"
                if state.series_registered
                else "位置合わせなし（各画像の座標のまま）"
            )
            results_col.controls.append(ft.Text(
                _safe_text(
                    f"時系列 時点 {tp}/{n_tp}"
                    f" · 確定 {saved}/{n_files}"
                    f" · {reg_note}"
                    f" · {cur['date']} {cur['filename']}"
                ),
                size=12,
                color=T.PRIMARY,
            ))
            if not state.series_registered:
                results_col.controls.append(ft.Text(
                    _safe_text(
                        "位置合わせが必要な場合は左のトグルを ON にして"
                        "「位置合わせを実行」を押してください。"
                    ),
                    size=11,
                    color=T.TEXT_MUTED,
                ))
            if saved < n_files:
                results_col.controls.append(ft.Text(
                    _safe_text(
                        f"未確定 {n_files - saved} 枚あり。"
                        "「次の未解析へ」またはサイドバーで画像を選んで解析を続けてください。"
                    ),
                    size=11,
                    color=T.ACCENT_AMBER,
                ))
                results_col.controls.append(
                    btn_primary("次の未解析へ ▶", on_click=on_continue_unanalyzed),
                )
            results_col.controls.append(
                ft.Row(
                    [
                        btn_toolbar("◀ 前の時点", on_click=on_series_prev),
                        btn_toolbar("次の時点 ▶", on_click=on_series_next),
                    ],
                    spacing=8,
                    wrap=True,
                )
            )
            results_col.controls.append(
                btn_toolbar("↻ この時点を再解析", on_click=on_reanalyze_timepoint),
            )
            results_col.controls.append(
                ft.Row(
                    [
                        btn_toolbar("CSV 保存", on_click=on_export_csv),
                        btn_toolbar("PDF ダウンロード", on_click=on_export_pdf),
                    ],
                    spacing=8,
                    wrap=True,
                )
            )
        elif stats and state.mode is None and not state.series_results:
            results_col.controls.append(
                btn_toolbar("PDF ダウンロード", on_click=on_export_pdf),
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
            "seg_review": "セグメンテーション確認 - Erase / Draw / 範囲消去・範囲外消去で修正、OK で確定",
            "seg_erase": "長押しで削除 (緑の内外どちらでも可) - やり直しで取消",
            "seg_draw": "長押しで暗部を追加 (緑の内外どちらでも可) - やり直しで取消",
            "seg_clear": (
                "範囲外消去: 残す範囲をフリーハンドで囲む"
                if edit_state.get("clear_outside")
                else "範囲消去: フリーハンドで囲んで指定"
            ),
            "seg_clear_confirm": (
                "範囲外のマスクを消去します - OK または やり直し"
                if edit_state.get("clear_outside")
                else "範囲内のマスクを消去します - OK または やり直し"
            ),
        }
        hint.value = _safe_text(msgs.get(state.mode or "", ""))
        if state.series_roi_review_active:
            hint.value = "ROI補正確認 — 要レビュー時点で OK（採用）または 補正破棄"
        elif not state.mode and modality_dd.value == "hra" and state.scale_um_px is None:
            hint.value = "HRA: 乳頭が見えない場合は「幅指定」でスケール設定"
        hint.color = T.PRIMARY if state.mode or hint.value else T.TEXT_MUTED

    def _load_cases_from_folder(folder: Path) -> None:
        """サブフォルダ→症例、ルート画像→未分類。現在症例の files をセット。"""
        cases = discover_cases(folder)
        state.cases = cases
        state.case_results = {}
        state.per_file_results = {}
        state.case_move_selected = {}
        ids = ordered_case_ids(cases)
        if not ids:
            state.case_id = UNCLASSIFIED_CASE_ID
            state.files = []
            return
        # 未分類より名前付き症例を優先表示
        prefer = next((c for c in ids if c != UNCLASSIFIED_CASE_ID), ids[0])
        state.case_id = prefer
        state.files = list(cases.get(prefer) or [])

    def _refresh_case_ui() -> None:
        ids = ordered_case_ids(state.cases) if state.cases else []
        case_dd.options = [
            ft.dropdown.Option(
                key=cid,
                text=_safe_text(f"{cid} ({len(state.cases.get(cid) or [])})"),
            )
            for cid in ids
        ]
        case_dd.value = state.case_id if state.case_id in ids else (
            ids[0] if ids else None
        )
        # 移動先は現在症例以外（同症例への移動は無意味）
        dest_ids = [cid for cid in ids if cid != state.case_id]
        move_case_dd.options = [
            ft.dropdown.Option(key=cid, text=_safe_text(cid)) for cid in dest_ids
        ]
        if move_case_dd.value not in dest_ids:
            move_case_dd.value = dest_ids[0] if dest_ids else None
        _refresh_case_move_list()

    def _bind_case_move_toggle(fname: str):
        def _handler(e: ft.ControlEvent) -> None:
            state.case_move_selected[fname] = bool(e.control.value)
            page.update()
        return _handler

    def _refresh_case_move_list() -> None:
        """現在症例の画像チェックリストを再構築。"""
        valid = {f.name for f in state.files}
        state.case_move_selected = {
            k: v for k, v in state.case_move_selected.items() if k in valid
        }
        case_move_list_col.controls.clear()
        if not state.files:
            case_move_list_col.controls.append(
                ft.Text("（画像なし）", size=11, color=T.TEXT_MUTED),
            )
            return
        for f in state.files:
            fname = f.name
            case_move_list_col.controls.append(
                ft.Row(
                    [
                        ft.Checkbox(
                            value=bool(state.case_move_selected.get(fname)),
                            on_change=_bind_case_move_toggle(fname),
                            scale=0.85,
                        ),
                        ft.Text(
                            _safe_text(fname),
                            size=11,
                            color=T.TEXT,
                            max_lines=1,
                            overflow=ft.TextOverflow.ELLIPSIS,
                            expand=True,
                        ),
                    ],
                    spacing=4,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                )
            )

    def _switch_case(new_case_id: str, *, load_first: bool = True) -> None:
        if not new_case_id or new_case_id == state.case_id:
            return
        if new_case_id not in state.cases:
            _set_status(f"症例がありません: {new_case_id}")
            return
        state.stash_current_case_results()
        _clear_series_display()
        _clear_reg_cache()
        state.case_id = new_case_id
        state.restore_case_results(new_case_id)
        state.files = list(state.cases.get(new_case_id) or [])
        state.file_idx = 0
        state.series_selected = {}
        state.case_move_selected = {}
        init_series_dates(state.files)
        if state.files and load_first:
            load_image_at(0, force=True)
        elif not state.files:
            state.img_gray = None
            refresh_image()
        _set_status(
            _safe_text(
                f"症例「{new_case_id}」: {len(state.files)} 枚"
                + ("（時系列モード）" if state.series_mode else "（独立解析）")
            )
        )
        _update_batch_ui()

    def on_load_files(_=None) -> None:
        state.input_dir = input_field.value.strip()
        state.output_dir = output_field.value.strip()
        folder = Path(state.input_dir)
        _load_cases_from_folder(folder)
        init_series_dates(state.files)
        refresh_series_dates_ui()
        n_cases = len(state.cases)
        n_files = sum(len(v) for v in state.cases.values())
        if state.files:
            load_image_at(0, new_folder=True)
            _set_status(
                _safe_text(
                    f"{n_cases} 症例 / {n_files} 枚 "
                    f"（現在: {state.case_id} · {len(state.files)} 枚）"
                )
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

    def _goto_series_timepoint(i: int, *, do_update: bool = True) -> None:
        if not state.series_results:
            return
        i = max(0, min(i, len(state.series_results) - 1))
        if state.series_time_idx != i:
            state.series_time_idx = i
        series_date_dd.value = str(i)
        fname = state.series_results[i]["filename"]
        state.file_idx = next(
            (j for j, f in enumerate(state.files) if f.name == fname),
            state.file_idx,
        )
        # Keep sidebar file dropdown label in sync without firing load_image.
        if file_dd.value != fname:
            gen = _begin_dropdown_sync()
            file_dd.value = fname
            _release_dropdown_sync(gen)
        refresh_image(do_update=False)
        refresh_results()
        if do_update:
            page.update()

    def on_series_prev(_=None) -> None:
        if not state.series_results:
            return
        _goto_series_timepoint(state.series_time_idx - 1)

    def on_series_next(_=None) -> None:
        if not state.series_results:
            return
        _goto_series_timepoint(state.series_time_idx + 1)

    def on_continue_unanalyzed(_=None) -> None:
        """時系列表示中でも未解析画像へ進める（部分確定バッチ用）。"""
        nxt = _first_unsaved_idx()
        if nxt is None:
            _set_status("未解析の画像はありません")
            page.update()
            return
        _clear_series_display()
        series_guard["launching"] = False
        load_image_at(nxt, force=True)
        saved = _batch_saved_count()
        n = len(state.files)
        _set_status(
            f"画像 {nxt + 1}/{n} · 確定済み {saved}/{n}"
            f" — 解析を続けてください: {state.files[nxt].name}"
        )
        _update_batch_ui()
        page.update()

    def on_file_change(e) -> None:
        action, payload = resolve_file_dropdown_change(
            skip=dd_sync["skip"],
            selected=e.control.value,
            series_results=state.series_results,
            files=state.files,
            file_idx=state.file_idx,
            series_launching=series_guard["launching"],
        )
        if action == "ignore" or payload is None:
            return
        if action == "series_timepoint":
            _goto_series_timepoint(payload)
            return
        # load_image: may exit an open series when user picks an unconfirmed file
        if series_guard["launching"]:
            return
        if state.series_results is not None:
            _clear_series_display()
            series_guard["launching"] = False
            load_image_at(payload, force=True)
        else:
            load_image_at(payload)
        if state.batch_mode:
            saved = _batch_saved_count()
            n = len(state.files)
            fname = state.files[payload].name
            tag = "[済] " if fname in state.per_file_results else ""
            _set_status(
                f"画像 {payload + 1}/{n} · 確定済み {saved}/{n} — {tag}{fname}"
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
                if edit_state.get("clear_outside"):
                    status.value = "範囲外のマスクを消去します - OK で実行"
                else:
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
        try:
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
        except Exception as exc:
            _log.exception("continuous edit failed")
            edit_state["pressing"] = False
            _set_status(f"編集エラー: {exc}")
            page.update()

    def _commit_edit_stroke() -> None:
        if edit_state["changed"] and edit_state["preview_mask"] is not None:
            edit_state["undo_stack"].append(state.mask.copy())
            state.mask = _finalize_mask(edit_state["preview_mask"].copy())
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
        # Optos DFD: 乳頭2点＋中心窩が揃った時点でスケール確定
        if (
            _current_scale_method() == "dfd"
            and state.disc_pts
            and len(state.disc_pts) == 2
        ):
            _try_apply_dfd_scale()
            _refresh_scale_info()
        if not _try_apply_width_scale() and state.scale_um_px is None:
            status.value = (
                "中心窩を指定しました — HRA の場合は先に「①' 幅指定」で mm を設定してください"
                if modality_dd.value == "hra"
                else "中心窩を指定しました — 先に乳頭計測（DFD/乳頭径）を完了してください"
            )
            state.mode = "foveal_confirm"
        else:
            state.mode = "foveal_confirm"
            src = state.calibration_source or "-"
            status.value = (
                f"中心窩: ({ox:.0f}, {oy:.0f}) · スケール {src} "
                f"{state.scale_um_px:.2f} μm/px — 補正強度を調整"
            )
        state.series_results = None
        state.mask = None
        state.stats = None
        # 中心窩補正のデフォルト強度は 0（補正なし）
        state.foveal_correction_strength = 0.0
        foveal_strength_slider.value = 0
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
        state.foveal_correction_skipped = strength <= 0
        state.foveal_correction_strength = strength
        state.series_results = None
        # 時系列モード: seg は Registration 後。ここでは準備確定を促す（自動 seg しない）。
        if state.series_mode:
            state.mode = None
            fname = state.files[state.file_idx].name if state.files else ""
            src = state.series_date_source.get(fname)
            if state.series_dates_manual.get(fname):
                src = "manual"
            if not is_datetime_source_confirmed(src):
                _set_status(
                    "中心窩確認済み — サイドバーで日時を「確定」したあと"
                    "「時系列準備として確定」を押してください"
                )
            else:
                _set_status(
                    "乳頭・中心窩を確認済み — 「時系列準備として確定」を押してください"
                    "（seg は Registration 後）"
                )
            refresh_hint()
            update_confirm_visibility()
            refresh_image()
            refresh_results()
            _update_batch_ui()
            page.update()
            return
        # 独立解析: OK → ③ 解析（セグメンテーション確認）へ自動進行
        _run_segmentation_review(status_prefix="解析")
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
        if state.series_mode:
            state.mode = None
            _set_status(
                "中心窩確定（補正スキップ）— 「時系列準備として確定」を押してください"
            )
            refresh_hint()
            update_confirm_visibility()
            refresh_image()
            refresh_results()
            _update_batch_ui()
            page.update()
            return
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
        _begin_seg_clear(outside=False)

    def on_seg_clear_outside_start(_=None) -> None:
        _begin_seg_clear(outside=True)

    def _begin_seg_clear(*, outside: bool) -> None:
        if state.mask is None:
            _set_status("先に解析を実行してください")
            page.update()
            return
        if state.mode == "seg_review" and edit_state["session_base"] is None:
            edit_state["session_base"] = state.mask.copy()
            edit_state["undo_stack"] = []
        state.clear_pts = []
        state.mode = "seg_clear"
        edit_state["clear_outside"] = outside
        edit_state["preview_mask"] = None
        edit_state["changed"] = False
        edit_state["pressing"] = False
        _set_status(
            "フリーハンドで残す範囲を囲んでください（範囲外を消去）"
            if outside else
            "フリーハンドで消去範囲を囲んでください"
        )
        refresh_hint()
        update_confirm_visibility()
        page.update()

    def on_seg_clear_ok(_=None) -> None:
        if state.mask is None or len(state.clear_pts) < 3:
            _set_status("フリーハンドで範囲を囲んでください")
            page.update()
            return
        outside = bool(edit_state.get("clear_outside"))
        edit_state["undo_stack"].append(state.mask.copy())
        new_mask, removed = clear_mask_in_freehand_region(
            state.mask, state.clear_pts, outside=outside,
        )
        if removed == 0:
            edit_state["undo_stack"].pop()
        else:
            state.mask = _finalize_mask(new_mask)
        state.clear_pts = []
        state.mode = "seg_review"
        edit_state["clear_outside"] = False
        if outside:
            _set_status(
                f"範囲外マスクを消去 ({removed} px) - 続けて編集するか OK"
                if removed else "範囲外にマスクがありません - やり直しまたは続行"
            )
        else:
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
        outside = bool(edit_state.get("clear_outside"))
        _set_status(
            "フリーハンドで残す範囲を囲んでください（範囲外を消去）"
            if outside else
            "フリーハンドで消去範囲を囲んでください"
        )
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
        state.disc_pts = [p1, p2]
        mid = disc_xy_from_pts(state.disc_pts)
        method = _current_scale_method()
        state.scale_method = method
        if method == "disc":
            if not _apply_disc_scale_from_pts():
                status.value = "2点が同一座標のためスケールを算出できません"
                page.update()
                return
            scale_txt = f"{state.scale_um_px:.2f} μm/px ({state.calibration_source})"
        else:
            # DFD: 中心窩指定後にスケール確定（既に中心窩があれば即適用）
            if state.fovea_xy is not None and _try_apply_dfd_scale():
                scale_txt = f"{state.scale_um_px:.2f} μm/px ({state.calibration_source})"
            else:
                state.scale_um_px = None
                state.calibration_source = None
                scale_txt = "DFD待機（中心窩指定後に 4.76mm で較正）"
        state.disc_draft_pts = []
        state.mode = None
        _invalidate_saved_results(reason="disc_ok")
        kept = _batch_saved_count()
        mid_txt = (
            f" · 乳頭中点 ({mid[0]:.0f},{mid[1]:.0f})"
            if mid is not None else ""
        )
        status.value = (
            f"乳頭計測完了: {scale_txt}{mid_txt}"
            + (f" — 他画像の確定 {kept} 枚は保持" if kept else "")
            + " — 中心窩をタップしてください"
        )
        disc_info.value = ""
        _refresh_scale_info()
        # ①完了 → ②中心窩へ自動進行
        set_mode("fovea")
        return

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
            # ⓪完了 → Optos:①乳頭 / HRA:①'幅指定 へ自動進行
            if modality_dd.value == "hra":
                status.value = "クロップ完了 — 幅指定へ進みます"
                refresh_hint()
                update_confirm_visibility()
                refresh_image()
                on_width_start()
                return
            status.value = "クロップ完了 — 乳頭計測へ進みます"
            refresh_hint()
            update_confirm_visibility()
            refresh_image()
            set_mode("disc_line")
            return
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
        state.calibration_source = calibration.SOURCE_WIDTH
        state.field_width_mm = field_mm
        state.disc_pts = []
        state.disc_draft_pts = []
        state.mode = None
        _invalidate_saved_results(reason="width_ok")
        kept = _batch_saved_count()
        disc_info.value = ""
        status.value = (
            f"幅指定完了: {w_px} px = {field_mm} mm → {state.scale_um_px:.2f} μm/px"
            + (f" — 他画像の確定 {kept} 枚は保持" if kept else "")
            + " — 中心窩をタップしてください"
        )
        _refresh_scale_info()
        # ①'完了 → ②中心窩へ自動進行
        set_mode("fovea")
        return

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

    def on_scale_method_change(_e=None) -> None:
        """スケール基準の切替。乳頭＋中心窩が揃っていれば即再計算。"""
        method = _current_scale_method()
        state.scale_method = method
        if method == "dfd":
            if state.fovea_xy is not None and state.disc_pts and len(state.disc_pts) == 2:
                if _try_apply_dfd_scale(allow_disc_fallback=False):
                    _invalidate_saved_results(reason="scale_method_dfd")
                    status.value = (
                        f"スケール基準を DFD に変更: {state.scale_um_px:.2f} μm/px "
                        f"({state.calibration_source})"
                    )
                else:
                    status.value = "DFD を算出できません（乳頭と中心窩を確認）"
            else:
                status.value = "DFD 基準を選択 — 乳頭計測後に中心窩を指定してください"
        else:
            if state.disc_pts and len(state.disc_pts) == 2:
                if _apply_disc_scale_from_pts():
                    _invalidate_saved_results(reason="scale_method_disc")
                    status.value = (
                        f"スケール基準を乳頭径に変更: {state.scale_um_px:.2f} μm/px "
                        f"({state.calibration_source})"
                    )
                else:
                    status.value = "乳頭径スケールを算出できません"
            else:
                status.value = "乳頭径基準を選択 — 「① 乳頭」で端点を指定してください"
        _refresh_scale_info()
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
                elif (
                    _current_scale_method() == "dfd"
                    and state.disc_pts
                    and len(state.disc_pts) == 2
                ):
                    status.value = "中心窩をタップ（DFD 4.76mm でスケール確定）"
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
        state.mask = _finalize_mask(state.mask)
        state.stats = ga_stats(state.mask, state.scale_um_px, state.fovea_xy)
        state.mode = None
        saved = 0
        n = len(state.files)
        # 確定後は自動起動せず、ユーザーが対象をチェックして④を押す運用。
        if state.batch_mode:
            _save_batch_result()
            state.series_selected.setdefault(state.files[state.file_idx].name, True)
            saved = _batch_saved_count()
            selected = _selected_count()
            pos = state.file_idx + 1
            if state.series_mode and selected >= 2:
                _set_status(
                    f"確定 {saved}/{n} (画像 {pos}/{n}) · 対象 {selected} 枚"
                    " - 「次へ ▶」で続行、または Registration"
                )
            elif state.series_mode:
                _set_status(
                    f"確定 {saved}/{n} (画像 {pos}/{n})"
                    " - 「次へ ▶」で次画像（対象2枚以上で Registration 可）"
                )
            else:
                _set_status(
                    f"確定 {saved}/{n} (画像 {pos}/{n} · 症例 {state.case_id})"
                    " - 独立解析。時系列が必要ならモードを ON"
                )
        else:
            _set_status(_safe_text(f"解析完了 · 症例 {state.case_id}"))
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

    review_state = {"diff": True}

    def _review_current_idx() -> int:
        results = state.series_results or []
        if not results:
            return 0
        return max(0, min(state.series_review_idx, len(results) - 1))

    def _all_reg_ok() -> bool:
        results = state.series_results or []
        return bool(results) and all(r.get("reg_ok") for r in results)

    def _review_display_rgb() -> np.ndarray | None:
        results = state.series_results or []
        if not results:
            return None
        idx = _review_current_idx()
        row = results[idx]
        if review_state["diff"] and idx > 0:
            from engine.overlay import registration_diff_overlay
            return registration_diff_overlay(results[0]["image"], row["image"])
        return state.series_overlays[idx] if state.series_overlays else _row_overlay(row)

    def _refresh_review_panel() -> None:
        panel = _ui.get("reg_review_panel")
        if panel is None:
            return
        results = state.series_results or []
        active = state.series_review_active and bool(results)
        panel.visible = active
        if not active:
            return
        idx = _review_current_idx()
        row = results[idx]
        n = len(results)
        method_label = REGISTRATION_METHODS.get(
            row.get("reg_method", "affine"), row.get("reg_method", "-")
        )
        is_ref = row.get("is_reference")
        _ui["reg_review_title"].value = _safe_text(
            f"位置合わせ確認 {idx + 1}/{n} · {row['date']} · {row['filename']}"
        )
        if is_ref:
            lm = landmarks_from_snapshot(
                state.per_file_results.get(row["filename"], row)
            )
            disc_note = (
                f" · 乳頭 ({lm['disc_xy'][0]:.0f},{lm['disc_xy'][1]:.0f})"
                if lm.get("disc_xy") else ""
            )
            fxy = lm.get("fovea_xy")
            fovea_note = (
                f"中心窩 ({fxy[0]:.0f},{fxy[1]:.0f}){disc_note}"
                if fxy else "固定点なし"
            )
            _ui["reg_review_info"].value = _safe_text(
                f"基準時点（最古）— 移動なし · {fovea_note}"
            )
        else:
            mark = "OK" if row.get("reg_ok") else "未確認"
            snap = state.per_file_results.get(row["filename"], {})
            lm = landmarks_from_snapshot(snap) if snap else {}
            disc_note = (
                f" · 乳頭 ({lm['disc_xy'][0]:.0f},{lm['disc_xy'][1]:.0f})"
                if lm.get("disc_xy") else ""
            )
            fxy = lm.get("fovea_xy")
            fovea_note = (
                f"固定点: 中心窩 ({fxy[0]:.0f},{fxy[1]:.0f}){disc_note}"
                if fxy else "固定点: 中心窩なし"
            )
            _ui["reg_review_info"].value = _safe_text(
                f"手法: {method_label} · 状態: {mark} · {fovea_note}"
                " · 整列後は中心窩が基準点へ重なります（差分表示推奨）"
            )
        ok_count = sum(1 for r in results if r.get("reg_ok"))
        _ui["reg_review_progress"].value = _safe_text(
            f"確認済み {ok_count}/{n}"
        )
        _ui["reg_ok_btn"].disabled = is_ref
        _ui["reg_redo_row"].visible = not is_ref
        _ui["reg_finish_btn"].disabled = not _all_reg_ok()
        _ui["reg_diff_switch"].value = review_state["diff"]
        _ui["reg_diff_switch"].disabled = is_ref
        _ui["reg_method_dd"].value = row.get("reg_method", "affine")

    def _enter_registration_review() -> None:
        state.series_review_active = True
        _refresh_review_panel()
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()

    def _goto_review(idx: int) -> None:
        results = state.series_results or []
        if not results:
            return
        state.series_review_idx = max(0, min(idx, len(results) - 1))
        if state.series_roi_review_active:
            _refresh_roi_review_panel()
        else:
            _refresh_review_panel()
        refresh_image()
        page.update()

    def on_review_prev(_=None) -> None:
        _goto_review(_review_current_idx() - 1)

    def on_review_next(_=None) -> None:
        _goto_review(_review_current_idx() + 1)

    def on_review_diff_toggle(e) -> None:
        review_state["diff"] = bool(e.control.value)
        refresh_image()
        page.update()

    def on_review_ok(_=None) -> None:
        results = state.series_results or []
        idx = _review_current_idx()
        if not results or idx <= 0:
            return
        results[idx]["reg_ok"] = True
        # 次の未確認へ自動で進む
        nxt = next(
            (i for i in range(idx + 1, len(results)) if not results[i].get("reg_ok")),
            None,
        )
        if nxt is not None:
            state.series_review_idx = nxt
        _refresh_review_panel()
        refresh_image()
        if _all_reg_ok():
            _set_status("全時点の位置合わせを確認しました — 「確認完了」で結果表示へ")
        page.update()

    def _set_registration_busy(busy: bool) -> int:
        """位置合わせ busy を設定。戻り値は進捗コールバック無効化用の世代。"""
        registration_guard["gen"] += 1
        registration_guard["running"] = busy
        gen = int(registration_guard["gen"])
        reg_apply_btn.disabled = busy
        redo = _registration_controls.get("redo_btn")
        if redo is not None:
            redo.disabled = busy
        if not busy:
            # Hide progress UI; bump gen above already invalidated late threadsafe ticks
            reg_progress_bar.visible = False
            reg_progress_label.visible = False
            reg_progress_bar.value = 0
            reg_progress_label.value = ""
        return gen

    def _show_registration_progress(
        done: int, total: int, label: str, *, gen: int | None = None,
    ) -> None:
        """位置合わせ進捗をステータス＋バーに反映（メインスレッド想定）。"""
        if not registration_guard["running"]:
            return
        if gen is not None and gen != registration_guard["gen"]:
            return
        total = max(int(total), 1)
        done = max(0, min(int(done), total))
        reg_progress_bar.value = done / total
        reg_progress_bar.visible = True
        detail = f" · {label}" if label else ""
        reg_progress_label.value = _safe_text(
            f"位置合わせ {done}/{total}{detail}"
        )
        reg_progress_label.visible = True
        _set_status(_safe_text(f"位置合わせ中… {done}/{total}{detail}"))

    def on_review_redo(_=None) -> None:
        """選択中の手法で現在時点を再位置合わせする。"""
        if registration_guard["running"]:
            return
        results = state.series_results or []
        idx = _review_current_idx()
        if not results or idx <= 0:
            return
        method = _ui["reg_method_dd"].value or "affine"
        per_file = state.per_file_results
        method_label = REGISTRATION_METHODS.get(method, method)

        async def _redo_task() -> None:
            _set_registration_busy(True)
            _set_status(_safe_text(f"再位置合わせ中… ({method_label})"))
            page.update()
            try:
                updated = await asyncio.to_thread(
                    reregister_timepoint,
                    results,
                    idx,
                    per_file,
                    method=method,
                    resegment_after_reg=state.series_resegment_enabled,
                )
            except Exception as exc:
                _set_status(f"再位置合わせエラー: {exc}")
            else:
                results[idx] = updated
                state.series_methods[updated["filename"]] = method
                # 変わったのは 1 時点だけなので、その行の overlay だけ差し替える
                if (
                    state.series_overlays is not None
                    and idx < len(state.series_overlays)
                ):
                    state.series_overlays[idx] = _row_overlay(updated)
                    state.series_gif = None
                    _recompute_series_estimates()
                else:
                    _rebuild_series_derived()
                _refresh_review_panel()
                refresh_image()
                refresh_results()
                _set_status(_safe_text(
                    f"{updated['filename']} を {method_label} で再位置合わせしました"
                ))
            finally:
                _set_registration_busy(False)
                page.update()

        page.run_task(_redo_task)

    def _finish_to_series_results_view() -> None:
        """位置合わせ（およびROI確認）完了後の通常時系列表示へ。"""
        state.series_review_active = False
        state.series_roi_review_active = False
        state.series_registered = True
        _refresh_review_panel()
        if "roi_review_panel" in _ui:
            _refresh_roi_review_panel()
        _update_registration_controls()
        n = len(state.series_results or [])
        selected = _selected_count()
        state.view_mode = "work"
        view_mode.value = "work"
        view_mode.visible = True
        series_date_dd.visible = True
        state.series_time_idx = _review_current_idx()
        series_date_dd.value = str(state.series_time_idx)
        _sync_file_idx_to_series_timepoint(update_dropdown=True)
        _set_status(_safe_text(
            f"時系列表示 ({n} 時点 / 対象 {selected} 枚) "
            "— 撮影日 / 作業・GIF・グラフで切替"
        ))
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    def _revert_series_roi_correction() -> None:
        """QC補正を解除し、各時点を補正前マスク（mask_pre_roi）に戻す。"""
        results = state.series_results
        if not results or not state.series_roi_applied:
            return
        from engine.timeseries import _compute_row_stats
        for row in results:
            pre = row.pop("mask_pre_roi", None)
            if pre is not None:
                row["mask"] = pre
                row["stats"] = _compute_row_stats(
                    pre,
                    row.get("scale_um_px"),
                    row.get("fovea_xy"),
                    row.get("stats") or {},
                )
            for key in (
                "roi_needs_review", "roi_reason", "roi_ok",
                "roi_raw_area", "roi_corrected_area", "roi_refine_method",
            ):
                row.pop(key, None)
        state.series_roi_applied = False
        state.series_roi_qc = None
        _rebuild_series_derived()
        if state.series_roi_review_active:
            # 確認モード中に解除された場合は結果表示へ戻す
            _finish_to_series_results_view()
        _set_status("マスクQC補正を解除しました（補正前マスクに戻しました）")
        refresh_image()
        refresh_results()

    def on_roi_correction_toggle(e: ft.ControlEvent) -> None:
        state.series_roi_enabled = bool(e.control.value)
        if not state.series_roi_enabled:
            if state.series_roi_applied:
                _revert_series_roi_correction()
            page.update()
            return
        if (
            state.series_results
            and state.series_registered
            and not state.series_roi_applied
        ):
            _run_roi_correction_after_registration()
        elif state.series_results and not state.series_registered:
            _set_status(
                "マスクQC補正は位置合わせ済みの時系列に適用されます"
                "（位置合わせ確認完了後に自動適用）"
            )
        page.update()

    roi_correction_sw.on_change = on_roi_correction_toggle

    def _all_roi_ok() -> bool:
        results = state.series_results or []
        if not results:
            return False
        return all(
            (not r.get("roi_needs_review")) or r.get("roi_ok")
            for r in results
        )

    def _refresh_roi_review_panel() -> None:
        panel = _ui.get("roi_review_panel")
        if panel is None:
            return
        results = state.series_results or []
        active = state.series_roi_review_active and bool(results)
        panel.visible = active
        if not active:
            return
        idx = _review_current_idx()
        row = results[idx]
        n = len(results)
        reason = row.get("roi_reason") or "ok"
        mark = "OK" if row.get("roi_ok") else "未確認"
        _ui["roi_review_title"].value = _safe_text(
            f"ROI補正確認 {idx + 1}/{n} · {row['date']} · {row['filename']}"
        )
        _ui["roi_review_info"].value = _safe_text(
            f"理由: {reason} · 状態: {mark} · "
            f"面積 {row.get('roi_raw_area', '?')} → {row.get('roi_corrected_area', '?')} px"
        )
        ok_count = sum(
            1 for r in results
            if (not r.get("roi_needs_review")) or r.get("roi_ok")
        )
        _ui["roi_review_progress"].value = _safe_text(f"確認済み {ok_count}/{n}")
        needs = bool(row.get("roi_needs_review"))
        _ui["roi_ok_btn"].disabled = (not needs) or bool(row.get("roi_ok"))
        _ui["roi_reject_btn"].disabled = not needs
        _ui["roi_finish_btn"].disabled = not _all_roi_ok()

    def _enter_roi_review() -> None:
        state.series_roi_review_active = True
        view_mode.visible = False
        series_date_dd.visible = False
        _refresh_roi_review_panel()
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()

    def on_roi_review_prev(_=None) -> None:
        _goto_review(_review_current_idx() - 1)
        _refresh_roi_review_panel()
        page.update()

    def on_roi_review_next(_=None) -> None:
        _goto_review(_review_current_idx() + 1)
        _refresh_roi_review_panel()
        page.update()

    def on_roi_review_ok(_=None) -> None:
        results = state.series_results or []
        idx = _review_current_idx()
        if not results or not results[idx].get("roi_needs_review"):
            return
        results[idx]["roi_ok"] = True
        nxt = next(
            (
                i for i in range(idx + 1, len(results))
                if results[i].get("roi_needs_review") and not results[i].get("roi_ok")
            ),
            None,
        )
        if nxt is not None:
            state.series_review_idx = nxt
        _refresh_roi_review_panel()
        refresh_image()
        if _all_roi_ok():
            _set_status("全時点のROI補正を確認しました — 「確認完了」で結果表示へ")
        page.update()

    def on_roi_review_reject(_=None) -> None:
        """補正を破棄し、位置合わせ後の元マスクに戻す。"""
        results = state.series_results or []
        idx = _review_current_idx()
        if not results or not results[idx].get("roi_needs_review"):
            return
        row = results[idx]
        pre = row.get("mask_pre_roi")
        if pre is None:
            return
        row["mask"] = pre.copy()
        row["roi_corrected_area"] = int(np.count_nonzero(pre))
        from engine.timeseries import _compute_row_stats as _crs
        row["stats"] = _crs(
            row["mask"], row.get("scale_um_px"), row.get("fovea_xy"), row.get("stats") or {},
        )
        row["roi_ok"] = True
        row["roi_reason"] = "rejected_keep_registered"
        _rebuild_series_derived()
        nxt = next(
            (
                i for i in range(idx + 1, len(results))
                if results[i].get("roi_needs_review") and not results[i].get("roi_ok")
            ),
            None,
        )
        if nxt is not None:
            state.series_review_idx = nxt
        _refresh_roi_review_panel()
        refresh_image()
        refresh_results()
        page.update()

    def on_roi_review_finish(_=None) -> None:
        if not _all_roi_ok():
            _set_status("未確認のROI時点があります（各時点で OK または 補正破棄）")
            page.update()
            return
        _finish_to_series_results_view()

    def _run_roi_correction_after_registration() -> None:
        """位置合わせ確定後: 整列済みマスクに対しROI補正→要レビューなら確認UI。

        「マスクQC補正」トグル OFF の場合は補正せず結果表示へ直行する。
        補正前マスクは各行の mask_pre_roi に保持され、トグル OFF で
        _revert_series_roi_correction により補正前へ戻せる。
        QCログ（SOP 11章相当）は series_roi_qc に保持し CSV へ出力する。

        先頭ROIが末尾ROI(+3px)に含まれない場合は、漏れ領域を先頭から削除し
        末尾へ追加してから refine する（手動修正ダイアログは出さない）。
        """
        if registration_guard["running"] or series_guard.get("roi_running"):
            return
        results = state.series_results
        if not results:
            return
        if not state.series_roi_enabled:
            _finish_to_series_results_view()
            return
        if state.series_roi_applied:
            _finish_to_series_results_view()
            return

        async def _roi_task() -> None:
            series_guard["roi_running"] = True
            _set_status("ROI時系列補正を計算中…")
            page.update()
            try:
                out = await asyncio.to_thread(apply_roi_correction_to_series, results)
                if state.series_results is not results:
                    # 適用中に時系列が破棄/再構築された — 結果は捨てる
                    return
                overlays = await asyncio.to_thread(
                    lambda: [_row_overlay(r) for r in out["results"]],
                )
                state.series_results = out["results"]
                state.series_roi_applied = True
                state.series_roi_qc = out["qc_log"]
                _rebuild_series_derived(overlays=overlays)
                needs = out["needs_review_indices"]
                n_corr = sum(1 for m in out["refine_methods"] if m)
                recon = out.get("endpoint_reconcile")
                recon_note = ""
                if recon:
                    recon_note = (
                        f" · 端点自動調整"
                        f"（先頭−{recon['removed_from_first_px']}px"
                        f" / 末尾+{recon['added_to_last_px']}px）"
                    )
                if needs:
                    state.series_review_idx = needs[0]
                    _enter_roi_review()
                    _set_status(_safe_text(
                        f"ROI補正確認 ({len(needs)} 時点が要レビュー) "
                        "— 各時点で OK / 補正破棄を選択"
                        f"{recon_note}"
                    ))
                else:
                    _finish_to_series_results_view()
                    if n_corr or recon:
                        _set_status(_safe_text(
                            f"時系列表示 ({len(out['results'])} 時点) "
                            f"— マスクQC補正 {n_corr} 時点"
                            f"{recon_note}"
                        ))
                    else:
                        _set_status(_safe_text(
                            f"時系列表示 ({len(out['results'])} 時点) "
                            "— マスクQC補正: 異常なし（マスク変更なし）"
                        ))
            except Exception as exc:
                _log.exception("ROI correction failed")
                _set_status(f"ROI補正エラー: {exc}")
                _finish_to_series_results_view()
            finally:
                series_guard["roi_running"] = False
                page.update()

        page.run_task(_roi_task)

    def _tag_series_case_id(results: list[dict]) -> list[dict]:
        """結果行に case_id / exam_datetime を付与。"""
        out = []
        for row in results:
            r = dict(row)
            r["case_id"] = r.get("case_id") or state.case_id
            r["exam_datetime"] = r.get("exam_datetime") or r.get("date")
            out.append(r)
        return out

    def _finish_aligned_pipeline(*, landing_filename: str | None = None) -> None:
        """整列QC完了後:（本線）seg → 測定 → 任意マスクQC。"""

        async def _pipeline() -> None:
            results = state.series_results or []
            if not results:
                return
            try:
                if state.series_resegment_enabled:
                    _set_status("整列空間でセグメンテーション中…")
                    page.update()
                    try:
                        results = await asyncio.to_thread(
                            segment_aligned_series,
                            results,
                            allow_warp_fallback=True,
                        )
                    except Exception as exc:
                        _set_status(f"整列後 seg エラー: {exc}")
                        return
                    results = _tag_series_case_id(results)
                    overlays = await asyncio.to_thread(
                        lambda: [_row_overlay(row) for row in results],
                    )
                    _apply_series_results(
                        results,
                        landing_filename=landing_filename,
                        overlays=overlays,
                    )
                else:
                    results = _tag_series_case_id(results)
                    state.series_results = results
                _run_roi_correction_after_registration()
            except Exception as exc:
                _log.exception("aligned pipeline failed")
                _set_status(f"整列後処理エラー: {exc}")
            finally:
                page.update()

        page.run_task(_pipeline)

    def on_review_finish(_=None) -> None:
        if not _all_reg_ok():
            _set_status("未確認の時点があります（各時点で OK を押してください）")
            page.update()
            return
        state.series_review_active = False
        state.series_registered = True
        _stash_reg_cache_from_series()
        _refresh_review_panel()
        prefer = None
        if state.series_results:
            prefer = state.series_results[
                min(state.series_time_idx, len(state.series_results) - 1)
            ]["filename"]
        _finish_aligned_pipeline(landing_filename=prefer)

    def _show_series_results_view(*, n: int, selected: int) -> None:
        """位置合わせなしで時系列結果を表示（グラフ優先・独立解析向け）。"""
        state.series_registered = False
        state.series_review_active = False
        state.series_roi_review_active = False
        state.view_mode = "graph"
        view_mode.value = "graph"
        view_mode.visible = True
        series_date_dd.visible = True
        _refresh_review_panel()
        _update_registration_controls()
        _set_status(_safe_text(
            f"時系列グラフ ({n} 時点 / 対象 {selected} 枚)"
            " — 位置合わせなし（各画像の座標のまま）"
        ))

    def _run_registration_on_series(
        *,
        landing_filename: str | None = None,
        launch_guard_gen: int | None = None,
    ) -> None:
        """選択済み時系列に位置合わせを適用し、確認モードへ入る（非同期）。

        本線は reg → 整列QC →（後段で）seg。ビルド時の再segは行わない。
        """
        def _finish_launch_guard() -> None:
            if launch_guard_gen is not None:
                _end_series_launch_guard(launch_guard_gen)

        if registration_guard["running"]:
            _finish_launch_guard()
            return
        if not state.series_mode:
            _finish_launch_guard()
            _set_status("時系列モードを ON にしてください")
            page.update()
            return
        selected = _selected_count()
        if selected < 2:
            _finish_launch_guard()
            _set_status(f"位置合わせには対象2枚以上が必要です (現在 選択 {selected} 枚)")
            page.update()
            return
        selected_set = set(_selected_confirmed())
        _ensure_selected_series_dates(selected_set)
        if not _selected_dates_ready():
            _finish_launch_guard()
            _set_status(
                "全対象画像の検査日時を確定してください（入力後「確定」）"
            )
            refresh_series_dates_ui()
            _update_registration_controls()
            page.update()
            return
        if not _selected_landmarks_ready():
            _finish_launch_guard()
            _set_status(
                "Registration 前に、対象の各画像で乳頭と中心窩を確認し"
                "「時系列準備として確定」してください"
            )
            refresh_series_dates_ui()
            _update_registration_controls()
            page.update()
            return
        prefer = landing_filename
        if prefer is None and state.series_results:
            prefer = state.series_results[
                min(state.series_time_idx, len(state.series_results) - 1)
            ]["filename"]
        elif prefer is None and state.files:
            prefer = state.files[state.file_idx].name
        per_file = state.per_file_results
        files = state.files
        series_dates = state.series_dates
        series_methods = state.series_methods
        prior_matrices, cache_confirmed = _prior_matrices_from_cache(
            selected_set, series_dates, series_methods,
        )

        async def _registration_task() -> None:
            busy_gen = _set_registration_busy(True)
            proceed_to_seg = False
            loop = asyncio.get_running_loop()
            if prior_matrices:
                _set_status("保存済みの位置合わせ行列を適用中…")
            else:
                _set_status("位置合わせを計算中…")
            reg_progress_bar.value = 0
            reg_progress_bar.visible = True
            reg_progress_label.value = ""
            reg_progress_label.visible = True
            page.update()

            def _on_progress(done: int, total: int, label: str) -> None:
                def _apply() -> None:
                    # Ignore ticks after finally cleared busy (classic stuck bar).
                    if (
                        not registration_guard["running"]
                        or registration_guard["gen"] != busy_gen
                    ):
                        return
                    _show_registration_progress(
                        done, total, label, gen=busy_gen,
                    )
                    page.update()

                try:
                    loop.call_soon_threadsafe(_apply)
                except RuntimeError:
                    _apply()

            try:
                # 本線: ビルド時は warp のみ。seg は整列QC後。
                results = await asyncio.to_thread(
                    build_series_from_saved,
                    per_file,
                    files,
                    series_dates,
                    register=True,
                    selected=selected_set,
                    methods=series_methods,
                    prior_matrices=prior_matrices,
                    progress_callback=_on_progress,
                    resegment_after_reg=False,
                )
                results = _tag_series_case_id(results)
            except ValueError as exc:
                _set_status(str(exc))
            except Exception as exc:
                _set_status(f"位置合わせエラー: {exc}")
            else:
                try:
                    overlays = await asyncio.to_thread(
                        lambda: [_row_overlay(row) for row in results],
                    )
                    _apply_series_results(
                        results, landing_filename=prefer, overlays=overlays,
                    )
                    if prior_matrices and cache_confirmed:
                        state.series_review_active = False
                        state.series_registered = True
                        _stash_reg_cache_from_series()
                        _refresh_review_panel()
                        proceed_to_seg = True
                    else:
                        state.series_review_active = True
                        state.series_review_idx = 1 if len(results) > 1 else 0
                        view_mode.visible = False
                        series_date_dd.visible = False
                        _update_registration_controls()
                        reused_note = (
                            "（行列再利用・要確認）" if prior_matrices else ""
                        )
                        _set_status(_safe_text(
                            f"位置合わせ確認 ({len(results)} 時点 / 対象 {selected} 枚)"
                            f"{reused_note}"
                            " — 各時点で OK / やり直し → 確認完了後に seg"
                        ))
                        _enter_registration_review()
                    _update_batch_ui(refresh_dropdown=False)
                    _refresh_file_dropdown_labels()
                except Exception as exc:
                    _log.exception("registration post-process failed")
                    _set_status(f"位置合わせ後処理エラー: {exc}")
            finally:
                _finish_launch_guard()
                _set_registration_busy(False)
                page.update()
            if proceed_to_seg:
                _finish_aligned_pipeline(landing_filename=prefer)

        page.run_task(_registration_task)

    def on_series_mode_toggle(e: ft.ControlEvent) -> None:
        state.series_mode = bool(e.control.value)
        state.series_registration_enabled = state.series_mode
        if not state.series_mode:
            _clear_series_display()
            _clear_reg_cache()
            # Hide progress UI immediately; invalidate late progress ticks.
            # Do not clear running here — in-flight task finally owns that.
            registration_guard["gen"] += 1
            reg_progress_bar.visible = False
            reg_progress_label.visible = False
            reg_progress_bar.value = 0
            reg_progress_label.value = ""
            if not registration_guard["running"]:
                series_guard["launching"] = False
                series_guard["roi_running"] = False
            _set_status("独立解析モード — 各画像で landmark → scale → seg → 測定")
        else:
            _set_status(
                "時系列モード — 日時「確定」→ 各画像で乳頭＋中心窩→準備確定"
                " → Registration → 整列QC → seg"
            )
        _update_batch_ui()
        page.update()

    def on_reseg_after_reg_toggle(e: ft.ControlEvent) -> None:
        state.series_resegment_enabled = bool(e.control.value)
        page.update()

    def on_apply_registration(_=None) -> None:
        if not state.series_mode:
            _set_status("時系列モードを ON にしてください")
            page.update()
            return
        if state.series_registered and state.series_results is not None:
            _set_status("この時系列は既に位置合わせ済みです")
            page.update()
            return
        _run_registration_on_series()

    reg_apply_btn.on_click = on_apply_registration
    series_mode_sw.on_change = on_series_mode_toggle
    reseg_after_reg_sw.on_change = on_reseg_after_reg_toggle
    prep_confirm_btn.on_click = on_prep_confirm

    def _launch_series_display(
        *,
        auto: bool = False,
        landing_filename: str | None = None,
    ) -> None:
        """時系列モード: Registration を起動（症例内・日時ゲート付き）。"""
        if registration_guard["running"] or series_guard["launching"]:
            return
        if not state.series_mode:
            _set_status("時系列モードを ON にしてください")
            page.update()
            return
        selected = _selected_count()
        if selected < 2:
            _set_status(f"時系列には対象2枚以上が必要です (現在 選択 {selected} 枚)")
            page.update()
            return
        prefer = landing_filename
        if prefer is None and state.files:
            prefer = state.files[state.file_idx].name
        selected_set = set(_selected_confirmed())
        _ensure_selected_series_dates(selected_set)
        refresh_series_dates_ui()
        if not _selected_dates_ready():
            _set_status(
                "全対象画像の検査日時を確定してください（入力後「確定」）"
            )
            _update_registration_controls()
            page.update()
            return
        if not _selected_landmarks_ready():
            _set_status(
                "Registration 前に、対象の各画像で乳頭（disc）と中心窩を確認し"
                "「時系列準備として確定」してください"
            )
            _update_registration_controls()
            page.update()
            return
        dd_gen = _begin_series_launch_guard()
        state.disc_diameter_um = float(disc_field.value or 1800)
        _run_registration_on_series(
            landing_filename=prefer,
            launch_guard_gen=dd_gen,
        )

    def on_series(_=None) -> None:
        _launch_series_display(auto=False)

    folder_picker_pending: dict = {"pick_input": False}

    def on_folder_picker_result(e: ft.FilePickerResultEvent) -> None:
        if folder_picker_pending.get("pick_input") and e.path:
            input_field.value = e.path
            on_load_files()
        folder_picker_pending["pick_input"] = False

    def on_pick_folder(_=None) -> None:
        folder_picker_pending["pick_input"] = True
        folder_picker.get_directory_path(dialog_title="入力フォルダを選択")

    pdf_export_pending: dict = {"bytes": None}

    def on_pdf_picker_result(e: ft.FilePickerResultEvent) -> None:
        if e.path and pdf_export_pending.get("bytes"):
            Path(e.path).write_bytes(pdf_export_pending["bytes"])
            status.value = f"PDF 保存: {e.path}"
            pdf_export_pending["bytes"] = None
            page.update()

    folder_picker = ft.FilePicker(on_result=on_folder_picker_result)
    pdf_picker = ft.FilePicker(on_result=on_pdf_picker_result)
    page.overlay.extend([folder_picker, pdf_picker, date_picker])

    def on_view_change(e) -> None:
        state.view_mode = e.control.value
        refresh_image()
        page.update()

    view_mode.on_change = on_view_change

    def on_graph_metric_change(e) -> None:
        state.graph_metric = e.control.value or "area_mm2"
        _refresh_graph()
        page.update()

    graph_metric_dd.on_change = on_graph_metric_change

    def on_series_date(e) -> None:
        if e.control.value is not None:
            _goto_series_timepoint(int(e.control.value))

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

    series_btn = btn_toolbar("④ Registration", on_click=on_series)
    series_btn.visible = False

    series_launch_btn = btn_primary("④ Registration", on_click=on_series)
    series_launch_btn.visible = False
    series_launch_row = ft.Column(
        [
            reseg_after_reg_sw,
            roi_correction_sw,
            ft.Row([series_launch_btn], spacing=8),
            reg_apply_btn,
            reg_progress_label,
            reg_progress_bar,
        ],
        spacing=6,
        visible=False,
    )

    toolbar = ft.Row(
        [
            btn_toolbar("⓪ クロップ", on_click=lambda _: set_mode("crop_drag")),
            btn_toolbar("① 乳頭", on_click=lambda _: set_mode("disc_line")),
            btn_toolbar("①' 幅指定", on_click=on_width_start),
            btn_toolbar("② 中心窩", on_click=lambda _: set_mode("fovea")),
            btn_primary("③ 解析", on_click=on_analyze),
            prep_confirm_btn,
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
            edit_state["clear_outside"] = False
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
        btn_outlined("範囲外消去", on_click=on_seg_clear_outside_start),
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
    scale_method_dd.on_change = on_scale_method_change
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
                        btn_ok("✓ OK（解析へ）", on_click=on_foveal_ok),
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

    reg_review_title = ft.Text("", size=13, weight=ft.FontWeight.W_600, color=T.PRIMARY)
    reg_review_info = ft.Text("", size=12, color=T.TEXT)
    reg_review_progress = ft.Text("", size=11, color=T.TEXT_MUTED)
    reg_method_dd = ft.Dropdown(
        label="やり直しの手法",
        width=300,
        text_size=12,
        value="landmark",
        options=[
            ft.dropdown.Option(key=k, text=v)
            for k, v in REGISTRATION_METHODS.items()
        ],
    )
    reg_ok_btn = btn_ok("✓ この位置合わせでOK", on_click=on_review_ok)
    reg_redo_btn = btn_outlined("↩ この手法でやり直し", on_click=on_review_redo)
    _registration_controls["redo_btn"] = reg_redo_btn
    reg_redo_row = ft.Row([reg_method_dd, reg_redo_btn], spacing=8, wrap=True)
    reg_diff_switch = ft.Switch(
        label="差分表示（緑=基準 / 桃=対象）", value=True,
        on_change=on_review_diff_toggle,
    )
    reg_finish_btn = btn_primary("確認完了 → 結果表示", on_click=on_review_finish)
    reg_review_nav = ft.Row(
        [
            btn_outlined("◀ 前", on_click=on_review_prev),
            btn_outlined("次 ▶", on_click=on_review_next),
            reg_review_progress,
        ],
        spacing=8,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    reg_review_panel = ft.Container(
        visible=False,
        bgcolor=T.PANEL_BG,
        border=ft.border.all(1, "#374151"),
        border_radius=8,
        padding=ft.padding.all(12),
        content=ft.Column(
            [
                reg_review_title,
                reg_review_info,
                reg_diff_switch,
                reg_review_nav,
                ft.Row([reg_ok_btn], spacing=8),
                reg_redo_row,
                ft.Divider(color="#374151"),
                reg_finish_btn,
            ],
            spacing=8,
            tight=True,
        ),
    )
    _ui.update({
        "reg_review_panel": reg_review_panel,
        "reg_review_title": reg_review_title,
        "reg_review_info": reg_review_info,
        "reg_review_progress": reg_review_progress,
        "reg_method_dd": reg_method_dd,
        "reg_ok_btn": reg_ok_btn,
        "reg_redo_row": reg_redo_row,
        "reg_diff_switch": reg_diff_switch,
        "reg_finish_btn": reg_finish_btn,
    })

    roi_review_title = ft.Text("", size=13, weight=ft.FontWeight.W_600, color=T.PRIMARY)
    roi_review_info = ft.Text("", size=12, color=T.TEXT)
    roi_review_progress = ft.Text("", size=11, color=T.TEXT_MUTED)
    roi_ok_btn = btn_ok("✓ この補正でOK", on_click=on_roi_review_ok)
    roi_reject_btn = btn_outlined("補正破棄（登録後マスクへ戻す）", on_click=on_roi_review_reject)
    roi_finish_btn = btn_primary("確認完了 → 結果表示", on_click=on_roi_review_finish)
    roi_review_nav = ft.Row(
        [
            btn_outlined("◀ 前", on_click=on_roi_review_prev),
            btn_outlined("次 ▶", on_click=on_roi_review_next),
            roi_review_progress,
        ],
        spacing=8,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    roi_review_panel = ft.Container(
        visible=False,
        bgcolor=T.PANEL_BG,
        border=ft.border.all(1, "#374151"),
        border_radius=8,
        padding=ft.padding.all(12),
        content=ft.Column(
            [
                roi_review_title,
                roi_review_info,
                roi_review_nav,
                ft.Row([roi_ok_btn, roi_reject_btn], spacing=8, wrap=True),
                ft.Divider(color="#374151"),
                roi_finish_btn,
            ],
            spacing=8,
            tight=True,
        ),
    )
    _ui.update({
        "roi_review_panel": roi_review_panel,
        "roi_review_title": roi_review_title,
        "roi_review_info": roi_review_info,
        "roi_review_progress": roi_review_progress,
        "roi_ok_btn": roi_ok_btn,
        "roi_reject_btn": roi_reject_btn,
        "roi_finish_btn": roi_finish_btn,
    })

    left = ft.Column(
        [
            ft.Text("GA Segmentation Pro", size=22, weight=ft.FontWeight.BOLD, color=T.PRIMARY),
            ft.Text(f"研究用 · build {APP_VERSION}", size=10, color=T.TEXT_MUTED),
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
            reg_review_panel,
            roi_review_panel,
            view_mode,
            series_date_dd,
            img_stack,
            graph_area,
        ],
        spacing=8,
        scroll=ft.ScrollMode.AUTO,
    )

    def on_case_change(e: ft.ControlEvent) -> None:
        cid = e.control.value
        if cid:
            _switch_case(cid)
            page.update()

    def on_create_case(_=None) -> None:
        name = (new_case_field.value or "").strip()
        if not name:
            _set_status("新規症例名を入力してください")
            page.update()
            return
        state.stash_current_case_results()
        state.cases, cid = create_case(state.cases, name)
        new_case_field.value = ""
        _switch_case(cid)
        _set_status(_safe_text(f"症例「{cid}」を作成 — 画像を選択して振り分けてください"))
        page.update()

    def on_case_select_all(_=None) -> None:
        for f in state.files:
            state.case_move_selected[f.name] = True
        _refresh_case_move_list()
        page.update()

    def on_case_deselect_all(_=None) -> None:
        state.case_move_selected = {}
        _refresh_case_move_list()
        page.update()

    def on_move_selected(_=None) -> None:
        to_case = move_case_dd.value
        if not to_case:
            _set_status("移動先の症例を選んでください（先に症例を作成）")
            page.update()
            return
        from_case = state.case_id
        if to_case == from_case:
            _set_status("移動先は現在の症例と異なるものを選んでください")
            page.update()
            return
        selected_paths = [
            f for f in state.files
            if state.case_move_selected.get(f.name)
        ]
        if not selected_paths:
            _set_status("移動する画像にチェックを入れてください")
            page.update()
            return
        try:
            # 移動前に現在症例の結果を退避し、各ファイルのスナップを持ち越し
            state.stash_current_case_results()
            snaps: dict[str, dict] = {}
            for path in selected_paths:
                snap = state.per_file_results.pop(path.name, None)
                if snap is not None:
                    snaps[path.name] = snap
            state.cases = move_files_to_case(
                state.cases, selected_paths, from_case, to_case,
            )
            state.case_results[from_case] = dict(state.per_file_results)
            dest = dict(state.case_results.get(to_case) or {})
            for fname, snap in snaps.items():
                snap["case_id"] = to_case
                dest[fname] = snap
            state.case_results[to_case] = dest
            state.case_move_selected = {}
            _clear_series_display()
            _clear_reg_cache()
            # 移動先へ切替（症例 isolation を維持）
            state.case_id = to_case
            state.restore_case_results(to_case)
            state.files = list(state.cases.get(to_case) or [])
            focus = selected_paths[0]
            state.file_idx = next(
                (i for i, f in enumerate(state.files) if f == focus), 0,
            )
            state.series_selected = {}
            init_series_dates(state.files)
            if state.files:
                load_image_at(state.file_idx, force=True)
            n = len(selected_paths)
            names = ", ".join(p.name for p in selected_paths[:3])
            if n > 3:
                names += f" 他{n - 3}枚"
            _set_status(_safe_text(f"{n} 枚を「{to_case}」へ移動（{names}）"))
            _update_batch_ui()
        except Exception as exc:
            _log.exception("case move failed")
            _set_status(f"症例移動エラー: {exc}")
        finally:
            # Always refresh; _clear_series_display already hid progress chrome.
            page.update()

    case_dd.on_change = on_case_change
    create_case_btn.on_click = on_create_case
    case_select_all_btn.on_click = on_case_select_all
    case_deselect_all_btn.on_click = on_case_deselect_all
    move_selected_btn.on_click = on_move_selected

    sidebar = ft.Column(
        [
            ft.Text("フォルダ / 症例", weight=ft.FontWeight.BOLD, color=T.PRIMARY),
            ft.Row([input_field, ft.IconButton(ft.Icons.FOLDER_OPEN, on_click=on_pick_folder, tooltip="フォルダ選択")]),
            output_field,
            btn_sidebar("画像リストを読込", on_click=on_load_files, expand=True),
            case_dd,
            ft.Row([new_case_field, create_case_btn], spacing=6),
            case_move_section,
            series_mode_sw,
            file_dd,
            series_dates_section,
            ft.Divider(color="#374151"),
            modality_dd,
            scale_method_dd,
            disc_field,
            ft.Text(
                "Optos 既定は DFD 4.76mm。乳頭径固定も選択可。"
                "HRA は幅指定を優先。",
                size=10,
                color=T.TEXT_MUTED,
            ),
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
        import os
        try:
            await asyncio.sleep(0.05)
            auto_dir = os.environ.get("GA_INPUT_DIR", "").strip()
            if not auto_dir:
                return
            input_field.value = auto_dir
            folder = Path(auto_dir)
            state.input_dir = auto_dir
            state.output_dir = output_field.value.strip()
            cases = await asyncio.to_thread(discover_cases, folder)
            state.cases = cases
            state.case_results = {}
            ids = ordered_case_ids(cases)
            if ids:
                prefer = next(
                    (c for c in ids if c != UNCLASSIFIED_CASE_ID), ids[0],
                )
                state.case_id = prefer
                state.files = list(cases.get(prefer) or [])
            else:
                state.case_id = UNCLASSIFIED_CASE_ID
                state.files = []
            init_series_dates(state.files)
            refresh_series_dates_ui()
            if state.files:
                path = state.files[0]
                gray = await asyncio.to_thread(load_gray, path)
                state.reset_analysis(keep_series=True)
                state.reset_per_image()
                state.img_full = gray
                state.img_gray = gray.copy()
                state.crop_box = None
                state.crop_pts = []
                _refresh_image_width_label()
                _refresh_scale_info()
                _refresh_file_dropdown()
                refresh_image()
                n_cases = len(state.cases)
                n_files = sum(len(v) for v in state.cases.values())
                _set_status(
                    _safe_text(
                        f"{n_cases} 症例 / {n_files} 枚（現在: {state.case_id}）"
                    )
                )
            else:
                _set_status(f"画像なし: {folder}")
                state.img_gray = None
                refresh_image()
            _update_batch_ui()
        except Exception as exc:
            _log.exception("deferred auto-load failed")
            _set_status(f"自動読込エラー: {exc}")
        finally:
            # Ensure progress chrome is not left visible after autoload.
            if not registration_guard["running"]:
                registration_guard["gen"] += 1
                reg_progress_bar.visible = False
                reg_progress_label.visible = False
                reg_progress_bar.value = 0
                reg_progress_label.value = ""
            try:
                page.update()
            except Exception:
                pass

    page.run_task(_deferred_auto_load)
