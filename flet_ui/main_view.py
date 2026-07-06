"""Flet メインビュー — engine を直接呼び出す（FastAPI なし）。"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime
from pathlib import Path

import cv2
import flet as ft
import numpy as np

from engine import calibration, segmentation
from engine.crop import apply_crop, normalize_box
from engine.io_utils import SUPPORTED_SUFFIXES, list_images, load_gray
from engine.life_table import LIFE_TABLE_SOURCE, format_life_expectancy, life_expectancy_years
from engine.overlay import make_overlay
from engine.region_growing import erase_oversegmentation_from_mask
from engine.stats import ga_stats
from engine.timeseries import (
    analyze_time_series,
    estimate_progression_rate_um_per_year,
    guess_image_date,
    overlays_to_gif_bytes,
)
from flet_ui.state import GAAppState
from flet_ui import theme as T
from flet_ui.theme import (
    apply_page_theme,
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
    prog_field = ft.TextField(
        label="進行速度 (μm/年)", value=str(int(state.prog)), width=160,
        keyboard_type=ft.KeyboardType.NUMBER,
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
    gif_img = ft.Image(visible=False, fit=ft.ImageFit.CONTAIN, width=DISPLAY_WIDTH)
    graph_area = ft.Column(visible=False)

    disp_scale = {"v": 1.0}
    disp_size = {"w": DISPLAY_WIDTH, "h": 200}
    drag = {"active": False, "start": (0.0, 0.0)}
    erase_state = {"pressing": False, "preview_mask": None, "changed": False}

    def _to_img_xy(local_x: float, local_y: float) -> tuple[float, float]:
        h, w = state.img_gray.shape
        ox = max(0.0, min(w - 1, local_x / disp_scale["v"]))
        oy = max(0.0, min(h - 1, local_y / disp_scale["v"]))
        return ox, oy

    def _display_mask() -> np.ndarray:
        if erase_state["preview_mask"] is not None:
            return erase_state["preview_mask"]
        if state.mask is not None:
            return state.mask
        return np.zeros_like(state.img_gray) if state.img_gray is not None else np.array([])

    def _mask_style() -> str:
        if state.mode in ("seg_review", "seg_erase"):
            return "blend"
        return "solid"

    def _overlay_disc_pts() -> list[tuple[float, float]] | None:
        if state.mode in ("disc_line", "disc_confirm") and len(state.disc_draft_pts) == 2:
            return state.disc_draft_pts
        return state.disc_pts or None

    def _build_rgb_frame(self_mask: np.ndarray | None = None) -> np.ndarray:
        """表示用 RGB フレームを生成（erase プレビュー用に mask を差し替え可能）。"""
        mask = self_mask if self_mask is not None else _display_mask()
        stats = state.stats
        return make_overlay(
            state.img_gray,
            mask,
            fovea_xy=state.fovea_xy,
            nearest_xy=stats["nearest_xy"] if stats else None,
            disc_pts=_overlay_disc_pts(),
            disc_diameter_um=state.disc_diameter_um,
            crop_pts=state.crop_pts if state.mode in ("crop_drag", "crop_confirm") else None,
            show_macula=state.stats is not None and state.mode not in ("seg_review", "seg_erase"),
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

    def load_image_at(idx: int) -> None:
        if not state.files:
            return
        state.file_idx = max(0, min(idx, len(state.files) - 1))
        path = state.files[state.file_idx]
        gray = load_gray(path)
        state.img_full = gray
        state.img_gray = gray.copy()
        state.crop_box = None
        state.crop_pts = []
        state.reset_analysis(keep_series=True)
        refresh_image()
        refresh_results()

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
                    ft.DataCell(ft.Text(f"{s['gawain']:.1f}" if s["gawain"] is not None else "—")),
                    ft.DataCell(ft.Text(
                        f"{s['dist_um']:.0f}" if s["dist_um"] is not None else "—")),
                ])
            )
        graph_area.controls.append(
            ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Date")),
                    ft.DataColumn(ft.Text("GA mm²")),
                    ft.DataColumn(ft.Text("GAWAIN")),
                    ft.DataColumn(ft.Text("Dist μm")),
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
        try:
            prog = float(prog_field.value)
        except ValueError:
            prog = state.prog
        cross = round(dist / prog, 1) if dist and prog > 0 else None

        results_col.controls.extend([
            _card("GA Area", f"{stats['area_mm2']} mm²" if stats else "—", T.ACCENT_RED),
            _card("GAWAIN", f"{stats['gawain']:.1f}" if stats and stats.get("gawain") is not None else "—", T.ACCENT_AMBER),
            _card("Dist to fovea", f"{dist:.0f} μm" if dist is not None else "—", T.ACCENT_CYAN),
            _card("平均余命", life),
            _card("Cross Point", f"{cross} years" if cross is not None else "—", "#10B981"),
        ])
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
                    f"#{l['id']}: {l['area_mm2']} mm², dist {l['dist_um']} μm",
                    size=12, color=T.TEXT_MUTED,
                ))
        page.update()

    def refresh_hint() -> None:
        msgs = {
            "crop_drag": "クロップ: ドラッグで矩形を描画",
            "crop_confirm": "クロップ範囲を確認して OK または やり直し",
            "disc_line": "乳頭計測: 乳頭の両端を結ぶ直線をドラッグ",
            "disc_confirm": "直線を確認して OK または やり直し",
            "fovea": "中心窩をタップ",
            "seg_review": "セグメンテーション確認 — Erase で修正、✓ OK で確定",
            "seg_erase": "長押しで削除（複数箇所可）— ✓ OK で確定",
        }
        hint.value = msgs.get(state.mode or "", "")
        hint.color = T.PRIMARY if state.mode else T.TEXT_MUTED

    def on_load_files(_=None) -> None:
        state.input_dir = input_field.value.strip()
        state.output_dir = output_field.value.strip()
        folder = Path(state.input_dir)
        state.files = list_images(folder)
        state.series_dates = {
            f.name: guess_image_date(f).isoformat() for f in state.files
        }
        file_dd.options = [ft.dropdown.Option(key=f.name, text=f.name) for f in state.files]
        if state.files:
            file_dd.value = state.files[0].name
            status.value = f"{len(state.files)} 件読み込み（{', '.join(sorted(SUPPORTED_SUFFIXES))}）"
            load_image_at(0)
        else:
            status.value = f"画像なし: {folder}"
            state.img_gray = None
            refresh_image()
        page.update()

    def on_file_change(e) -> None:
        if not e.control.value:
            return
        for i, f in enumerate(state.files):
            if f.name == e.control.value:
                load_image_at(i)
                break

    file_dd.on_change = on_file_change

    def on_pan_start(e: ft.DragStartEvent) -> None:
        if state.img_gray is None or state.mode in ("seg_review", "seg_erase"):
            return
        if state.mode == "disc_line":
            drag["active"] = True
            drag["start"] = (e.local_x, e.local_y)
            p = _to_img_xy(e.local_x, e.local_y)
            state.disc_draft_pts = [p, p]
            refresh_image(do_update=False)
        elif state.mode == "crop_drag":
            drag["active"] = True
            drag["start"] = (e.local_x, e.local_y)
            p = _to_img_xy(e.local_x, e.local_y)
            state.crop_pts = [p, p]
            refresh_image(do_update=False)

    def on_pan_update(e: ft.DragUpdateEvent) -> None:
        if not drag["active"] or state.img_gray is None:
            return
        if state.mode == "disc_line":
            p1 = _to_img_xy(*drag["start"])
            p2 = _to_img_xy(e.local_x, e.local_y)
            state.disc_draft_pts = [p1, p2]
            refresh_image()
        elif state.mode == "crop_drag":
            p1 = _to_img_xy(*drag["start"])
            p2 = _to_img_xy(e.local_x, e.local_y)
            state.crop_pts = [p1, p2]
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
                status.value = f"乳頭径ライン: {d_px:.0f} px — OK で確定"
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
        update_confirm_visibility()
        refresh_hint()
        refresh_image()
        page.update()

    async def continuous_erase(ix: int, iy: int) -> None:
        """長押し中、パワー拡大に合わせて半透明マスクを毎フレーム更新。"""
        a_power = 0.5
        while erase_state["pressing"]:
            if state.img_gray is None or state.mask is None:
                break
            preview, removed = erase_oversegmentation_from_mask(
                state.img_gray, state.mask, (ix, iy), a_power,
            )
            erase_state["preview_mask"] = preview
            if removed > 0:
                erase_state["changed"] = True
            rgb = _build_rgb_frame(preview)
            _push_image_rgb(rgb, do_update=True, jpeg_quality=72)
            status.value = (
                f"Erase中… パワー {a_power:.1f}（削除 {removed}px）"
                if removed else f"Erase中… パワー {a_power:.1f}（明るい縁へ拡大中）"
            )
            a_power = min(6.0, a_power + 0.2)
            await asyncio.sleep(0.04)

    def on_tap_down(e: ft.TapEvent) -> None:
        if state.img_gray is None:
            return
        if state.mode == "seg_erase":
            ox, oy = _to_img_xy(e.local_x, e.local_y)
            ix, iy = int(ox), int(oy)
            erase_state["pressing"] = True
            erase_state["preview_mask"] = None
            erase_state["changed"] = False
            erase_state["seed"] = (ix, iy)
            status.value = "Erase中…（元画像の明るさで判定）"
            preview, removed = erase_oversegmentation_from_mask(
                state.img_gray, state.mask, (ix, iy), 0.5,
            )
            erase_state["preview_mask"] = preview
            if removed > 0:
                erase_state["changed"] = True
                _push_image_rgb(_build_rgb_frame(preview), do_update=True, jpeg_quality=72)

            async def _run() -> None:
                await continuous_erase(ix, iy)

            page.run_task(_run)
            return
        if state.mode != "fovea":
            return
        ox, oy = _to_img_xy(e.local_x, e.local_y)
        state.fovea_xy = (ox, oy)
        state.mode = None
        state.series_results = None
        if state.mask is not None and state.scale_um_px:
            state.stats = ga_stats(state.mask, state.scale_um_px, state.fovea_xy)
        status.value = f"中心窩: ({ox:.0f}, {oy:.0f})"
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    def on_tap_up(_e: ft.TapEvent) -> None:
        if state.mode != "seg_erase":
            return
        erase_state["pressing"] = False
        if erase_state["changed"] and erase_state["preview_mask"] is not None:
            state.mask = erase_state["preview_mask"].copy()
            status.value = "削除を適用 — 続けて別の箇所を長押し、または ✓ OK"
            state.mode = "seg_erase"
        else:
            status.value = "変化なし — 別の箇所を長押しするか ✓ OK で確定"
        erase_state["preview_mask"] = None
        erase_state["changed"] = False
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
        except ValueError as exc:
            status.value = str(exc)
            page.update()
            return
        state.disc_draft_pts = []
        state.mode = None
        state.series_results = None
        status.value = f"乳頭計測完了: {state.scale_um_px:.2f} μm/px"
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    def on_disc_redo(_=None) -> None:
        state.disc_draft_pts = []
        state.mode = "disc_line"
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
            state.crop_pts = []
            state.mode = None
            state.reset_analysis(keep_series=True)
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

    def set_mode(mode: str) -> None:
        state.mode = mode
        drag["active"] = False
        if mode == "disc_line":
            state.disc_pts = []
            state.disc_draft_pts = []
            state.scale_um_px = None
            status.value = "乳頭の両端を結ぶ直線をドラッグしてください"
        elif mode == "crop_drag":
            state.crop_pts = []
            status.value = "クロップ範囲をドラッグしてください"
        elif mode == "fovea":
            status.value = "中心窩の位置をタップしてください"
        refresh_hint()
        update_confirm_visibility()
        page.update()

    def on_analyze(_=None) -> None:
        if state.scale_um_px is None:
            status.value = "先に乳頭計測を完了してください"
        elif state.fovea_xy is None:
            status.value = "先に中心窩を指定してください"
        elif state.img_gray is None:
            status.value = "画像を読み込んでください"
        else:
            state.mask = segmentation.segment_ga(state.img_gray)
            state.stats = None
            state.mode = "seg_review"
            erase_state["pressing"] = False
            erase_state["preview_mask"] = None
            status.value = "セグメンテーション完了 — Erase で修正（複数可）、✓ OK で確定"
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    def on_seg_ok(_=None) -> None:
        if state.mask is None:
            status.value = "マスクがありません"
            page.update()
            return
        erase_state["pressing"] = False
        erase_state["preview_mask"] = None
        erase_state["changed"] = False
        state.stats = ga_stats(state.mask, state.scale_um_px, state.fovea_xy)
        state.mode = None
        status.value = "解析完了"
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    def on_seg_erase_click(_=None) -> None:
        if state.mask is None:
            status.value = "先に解析を実行してください"
            page.update()
            return
        state.mode = "seg_erase"
        status.value = "長押しで削除（複数箇所可）— ✓ OK で確定"
        refresh_hint()
        update_confirm_visibility()
        page.update()

    def on_series(_=None) -> None:
        if len(state.files) < 2:
            status.value = "時系列には2枚以上必要です"
            page.update()
            return
        if state.scale_um_px is None or state.fovea_xy is None:
            status.value = "先に乳頭計測と中心窩指定を完了してください"
            page.update()
            return
        status.value = "時系列解析中…"
        page.update()
        images = [load_gray(f) for f in state.files]
        dates = [
            datetime.fromisoformat(state.series_dates[f.name]).date()
            for f in state.files
        ]
        state.disc_diameter_um = float(disc_field.value or 1800)
        results = analyze_time_series(
            images, state.files, dates,
            state.scale_um_px, state.fovea_xy, register=True,
        )
        overlays = []
        for row in results:
            s = row["stats"]
            overlays.append(make_overlay(
                row["image"], row["mask"],
                fovea_xy=state.fovea_xy,
                nearest_xy=s.get("nearest_xy"),
                disc_pts=state.disc_pts,
                disc_diameter_um=state.disc_diameter_um,
                show_macula=True,
                scale_um_px=state.scale_um_px,
            ))
        state.series_results = results
        state.series_overlays = overlays
        state.series_gif = overlays_to_gif_bytes(overlays)
        state.estimated_prog = estimate_progression_rate_um_per_year(results)
        if state.estimated_prog:
            prog_field.value = str(int(state.estimated_prog))
        view_mode.visible = True
        series_date_dd.visible = True
        series_date_dd.options = [
            ft.dropdown.Option(key=str(i), text=f"{r['date']} — {r['filename']}")
            for i, r in enumerate(results)
        ]
        series_date_dd.value = "0"
        status.value = f"時系列完了 ({len(results)} 時点)"
        refresh_image()
        refresh_results()
        page.update()

    def on_pick_folder(_=None) -> None:
        async def _pick():
            path = await folder_picker.get_directory_path(dialog_title="入力フォルダを選択")
            if path:
                input_field.value = path
                on_load_files()
        page.run_task(_pick)

    folder_picker = ft.FilePicker(on_result=lambda _: None)
    page.overlay.append(folder_picker)

    def on_view_change(e) -> None:
        state.view_mode = e.control.value
        refresh_image()
        page.update()

    view_mode.on_change = on_view_change

    def on_series_date(e) -> None:
        if e.control.value is not None:
            state.series_time_idx = int(e.control.value)
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

    toolbar = ft.Row(
        [
            btn_toolbar("⓪ クロップ", on_click=lambda _: set_mode("crop_drag")),
            btn_toolbar("① 乳頭", on_click=lambda _: set_mode("disc_line")),
            btn_toolbar("② 中心窩", on_click=lambda _: set_mode("fovea")),
            btn_primary("③ 解析", on_click=on_analyze),
            btn_toolbar("④ 時系列", on_click=on_series),
        ],
        wrap=True,
        spacing=8,
    )

    def update_confirm_visibility() -> None:
        cancel_row.visible = state.mode in (
            "disc_line", "crop_drag", "fovea", "seg_review", "seg_erase",
        )
        disc_confirm_row.visible = state.mode == "disc_confirm"
        crop_confirm_row.visible = state.mode == "crop_confirm"
        seg_confirm_row.visible = state.mode in ("seg_review", "seg_erase")

    def on_cancel(_=None) -> None:
        if state.mode in ("seg_review", "seg_erase"):
            state.mask = None
            state.stats = None
            erase_state["pressing"] = False
            erase_state["preview_mask"] = None
            state.mode = None
        else:
            state.cancel_mode()
        drag["active"] = False
        status.value = ""
        refresh_hint()
        update_confirm_visibility()
        refresh_image()
        refresh_results()
        page.update()

    cancel_row = ft.Row([
        btn_outlined("✕ キャンセル", on_click=on_cancel),
    ], visible=False)

    disc_confirm_row = ft.Row([
        btn_ok("✓ OK", on_click=on_disc_ok),
        btn_outlined("↩ やり直し", on_click=on_disc_redo),
    ], visible=False)

    crop_confirm_row = ft.Row([
        btn_ok("✓ OK", on_click=on_crop_ok),
        btn_outlined("↩ やり直し", on_click=on_crop_redo),
    ], visible=False)

    seg_confirm_row = ft.Row([
        btn_ok("✓ OK", on_click=on_seg_ok),
        btn_erase("Erase 白領域（長押し）", on_click=on_seg_erase_click),
    ], visible=False)

    left = ft.Column(
        [
            ft.Text("GA Segmentation Pro", size=22, weight=ft.FontWeight.BOLD, color=T.PRIMARY),
            ft.Text("研究用", size=11, color=T.TEXT_MUTED),
            toolbar,
            cancel_row,
            disc_confirm_row,
            crop_confirm_row,
            seg_confirm_row,
            hint,
            status,
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
            ft.Divider(color="#374151"),
            disc_field,
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

    async def _deferred_auto_load() -> None:
        import asyncio
        import os
        await asyncio.sleep(0.05)
        auto_dir = os.environ.get("GA_INPUT_DIR", "").strip()
        if auto_dir:
            input_field.value = auto_dir
            on_load_files()

    page.run_task(_deferred_auto_load)
