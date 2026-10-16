"""GA Segmentation Optos の画面。"""

from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path

import cv2
import flet as ft
import numpy as np

from engine.optos import px_from_mm, window_box
from engine.overlay import make_overlay
from flet_ui import theme as T
from flet_ui.theme import apply_page_theme
from optos_ui.folder_dialog import choose_folder
from optos_ui.session import OptosSession

VIEW_W = 760
VIEW_H = 640


def _jpeg_b64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _empty_px() -> str:
    rgb = np.zeros((200, VIEW_W, 3), dtype=np.uint8)
    return _jpeg_b64(rgb)


async def build_optos_view(page: ft.Page, session: OptosSession) -> None:
    status = ft.Text("フォルダを開いてください", color=T.TEXT_MUTED, size=13)
    metrics = ft.Text("", color=T.TEXT, size=13, selectable=True)
    series_text = ft.Text("", color=T.ACCENT_AMBER, size=13, selectable=True)
    date_field = ft.TextField(label="検査日時", width=220, dense=True, text_size=13)
    age_field = ft.TextField(label="年齢", width=100, dense=True, text_size=13)
    sex_dd = ft.Dropdown(
        label="性別",
        width=140,
        dense=True,
        options=[ft.dropdown.Option("Male"), ft.dropdown.Option("Female")],
        value="Male",
    )
    case_dd = ft.Dropdown(label="症例", width=220, dense=True, options=[])
    visit_dd = ft.Dropdown(label="時点", width=280, dense=True, options=[])
    folder_field = ft.TextField(label="入力フォルダ", width=640, dense=True, text_size=13)
    img = ft.Image(src_base64=_empty_px(), width=VIEW_W, height=VIEW_H, fit=ft.ImageFit.FILL)
    tool_label = ft.Text("道具: 乳頭", color=T.PRIMARY, size=12)
    next_step = ft.Text(session.next_instruction(), size=15, color=T.ACCENT_AMBER, selectable=True)

    view = {"origin": (0, 0), "zoom": 1.0, "pan": (0.0, 0.0)}

    def _set_status(text: str) -> None:
        status.value = text

    def _refresh_lists() -> None:
        case_dd.options = [
            ft.dropdown.Option(key, f"{key} ({len(session.cases[key])})")
            for key in session.cases
        ]
        case_dd.value = session.case_id or None
        visit_dd.options = []
        for i, path in enumerate(session.files):
            snap = session.snap_for(path.name)
            mark = "済 " if snap is not None and snap.confirmed else ""
            if i == session.index and session.scale_um_px:
                scale_note = f"  {session.scale_um_px:.1f}µm/px"
            elif snap is not None and snap.scale_um_px:
                scale_note = f"  {snap.scale_um_px:.1f}µm/px"
            else:
                scale_note = ""
            visit_dd.options.append(
                ft.dropdown.Option(str(i), f"{mark}{path.name}{scale_note}")
            )
        if session.files:
            visit_dd.value = str(session.index)
        date_field.value = session.date_iso
        age_field.value = session.age
        sex_dd.value = session.sex

    def _paint() -> None:
        next_step.value = session.next_instruction()
        if session.gray is None:
            img.src_base64 = _empty_px()
            img.width = VIEW_W
            img.height = 200
            metrics.value = ""
            return
        rgb, origin, src_zoom = _compose()
        view["origin"] = origin
        view["zoom"] = src_zoom
        shown = _viewport(rgb)
        img.src_base64 = _jpeg_b64(shown)
        img.width = shown.shape[1]
        img.height = shown.shape[0]
        stats = session.live_stats()
        if stats is None:
            metrics.value = "面積は中心 6 mm だけを数えます"
        else:
            scale = f"{session.scale_um_px:.2f}" if session.scale_um_px else "—"
            metrics.value = (
                f"中心6mm {stats['area_mm2']:.3f} mm²"
                f"  ·  √{stats['area_mm_sqrt']:.2f} mm"
                f"  ·  病変 {stats['lesion_count']}"
                f"  ·  GAWAIN {stats['gawain']}"
                f"  ·  中心窩まで {stats['dist_um']} µm"
                f"\n6mm の外 {stats['outside_quant_px']} px は面積に入れていません"
                f"  ·  スケール {scale} µm/px"
            )
        series_text.value = session.series_note
        tool_label.value = f"道具: {_tool_name(session.tool)}"
        _refresh_lists()

    def _compose() -> tuple[np.ndarray, tuple[int, int], float]:
        assert session.gray is not None
        gray = session.gray
        mask = session.mask if session.mask is not None else np.zeros(gray.shape[:2], dtype=np.uint8)
        if session.fovea_xy is not None and session.scale_um_px:
            x0, y0, x1, y1 = window_box(gray.shape[:2], session.fovea_xy, session.scale_um_px)
            crop_g = gray[y0:y1, x0:x1]
            crop_m = mask[y0:y1, x0:x1]
            fov = (session.fovea_xy[0] - x0, session.fovea_xy[1] - y0)
            rgb = make_overlay(crop_g, crop_m, fovea_xy=fov, mask_style="blend")
            r6 = px_from_mm(6.0, session.scale_um_px)
            cv2.circle(rgb, (int(fov[0]), int(fov[1])), r6, (34, 211, 238), 1)
            if session.disc_xy is not None:
                dc = (int(session.disc_xy[0] - x0), int(session.disc_xy[1] - y0))
                cv2.circle(rgb, dc, px_from_mm(0.9, session.scale_um_px), (251, 191, 36), 1)
            return rgb, (x0, y0), float(session.zoom)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        if session.disc_xy is not None:
            cv2.circle(rgb, (int(session.disc_xy[0]), int(session.disc_xy[1])), 8, (251, 191, 36), 2)
        fit = min(VIEW_W / rgb.shape[1], VIEW_H / rgb.shape[0])
        return rgb, (0, 0), fit

    def _viewport(rgb: np.ndarray) -> np.ndarray:
        z = float(view["zoom"])
        if session.fovea_xy is None:
            zw, zh = max(1, int(rgb.shape[1] * z)), max(1, int(rgb.shape[0] * z))
            return cv2.resize(rgb, (zw, zh), interpolation=cv2.INTER_AREA)
        zw, zh = max(1, int(rgb.shape[1] * z)), max(1, int(rgb.shape[0] * z))
        big = cv2.resize(rgb, (zw, zh), interpolation=cv2.INTER_NEAREST)
        px, py = session.pan_xy
        x0 = int(np.clip(px * z, 0, max(0, zw - VIEW_W)))
        y0 = int(np.clip(py * z, 0, max(0, zh - VIEW_H)))
        view["pan"] = (x0, y0)
        return big[y0:y0 + VIEW_H, x0:x0 + VIEW_W].copy()

    def _to_image(local_x: float, local_y: float) -> tuple[float, float]:
        ox, oy = view["origin"]
        z = float(view["zoom"]) or 1.0
        if session.fovea_xy is None:
            return ox + local_x / z, oy + local_y / z
        pan_x, pan_y = view["pan"]
        return ox + (pan_x + local_x) / z, oy + (pan_y + local_y) / z

    def _after(text: str) -> None:
        if text:
            _set_status(text)
        _paint()
        page.update()

    def _open_typed(path: str) -> None:
        folder = Path(path).expanduser()
        if not folder.is_dir():
            _after(f"フォルダが見つかりません: {folder}")
            return
        folder_field.value = str(folder)
        _after(session.load_folder(str(folder)))

    async def _choose_and_open() -> None:
        _set_status("フォルダ選択を開きました。Safari の後ろに出ていたら手前へ")
        page.update()
        try:
            path = await asyncio.to_thread(choose_folder)
        except Exception as exc:
            _after(f"フォルダ選択に失敗しました: {exc}")
            return
        if not path:
            _after("フォルダ選択をキャンセルしました")
            return
        _open_typed(path)

    async def on_pick_folder(_=None) -> None:
        if sys.platform == "darwin":
            await _choose_and_open()
            return
        if page.web:
            _after("入力欄にフォルダの絶対パスを入れて「開く」を押してください")
            return
        picker.get_directory_path(dialog_title="Optos 画像フォルダ")

    def on_folder_result(e: ft.FilePickerResultEvent) -> None:
        if not e.path:
            _after("フォルダは選ばれませんでした")
            return
        _open_typed(e.path)

    async def on_load(_=None) -> None:
        typed = (folder_field.value or "").strip()
        if not typed:
            await on_pick_folder()
            return
        _open_typed(typed)

    def on_case(e) -> None:
        if e.control.value:
            session.select_case(e.control.value)
            _after(session.files[session.index].name if session.files else "")

    def on_visit(e) -> None:
        if e.control.value is None:
            return
        _after(session.open_index(int(e.control.value)))

    def on_date_submit(_e) -> None:
        try:
            _after(session.set_date_text(date_field.value or ""))
        except ValueError as exc:
            _after(str(exc))

    def on_tool(name: str):
        def _click(_=None) -> None:
            session.tool = name
            session.clear_pts = []
            _after(f"道具を {_tool_name(name)} にしました")
        return _click

    def on_zoom(level: int):
        def _click(_=None) -> None:
            session.zoom = level
            session.pan_xy = (0.0, 0.0)
            _after(f"表示 {level}x")
        return _click

    def on_reseg(_=None) -> None:
        _after(session.resegment())

    def on_confirm(_=None) -> None:
        _after(session.confirm())

    def on_undo(_=None) -> None:
        _after(session.undo_mask())

    def on_redo(_=None) -> None:
        _after(session.redo_mask())

    def on_close_tool(_=None) -> None:
        session.close_tool()
        _after("道具を閉じました。マスクはそのままです")

    def on_series(_=None) -> None:
        session.age = age_field.value or ""
        session.sex = sex_dd.value or "Male"
        _after(session.run_series())

    def on_tap_down(e: ft.TapEvent) -> None:
        if session.gray is None:
            return
        xy = _to_image(e.local_x, e.local_y)
        if session.tool == "disc":
            _after(session.place_disc(xy))
        elif session.tool == "fovea":
            _after(session.place_fovea(xy))
        elif session.tool == "grow":
            _after(session.edit_at(xy))
        elif session.tool in ("brush_add", "brush_erase"):
            _after(session.edit_at(xy))
        elif session.tool == "clear":
            session.begin_clear(xy)

    def on_pan_start(e: ft.DragStartEvent) -> None:
        if session.tool in ("brush_add", "brush_erase"):
            session.edit_at(_to_image(e.local_x, e.local_y), record=True)
        elif session.tool == "clear":
            session.begin_clear(_to_image(e.local_x, e.local_y))
        elif session.tool == "pan" and session.fovea_xy is not None:
            view["drag"] = (e.local_x, e.local_y, session.pan_xy)

    def on_pan_update(e: ft.DragUpdateEvent) -> None:
        if session.tool in ("brush_add", "brush_erase"):
            session.edit_at(_to_image(e.local_x, e.local_y), record=False)
            _paint()
            page.update()
        elif session.tool == "clear":
            session.extend_clear(_to_image(e.local_x, e.local_y))
        elif session.tool == "pan" and session.fovea_xy is not None and "drag" in view:
            sx, sy, (px, py) = view["drag"]
            z = float(session.zoom) or 1.0
            session.pan_xy = (px - (e.local_x - sx) / z, py - (e.local_y - sy) / z)
            _paint()
            page.update()

    def on_pan_end(_e: ft.DragEndEvent) -> None:
        if session.tool == "clear":
            _after(session.finish_clear())

    picker = ft.FilePicker(on_result=on_folder_result)
    page.overlay.append(picker)

    def _btn(label: str, handler) -> ft.ElevatedButton:
        return ft.ElevatedButton(label, on_click=handler, height=34, style=ft.ButtonStyle(text_style=ft.TextStyle(size=12)))

    gesture = ft.GestureDetector(
        content=img,
        on_tap_down=on_tap_down,
        on_pan_start=on_pan_start,
        on_pan_update=on_pan_update,
        on_pan_end=on_pan_end,
        mouse_cursor=ft.MouseCursor.PRECISE,
    )

    page.add(
        ft.Column(
            [
                ft.Text("GA Segmentation Optos", size=22, weight=ft.FontWeight.BOLD, color=T.PRIMARY),
                ft.Row(
                    [
                        folder_field,
                        _btn("開く", on_load),
                        _btn("フォルダ選択", on_pick_folder),
                    ],
                    spacing=8,
                ),
                ft.Row([case_dd, visit_dd, date_field, _btn("日時を反映", on_date_submit)], spacing=8),
                ft.Row(
                    [
                        gesture,
                        ft.Container(
                            width=420,
                            bgcolor=T.PANEL_BG,
                            border_radius=8,
                            padding=12,
                            content=ft.Column(
                                [
                                    ft.Text("次にすること", size=12, color=T.TEXT_MUTED),
                                    next_step,
                                    ft.Divider(color="#374151"),
                                    tool_label,
                                    ft.Row([
                                        _btn("乳頭", on_tool("disc")),
                                        _btn("中心窩", on_tool("fovea")),
                                        _btn("移動", on_tool("pan")),
                                    ], wrap=True, spacing=6),
                                    ft.Row([
                                        _btn("縁まで", on_tool("grow")),
                                        _btn("塗る", on_tool("brush_add")),
                                        _btn("消す", on_tool("brush_erase")),
                                        _btn("範囲消去", on_tool("clear")),
                                    ], wrap=True, spacing=6),
                                    ft.Row([
                                        _btn("1x", on_zoom(1)),
                                        _btn("2x", on_zoom(2)),
                                        _btn("4x", on_zoom(4)),
                                        _btn("道具を閉じる", on_close_tool),
                                    ], wrap=True, spacing=6),
                                    ft.Row([
                                        _btn("再セグ", on_reseg),
                                        _btn("取消", on_undo),
                                        _btn("やり直し", on_redo),
                                        _btn("この時点を確定", on_confirm),
                                    ], wrap=True, spacing=6),
                                    ft.Divider(color="#374151"),
                                    metrics,
                                    ft.Text(
                                        "黄斑窓のシアン円が定量する 6 mm です。黄の円は乳頭です。スケールは時点ごとの乳頭–中心窩（4.76 mm）で、ズームが違っても共有しません。",
                                        size=11,
                                        color=T.TEXT_MUTED,
                                    ),
                                    ft.Divider(color="#374151"),
                                    ft.Row([age_field, sex_dd], spacing=8),
                                    _btn("時系列を計算", on_series),
                                    series_text,
                                ],
                                spacing=8,
                                scroll=ft.ScrollMode.AUTO,
                            ),
                        ),
                    ],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
                status,
            ],
            spacing=10,
            expand=True,
            scroll=ft.ScrollMode.AUTO,
        )
    )
    case_dd.on_change = on_case
    visit_dd.on_change = on_visit
    _paint()
    page.update()


def _tool_name(tool: str) -> str:
    return {
        "disc": "乳頭",
        "fovea": "中心窩",
        "pan": "移動",
        "grow": "縁まで",
        "brush_add": "塗る",
        "brush_erase": "消す",
        "clear": "範囲消去",
    }.get(tool, tool)
