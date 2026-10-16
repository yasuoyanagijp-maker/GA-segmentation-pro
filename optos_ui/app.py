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
from ga_reading.bundle import stamped_dir, write_grader_bundle, write_reader_csv
from ga_reading.merge import merge_graders
from ga_reading.queue import find_first_grader_csv, recheck_images, second_reader_images
from ga_reading.triad import resolve_triad
from optos_ui.csv_export import SERIES_COLUMNS, VISIT_COLUMNS, csv_bytes, download_name, write_download
from optos_ui.folder_dialog import choose_file, choose_folder
from optos_ui.session import OptosSession

_ROLE_FIRST = "GRADER1"
_ROLE_SECOND = "READER2"
_ROLE_FINAL = "FINAL"

VIEW_W = 760
VIEW_H = 640


def analysis_display_zoom(scale_um_px: float, user_zoom: float) -> float:
    """1x で中心 6 mm の円が画面に収まる。2x・4x はその倍率。"""
    diameter = max(1.0, 2.0 * px_from_mm(6.0, scale_um_px))
    fit = min(VIEW_W / (diameter * 1.08), VIEW_H / (diameter * 1.08), 1.0)
    return fit * float(user_zoom)


def centered_crop_origin(
    center_xy: tuple[float, float],
    pan_xy: tuple[float, float],
    zoom: float,
    image_wh: tuple[int, int],
) -> tuple[int, int]:
    """拡大後画像で、center がビュー中央に来る切り出し原点。"""
    zw, zh = image_wh
    cx = (float(center_xy[0]) + float(pan_xy[0])) * float(zoom)
    cy = (float(center_xy[1]) + float(pan_xy[1])) * float(zoom)
    x0 = int(np.clip(round(cx - VIEW_W / 2), 0, max(0, zw - VIEW_W)))
    y0 = int(np.clip(round(cy - VIEW_H / 2), 0, max(0, zh - VIEW_H)))
    return x0, y0


# 見出しと値を同じ幅にする。日付を含めて右の数字がずれないようにする。
_SERIES_COLS = (
    ("date", "日付", None, 92, "left"),
    ("area_mm_sqrt", "√面積", 2, 50, "right"),
    ("gawain", "GAWAIN", 1, 56, "right"),
    ("dist_um_nearest", "最近傍", 0, 52, "right"),
    ("gap_um", "縁まで", 0, 52, "right"),
    ("intact_1mm_mm2", "中心1mm", 2, 64, "right"),
)


def _series_cell(text: str, width: int, *, align: str, color: str, bold: bool = False) -> ft.Container:
    return ft.Container(
        width=width,
        content=ft.Text(
            text,
            size=12,
            color=color,
            weight=ft.FontWeight.BOLD if bold else None,
            text_align=ft.TextAlign.RIGHT if align == "right" else ft.TextAlign.LEFT,
            no_wrap=True,
            font_family="monospace",
        ),
    )


def _num(value: object, digits: int) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


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
    series_detail = ft.Column(spacing=2)
    series_warn = ft.Text("", color=T.ACCENT_AMBER, size=12, selectable=True)
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
    reading = {"scan_root": "", "first_csv": "", "second_csv": "", "summary": "", "final_csv": ""}
    role_dd = ft.Dropdown(
        label="読影担当",
        width=360,
        dense=True,
        value=_ROLE_FIRST,
        options=[
            ft.dropdown.Option(_ROLE_FIRST, "第1グレーダー"),
            ft.dropdown.Option(_ROLE_SECOND, "第2リーダー"),
            ft.dropdown.Option(_ROLE_FINAL, "最終読影者"),
        ],
    )
    role_hint = ft.Text(
        "画像フォルダを開いて計測し、終わったら「第1読影を書き出す」。そのフォルダが export の親になります。",
        size=12,
        color=T.TEXT_MUTED,
        selectable=True,
    )
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
        _fill_series_detail()
        tool_label.value = f"道具: {_tool_name(session.tool)}"
        _refresh_lists()

    def _fill_series_detail() -> None:
        series_detail.controls.clear()
        table = session.series_visit_table
        if not table:
            series_warn.value = ""
            return
        series_detail.controls.append(
            ft.Row(
                [
                    _series_cell(label, width, align=align, color=T.TEXT_MUTED)
                    for _key, label, _digits, width, align in _SERIES_COLS
                ],
                spacing=4,
            )
        )
        any_rev = False
        for row in table:
            rev = set(row.get("reversed") or [])
            any_rev = any_rev or bool(rev)
            cells = []
            for key, _label, digits, width, align in _SERIES_COLS:
                if key == "date":
                    shown = str(row.get("date") or "—")
                    marked = False
                else:
                    shown = _num(row.get(key), digits or 0)
                    marked = key in rev
                cells.append(
                    _series_cell(
                        shown,
                        width,
                        align=align,
                        color=T.ACCENT_AMBER if marked else T.TEXT,
                        bold=marked,
                    )
                )
            series_detail.controls.append(ft.Row(cells, spacing=4))
        series_warn.value = (
            "黄色は前の時点より逆行しています。その時点のマスクと中心窩を見直して、再セグしてください。"
            if any_rev
            else ""
        )

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
            return rgb, (x0, y0), analysis_display_zoom(session.scale_um_px, session.zoom)
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
        zw, zh = max(1, int(round(rgb.shape[1] * z))), max(1, int(round(rgb.shape[0] * z)))
        interp = cv2.INTER_AREA if z < 1 else cv2.INTER_NEAREST
        big = cv2.resize(rgb, (zw, zh), interpolation=interp)
        ox, oy = view["origin"]
        center = (session.fovea_xy[0] - ox, session.fovea_xy[1] - oy)
        x0, y0 = centered_crop_origin(center, session.pan_xy, z, (zw, zh))
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
            await on_reading_start()
            return
        if _role() == _ROLE_SECOND:
            try:
                images = second_reader_images(Path(typed))
                reading["scan_root"] = str(Path(typed).resolve())
                found = find_first_grader_csv(Path(typed))
                reading["first_csv"] = str(found) if found else ""
                _after(session.load_paths(images, label="第2") + "。第1のマスクは開きません。")
            except ValueError as exc:
                _after(str(exc))
            return
        if _role() == _ROLE_FINAL:
            try:
                summary = Path(typed)
                images = recheck_images(summary)
                reading["summary"] = str(summary.resolve())
                reading["scan_root"] = str(summary.resolve().parent.parent)
                _after(session.load_paths(images, label="最終") + f"。RECHECK {len(images)} 画像。")
            except ValueError as exc:
                _after(str(exc))
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

    def _sync_age() -> None:
        session.age = age_field.value or ""
        session.sex = sex_dd.value or "Male"

    def on_series(_=None) -> None:
        _sync_age()
        _after(session.run_series())

    def on_csv_visits(_=None) -> None:
        rows = session.visit_rows()
        if not rows:
            _after("マスクのある時点がありません")
            return
        path = write_download(download_name(session.case_id, "visits"), csv_bytes(VISIT_COLUMNS, rows))
        _after(f"個別CSV: {path}")

    def on_csv_series(_=None) -> None:
        _sync_age()
        note = session.run_series()
        if session.series_summary is None:
            _after(note)
            return
        path = write_download(
            download_name(session.case_id, "series"),
            csv_bytes(SERIES_COLUMNS, [session.series_summary]),
        )
        _after(f"時系列CSV: {path}")

    def _role() -> str:
        return role_dd.value or _ROLE_FIRST

    def _show(title: str, body: str) -> None:
        dlg = ft.AlertDialog(
            title=ft.Text(title),
            content=ft.Container(width=560, content=ft.Text(body, selectable=True, size=12)),
        )

        def _close(_=None) -> None:
            if hasattr(page, "close"):
                page.close(dlg)
            else:
                dlg.open = False
                page.update()

        dlg.actions = [ft.TextButton("閉じる", on_click=_close)]
        if hasattr(page, "open"):
            page.open(dlg)
        else:
            page.dialog = dlg
            dlg.open = True
            page.update()

    def _scan_root() -> Path:
        root = Path(reading["scan_root"] or session.folder or ".").resolve()
        if root.name == "images" and root.parent.name == "export":
            return root.parent.parent
        return root

    def _apply_role(_=None) -> None:
        role = _role()
        if role == _ROLE_FIRST:
            role_hint.value = (
                "画像フォルダを開いて計測し、終わったら「第1読影を書き出す」。"
                "そのフォルダが export の親になります。"
            )
            start_btn.text = "画像フォルダを開く"
        elif role == _ROLE_SECOND:
            role_hint.value = (
                "第1の出力フォルダ（export の親）を開きます。マスクは読まず、同じ画像を測り直します。"
                "保存したあと「統合解析データ」で採用 CSV を作ります。"
            )
            start_btn.text = "第1の出力フォルダを開く"
        else:
            role_hint.value = (
                "RECHECK の *_summary.md を選ぶと、その画像だけが開きます。"
                "保存したあと「トライアッド確定」で中央値を書き出します。他の読影値は画面に出しません。"
            )
            start_btn.text = "summary.md を選ぶ"
        first_export.visible = role == _ROLE_FIRST
        second_export.visible = role == _ROLE_SECOND
        merge_btn.visible = role == _ROLE_SECOND
        final_export.visible = role == _ROLE_FINAL
        triad_btn.visible = role == _ROLE_FINAL
        page.update()

    async def on_reading_start(_=None) -> None:
        role = _role()
        if role == _ROLE_FIRST:
            await on_pick_folder()
            return
        try:
            if role == _ROLE_SECOND:
                path = await asyncio.to_thread(
                    choose_folder,
                    "第1グレーダーの出力フォルダ（export の親）を選んでください",
                )
                if not path:
                    _after("フォルダ選択をキャンセルしました")
                    return
                images = second_reader_images(Path(path))
                reading["scan_root"] = str(Path(path).resolve())
                found = find_first_grader_csv(Path(path))
                reading["first_csv"] = str(found) if found else ""
                folder_field.value = reading["scan_root"]
                note = session.load_paths(images, label="第2") + "。第1のマスクは開きません。"
                if found:
                    note += f" 統合には {found.name} を使います。"
                else:
                    note += " 第1の grader.csv は統合のときに再検索します。"
                _after(note)
                return
            path = await asyncio.to_thread(choose_file, "RECHECK の summary.md を選んでください")
            if not path:
                _after("ファイル選択をキャンセルしました")
                return
            summary = Path(path)
            images = recheck_images(summary)
            reading["summary"] = str(summary.resolve())
            reading["scan_root"] = str(summary.resolve().parent.parent)
            folder_field.value = reading["summary"]
            _after(
                session.load_paths(images, label="最終")
                + f"。RECHECK {len(images)} 画像。他の読影値は表示しません。"
            )
        except (ValueError, OSError, RuntimeError) as exc:
            _after(str(exc))

    async def on_export_first(_=None) -> None:
        rows = session.reading_payload()
        if not rows:
            _after("確定した計測がありません")
            return
        path = await asyncio.to_thread(
            choose_folder,
            "第1読影の出力フォルダを選んでください。ここが export の親になります",
        )
        if not path:
            _after("出力フォルダの選択をキャンセルしました")
            return
        saved = write_grader_bundle(Path(path), rows)
        reading["scan_root"] = str(Path(path).resolve())
        reading["first_csv"] = str(saved)
        _after(f"第1読影を書き出しました。第2はこのフォルダを開きます: {path}")

    def on_export_second(_=None) -> None:
        rows = session.reading_payload()
        if not rows:
            _after("確定した計測がありません")
            return
        scan = _scan_root()
        saved = write_reader_csv(stamped_dir(scan, "second_reader_output"), rows)
        reading["scan_root"] = str(scan)
        reading["second_csv"] = str(saved)
        merge_btn.disabled = False
        _after(f"第2読影を保存しました。統合解析データが使えます: {saved}")

    def on_merge(_=None) -> None:
        second = reading["second_csv"]
        if not second or not Path(second).is_file():
            _after("先に第2読影を保存してください")
            return
        first = reading["first_csv"]
        if not first or not Path(first).is_file():
            found = find_first_grader_csv(Path(reading["scan_root"]))
            first = str(found) if found else ""
        if not first:
            _show(
                "統合解析データ",
                "第1の grader.csv が見つかりません。第1の出力フォルダ（export の親）を開いたか確認してください。",
            )
            return
        out = stamped_dir(Path(reading["scan_root"]), "integrated_output")
        try:
            summary = merge_graders(Path(first), Path(second), out, prefix="GA")
        except ValueError as exc:
            _after(str(exc))
            return
        _show(
            "統合解析データを作成しました",
            "\n".join([
                f"突合 {summary['n_matched']} 行",
                f"RECHECK {summary['recheck_cells']} セル / {summary['recheck_files']} 画像",
                f"採用: {summary['adopted_csv']}",
                f"参考（提出しない）: {summary['avg_fallback_csv']}",
                f"最終読影者はこれを選びます: {summary['summary_md']}",
            ]),
        )
        _set_status(f"統合しました: {summary['summary_md']}")
        page.update()

    def on_export_final(_=None) -> None:
        rows = session.reading_payload()
        if not rows:
            _after("確定した計測がありません")
            return
        if not reading["summary"]:
            _after("先に summary.md を選んでください")
            return
        scan = Path(reading["summary"]).resolve().parent.parent
        saved = write_reader_csv(stamped_dir(scan, "final_reader_output"), rows)
        reading["scan_root"] = str(scan)
        reading["final_csv"] = str(saved)
        triad_btn.disabled = False
        _after(f"最終読影を保存しました。トライアッド確定が使えます: {saved}")

    def on_triad(_=None) -> None:
        final_csv = reading["final_csv"]
        summary_md = reading["summary"]
        if not final_csv or not Path(final_csv).is_file():
            _after("先に最終読影を保存してください")
            return
        if not summary_md:
            _after("summary.md が選ばれていません")
            return
        out = Path(final_csv).parent
        try:
            preview = resolve_triad(Path(summary_md), Path(final_csv), out, dry_run=True)
        except ValueError as exc:
            _after(str(exc))
            return
        holder: dict = {}

        def _close(_=None) -> None:
            if hasattr(page, "close"):
                page.close(holder["dlg"])
            else:
                holder["dlg"].open = False
                page.update()

        def _write(_=None) -> None:
            _close()
            try:
                result = resolve_triad(Path(summary_md), Path(final_csv), out)
            except ValueError as exc:
                _after(str(exc))
                return
            _after(f"確定しました: {result['triad_csv']}")

        holder["dlg"] = ft.AlertDialog(
            modal=True,
            title=ft.Text("トライアッド確定"),
            content=ft.Container(
                width=560,
                content=ft.Text(
                    "\n".join([
                        f"RECHECK 画像 {preview['recheck_images']}",
                        f"中央値で埋めるセル {preview['filled_cells']}",
                        f"要レビュー {preview['review_cells']}",
                        "",
                        *preview["preview"][:30],
                        "",
                        "この内容で書き出します。すでに採用したセルと、既存の採用 CSV はそのままです。",
                    ]),
                    selectable=True,
                    size=12,
                ),
            ),
            actions=[
                ft.TextButton("キャンセル", on_click=_close),
                ft.ElevatedButton("本確定", on_click=_write),
            ],
        )
        if hasattr(page, "open"):
            page.open(holder["dlg"])
        else:
            page.dialog = holder["dlg"]
            holder["dlg"].open = True
            page.update()

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
            z = float(view["zoom"]) or 1.0
            session.pan_xy = (px - (e.local_x - sx) / z, py - (e.local_y - sy) / z)
            _paint()
            page.update()

    def on_pan_end(_e: ft.DragEndEvent) -> None:
        if session.tool == "clear":
            _after(session.finish_clear())

    start_btn = ft.ElevatedButton(
        "画像フォルダを開く",
        on_click=lambda _: page.run_task(on_reading_start),
        height=36,
    )
    first_export = ft.ElevatedButton(
        "第1読影を書き出す",
        on_click=lambda _: page.run_task(on_export_first),
        height=34,
    )
    second_export = ft.ElevatedButton(
        "第2読影を保存",
        on_click=on_export_second,
        height=34,
        visible=False,
    )
    merge_btn = ft.ElevatedButton(
        "統合解析データ",
        on_click=on_merge,
        height=34,
        visible=False,
        disabled=True,
    )
    final_export = ft.ElevatedButton(
        "最終読影を保存",
        on_click=on_export_final,
        height=34,
        visible=False,
    )
    triad_btn = ft.ElevatedButton(
        "トライアッド確定",
        on_click=on_triad,
        height=34,
        visible=False,
        disabled=True,
    )
    role_dd.on_change = _apply_role

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
                ft.Row([role_dd, start_btn], spacing=8),
                role_hint,
                ft.Row(
                    [
                        folder_field,
                        _btn("開く", on_load),
                        _btn("フォルダ選択", lambda _: page.run_task(on_reading_start)),
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
                                    ft.Row([
                                        _btn("個別CSV", on_csv_visits),
                                        _btn("時系列CSV", on_csv_series),
                                    ], spacing=6),
                                    first_export,
                                    second_export,
                                    merge_btn,
                                    final_export,
                                    triad_btn,
                                    series_text,
                                    series_detail,
                                    series_warn,
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
