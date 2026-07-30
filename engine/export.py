"""export.py – 個別 PDF / バッチ CSV 出力。"""

from __future__ import annotations

import csv
import io
import logging
import os
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fpdf import FPDF

from engine.gawain import GAWAIN_LABEL, format_gawain_display
from engine.gawain_figure import render_gawain_figure_bytes

_log = logging.getLogger(__name__)

# fpdf2 core fonts (Helvetica 等) は Latin-1 のみ。日本語 (未分類 / 乳頭…) 用。
_PDF_UNICODE_FONT = "GAUnicode"
_PDF_CORE_FONT = "Helvetica"

# macOS / Linux でよくある日本語対応フォント（先勝ち）。
_UNICODE_FONT_CANDIDATES = (
    # Optional override
    # (resolved via GA_PDF_FONT env in _resolve_unicode_font_path)
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    Path("/Library/Fonts/AppleGothic.ttf"),
    Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"),
    Path("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"),
    Path("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"),
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf"),
    Path("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansJP-Regular.otf"),
)


@lru_cache(maxsize=1)
def _resolve_unicode_font_path() -> Path | None:
    """日本語を描画できる TTF/TTC/OTF を探す。見つからなければ None。"""
    override = os.environ.get("GA_PDF_FONT", "").strip()
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return p
        _log.warning("GA_PDF_FONT not found: %s", override)
    for p in _UNICODE_FONT_CANDIDATES:
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


def _configure_pdf_font(pdf: FPDF) -> str:
    """Unicode フォントを登録してファミリー名を返す。失敗時は Helvetica。"""
    path = _resolve_unicode_font_path()
    if path is None:
        _log.warning(
            "No Unicode PDF font found; Japanese text may fail. "
            "Set GA_PDF_FONT to a TTF/OTF path."
        )
        return _PDF_CORE_FONT
    try:
        pdf.add_font(_PDF_UNICODE_FONT, "", str(path))
        # 同一ファイルを B/I としても登録（スタイル切替用; 太字合成は環境依存）
        pdf.add_font(_PDF_UNICODE_FONT, "B", str(path))
        pdf.add_font(_PDF_UNICODE_FONT, "I", str(path))
        return _PDF_UNICODE_FONT
    except Exception:
        _log.exception("Failed to register Unicode PDF font: %s", path)
        return _PDF_CORE_FONT


def _dash(val) -> str:
    if val is None:
        return "-"
    return str(val)


def _pdf_safe(text: str) -> str:
    return text.replace("\u2013", "-").replace("\u2014", "-").replace("—", "-")

def _foveal_correction_label(
    *,
    applied: bool | None,
    skipped: bool | None,
    strength: float | None,
) -> str:
    if skipped:
        return "skipped"
    if applied and strength and strength > 0:
        return f"applied ({int(strength * 100)}%)"
    return "none"


def _calibration_label(source: str | None, field_width_mm: float | None) -> str:
    from engine.calibration import calibration_source_label

    if source == "width":
        fw = _dash(field_width_mm)
        return f"field width ({fw} mm)"
    if not source:
        return "-"
    # disc / disc_1800um / dfd_4.76mm 等
    label = calibration_source_label(source)
    if label == "-":
        return "-"
    return label


def gawain_figure_png_from_context(ctx: dict) -> bytes | None:
    """解析コンテキストから GAWAIN 説明図 PNG を生成する。"""
    img = ctx.get("image")
    mask = ctx.get("mask")
    fovea = ctx.get("fovea_xy")
    scale = ctx.get("scale_um_px")
    if img is None or mask is None or fovea is None or not scale:
        return None
    title = ctx.get("filename") or ""
    png_bytes, _ = render_gawain_figure_bytes(
        img, mask, fovea, scale, title=title,
    )
    return png_bytes


def build_batch_csv_bytes(
    results: list[dict],
    roi_qc: list[dict] | None = None,
) -> bytes:
    """時系列結果を CSV バイト列にする。

    roi_qc: マスクQC補正のログ（apply_roi_correction_to_series の
    ``qc_log``）。指定時は本表の下に「ROI QC correction log」セクション
    （SOP 11章相当: raw/補正後面積・前時点Dice・異常判定・補正手法）を
    追記する。
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "case_id", "date", "exam_datetime", "filename",
        "area_mm2", "area_mm_sqrt",
        "gawain", "dist_um", "lesion_count",
        "calibration_source", "mask_source",
    ])
    for row in results:
        s = row.get("stats") or {}
        exam_dt = row.get("exam_datetime") or row.get("date") or ""
        writer.writerow([
            row.get("case_id") or "",
            row.get("date") or "",
            exam_dt,
            row.get("filename") or "",
            s.get("area_mm2"),
            s.get("area_mm_sqrt"),
            s.get("gawain"),
            s.get("dist_um"),
            s.get("lesion_count"),
            row.get("calibration_source") or "",
            row.get("mask_source") or "",
        ])

    if roi_qc:
        writer.writerow([])
        writer.writerow(["ROI QC correction log"])
        writer.writerow([
            "index", "date", "filename",
            "raw_area_px", "corrected_area_px",
            "dice_with_prev_raw", "dice_with_prev_corrected",
            "refine_method", "area_anomaly", "shape_anomaly",
            "needs_review", "reason",
        ])
        for rec in roi_qc:
            writer.writerow([
                rec.get("index"),
                rec.get("date"),
                rec.get("filename"),
                rec.get("raw_area"),
                rec.get("corrected_area"),
                rec.get("dice_with_prev_raw"),
                rec.get("dice_with_prev_corrected"),
                rec.get("refine_method") or "",
                rec.get("area_anomaly"),
                rec.get("shape_anomaly"),
                rec.get("needs_review"),
                rec.get("reason"),
            ])
    return buf.getvalue().encode("utf-8-sig")


def _write_rgb_png_tmp(overlay_rgb: np.ndarray) -> Path:
    tmp = Path(tempfile.mkstemp(suffix=".png")[1])
    cv2.imwrite(str(tmp), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    return tmp


def _write_png_bytes_tmp(png_bytes: bytes) -> Path:
    tmp = Path(tempfile.mkstemp(suffix=".png")[1])
    tmp.write_bytes(png_bytes)
    return tmp


def generate_pdf_report_bytes(
    *,
    filename: str,
    date: str,
    stats: dict,
    overlay_rgb: np.ndarray,
    gawain_figure_png: bytes | None = None,
    scale_um_px: float | None = None,
    calibration_source: str | None = None,
    field_width_mm: float | None = None,
    fovea_xy: tuple[float, float] | None = None,
    foveal_correction_applied: bool | None = None,
    foveal_correction_skipped: bool | None = None,
    foveal_correction_strength: float | None = None,
    roi_qc_note: str | None = None,
    case_id: str | None = None,
    exam_datetime: str | None = None,
) -> bytes:
    """1時点の解析結果 PDF（計測表 + オーバーレイ + 任意 GAWAIN 図）をバイト列で返す。

    roi_qc_note: マスクQC補正の1行サマリ（適用時のみ）。Summary metrics の
    直下に "Mask QC: ..." として印字される。
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    # case_id「未分類」や calibration_source_label（乳頭/幅指定）は JP を含むため
    # Helvetica ではなく Unicode フォントを使う。
    font = _configure_pdf_font(pdf)

    pdf.set_font(font, "B", 14)
    pdf.cell(0, 10, "GA Segmentation Pro - Analysis Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font, size=11)
    if case_id:
        pdf.cell(0, 7, f"Case: {_pdf_safe(case_id)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"File: {_pdf_safe(filename)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(
        0, 7,
        f"Exam datetime: {_pdf_safe(exam_datetime or date)}",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(3)

    s = stats
    pdf.set_font(font, "B", 11)
    pdf.cell(0, 7, "Summary metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font, size=10)
    for label, val in [
        ("GA Area (mm2)", s.get("area_mm2")),
        ("GA Area SQRT (mm)", s.get("area_mm_sqrt")),
        (_pdf_safe(GAWAIN_LABEL), _pdf_safe(format_gawain_display(s.get("gawain")))),
        ("Dist to fovea (um)", s.get("dist_um")),
        ("Lesion count", s.get("lesion_count")),
    ]:
        pdf.cell(0, 6, _pdf_safe(f"{label}: {_dash(val)}"), new_x="LMARGIN", new_y="NEXT")
    if roi_qc_note:
        pdf.set_font(font, "I", 9)
        pdf.cell(0, 6, _pdf_safe(f"Mask QC: {roi_qc_note}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font, size=10)

    lesions = s.get("lesions") or []
    if lesions:
        pdf.ln(2)
        pdf.set_font(font, "B", 11)
        pdf.cell(0, 7, "Per-lesion metrics", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font, size=10)
        pdf.cell(18, 6, "ID", border=1)
        pdf.cell(42, 6, "Area (mm2)", border=1)
        pdf.cell(42, 6, "Dist (um)", border=1, new_x="LMARGIN", new_y="NEXT")
        for lesion in lesions:
            pdf.cell(18, 6, _dash(lesion.get("id")), border=1)
            pdf.cell(42, 6, _dash(lesion.get("area_mm2")), border=1)
            pdf.cell(42, 6, _dash(lesion.get("dist_um")), border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)
    pdf.set_font(font, "B", 11)
    pdf.cell(0, 7, "Calibration & landmarks", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font, size=10)
    fovea_text = "-"
    if fovea_xy is not None:
        fovea_text = f"({fovea_xy[0]:.1f}, {fovea_xy[1]:.1f})"
    for label, val in [
        ("Scale (um/px)", scale_um_px),
        ("Calibration source", _calibration_label(calibration_source, field_width_mm)),
        ("Fovea (px)", fovea_text),
        (
            "Foveal correction",
            _foveal_correction_label(
                applied=foveal_correction_applied,
                skipped=foveal_correction_skipped,
                strength=foveal_correction_strength,
            ),
        ),
    ]:
        pdf.cell(0, 6, _pdf_safe(f"{label}: {_dash(val)}"), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)
    pdf.set_font(font, "B", 11)
    pdf.cell(0, 7, "Segmentation overlay", new_x="LMARGIN", new_y="NEXT")
    overlay_tmp = _write_rgb_png_tmp(overlay_rgb)
    try:
        pdf.image(str(overlay_tmp), w=170)
    finally:
        overlay_tmp.unlink(missing_ok=True)

    if gawain_figure_png:
        pdf.add_page()
        pdf.set_font(font, "B", 12)
        pdf.cell(0, 8, "GAWAIN workflow (Figure 1 A-E)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        gaw_tmp = _write_png_bytes_tmp(gawain_figure_png)
        try:
            pdf.image(str(gaw_tmp), w=190)
        finally:
            gaw_tmp.unlink(missing_ok=True)

    return bytes(pdf.output())

def suggested_csv_filename() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"GA_batch_{ts}.csv"


def suggested_pdf_filename(filename: str) -> str:
    stem = Path(filename).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"GA_{stem}_{ts}.pdf"
