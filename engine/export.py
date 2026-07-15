"""export.py – 個別 PDF / バッチ CSV 出力。"""

from __future__ import annotations

import csv
import io
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fpdf import FPDF

from engine.gawain import GAWAIN_LABEL, format_gawain_display
from engine.gawain_figure import render_gawain_figure_bytes


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
    if source == "disc":
        return "optic disc line"
    if source == "width":
        fw = _dash(field_width_mm)
        return f"field width ({fw} mm)"
    return "-"


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
        "date", "filename", "area_mm2", "area_mm_sqrt",
        "gawain", "dist_um", "lesion_count",
    ])
    for row in results:
        s = row["stats"]
        writer.writerow([
            row["date"],
            row["filename"],
            s.get("area_mm2"),
            s.get("area_mm_sqrt"),
            s.get("gawain"),
            s.get("dist_um"),
            s.get("lesion_count"),
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
) -> bytes:
    """1時点の解析結果 PDF（計測表 + オーバーレイ + 任意 GAWAIN 図）をバイト列で返す。

    roi_qc_note: マスクQC補正の1行サマリ（適用時のみ）。Summary metrics の
    直下に "Mask QC: ..." として印字される。
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "GA Segmentation Pro - Analysis Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 7, f"File: {_pdf_safe(filename)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Date: {_pdf_safe(date)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    s = stats
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Summary metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    for label, val in [
        ("GA Area (mm2)", s.get("area_mm2")),
        ("GA Area SQRT (mm)", s.get("area_mm_sqrt")),
        (_pdf_safe(GAWAIN_LABEL), _pdf_safe(format_gawain_display(s.get("gawain")))),
        ("Dist to fovea (um)", s.get("dist_um")),
        ("Lesion count", s.get("lesion_count")),
    ]:
        pdf.cell(0, 6, _pdf_safe(f"{label}: {_dash(val)}"), new_x="LMARGIN", new_y="NEXT")
    if roi_qc_note:
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, _pdf_safe(f"Mask QC: {roi_qc_note}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)

    lesions = s.get("lesions") or []
    if lesions:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Per-lesion metrics", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        pdf.cell(18, 6, "ID", border=1)
        pdf.cell(42, 6, "Area (mm2)", border=1)
        pdf.cell(42, 6, "Dist (um)", border=1, new_x="LMARGIN", new_y="NEXT")
        for lesion in lesions:
            pdf.cell(18, 6, _dash(lesion.get("id")), border=1)
            pdf.cell(42, 6, _dash(lesion.get("area_mm2")), border=1)
            pdf.cell(42, 6, _dash(lesion.get("dist_um")), border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Calibration & landmarks", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
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
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Segmentation overlay", new_x="LMARGIN", new_y="NEXT")
    overlay_tmp = _write_rgb_png_tmp(overlay_rgb)
    try:
        pdf.image(str(overlay_tmp), w=170)
    finally:
        overlay_tmp.unlink(missing_ok=True)

    if gawain_figure_png:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
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
