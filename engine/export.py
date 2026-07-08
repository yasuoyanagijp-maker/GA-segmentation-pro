"""export.py – 個別 PDF / バッチ CSV 出力。"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fpdf import FPDF

from engine.gawain import GAWAIN_LABEL, format_gawain_display


def _dash(val) -> str:
    if val is None:
        return "-"
    return str(val)


def build_batch_csv_bytes(results: list[dict]) -> bytes:
    """時系列結果を CSV バイト列にする。"""
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
    return buf.getvalue().encode("utf-8-sig")


def _pdf_safe(text: str) -> str:
    return text.replace("\u2013", "-").replace("\u2014", "-").replace("—", "-")


def generate_pdf_report(
    *,
    filename: str,
    date: str,
    stats: dict,
    overlay_rgb: np.ndarray,
    output_path: Path,
) -> Path:
    """1時点の解析結果 PDF を保存。"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "GA Segmentation Pro - Analysis Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 7, f"File: {filename}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Date: {date}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    s = stats
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    for label, val in [
        ("GA Area (mm2)", s.get("area_mm2")),
        ("GA Area SQRT (mm)", s.get("area_mm_sqrt")),
        (_pdf_safe(GAWAIN_LABEL), _pdf_safe(format_gawain_display(s.get("gawain")))),
        ("Dist to fovea (um)", s.get("dist_um")),
        ("Lesion count", s.get("lesion_count")),
    ]:
        pdf.cell(0, 6, _pdf_safe(f"{label}: {_dash(val)}"), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    tmp = output_path.parent / f"_overlay_{output_path.stem}.png"
    cv2.imwrite(str(tmp), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    try:
        pdf.image(str(tmp), w=170)
    finally:
        tmp.unlink(missing_ok=True)

    pdf.output(str(output_path))
    return output_path


def suggested_csv_filename() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"GA_batch_{ts}.csv"


def suggested_pdf_filename(filename: str) -> str:
    stem = Path(filename).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"GA_{stem}_{ts}.pdf"
