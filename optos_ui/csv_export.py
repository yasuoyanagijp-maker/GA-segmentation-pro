"""Optos の時点表と時系列を CSV にする。"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path

# 進行すると増える列と、減る列。GAWAIN は残存網膜の指数なので減る。
PROGRESS_UP = ("area_mm_sqrt",)
PROGRESS_DOWN = ("gawain", "dist_um_nearest", "gap_um", "intact_1mm_mm2")
REVERSAL_TOLERANCE = {
    "area_mm_sqrt": 0.01,
    "gawain": 0.1,
    "dist_um_nearest": 1.0,
    "gap_um": 1.0,
    "intact_1mm_mm2": 0.01,
}


def mark_reversals(rows: list[dict]) -> list[dict]:
    """前の時点より進行と逆向きなら、その列名を reversed に入れる。"""
    out: list[dict] = []
    previous: dict | None = None
    for row in rows:
        flagged: list[str] = []
        if previous is not None:
            for key in PROGRESS_UP:
                flagged.extend(_reversed(previous, row, key, increase=True))
            for key in PROGRESS_DOWN:
                flagged.extend(_reversed(previous, row, key, increase=False))
        copied = dict(row)
        copied["reversed"] = flagged
        out.append(copied)
        previous = row
    return out


def _reversed(previous: dict, row: dict, key: str, *, increase: bool) -> list[str]:
    before = previous.get(key)
    after = row.get(key)
    if before is None or after is None:
        return []
    delta = float(after) - float(before)
    tol = REVERSAL_TOLERANCE[key]
    if increase and delta < -tol:
        return [key]
    if not increase and delta > tol:
        return [key]
    return []


VISIT_COLUMNS = (
    "case_id",
    "date",
    "filename",
    "confirmed",
    "scale_um_per_px",
    "disc_x",
    "disc_y",
    "fovea_x",
    "fovea_y",
    "area_mm2",
    "area_mm2_24mm",
    "area_mm_sqrt",
    "lesion_count",
    "gawain",
    "dist_um_nearest",
    "gap_um",
    "intact_1mm_mm2",
    "outside_6mm_px",
)

SERIES_COLUMNS = (
    "case_id",
    "n_visits",
    "date_first",
    "date_last",
    "area_mm2_per_year",
    "area_mm_sqrt_per_year",
    "gap_um_latest",
    "gap_um_per_year",
    "years_to_fovea",
    "foveal_note",
    "intact_1mm_first_mm2",
    "intact_1mm_last_mm2",
    "intact_1mm_mm2_per_year",
    "age",
    "sex",
    "life_expectancy_years",
)


def csv_bytes(columns: tuple[str, ...], rows: list[dict]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in columns})
    return buf.getvalue().encode("utf-8-sig")


def download_name(case_id: str, kind: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (case_id or "case"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"GA_optos_{safe}_{kind}_{stamp}.csv"


def write_download(filename: str, data: bytes) -> Path:
    dest = Path.home() / "Downloads"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / filename
    path.write_bytes(data)
    return path
