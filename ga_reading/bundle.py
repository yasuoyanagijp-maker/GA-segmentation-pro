"""第1の確定を、第2が同じ画像を取れる export に書く。"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from ga_reading.absent import absent_row, write_empty_mask
from ga_reading.merge import write_csv
from ga_reading.metrics import (
    FILE_COL,
    GA_PRIMARY_METRICS,
    LESION_PRESENT,
    PRESENT_QUALITY,
    QUALITY,
)


def reading_root(case_id: str, downloads: Path | None = None) -> Path:
    base = downloads or (Path.home() / "Downloads" / "GA_optos_reading")
    return base / (case_id or "case")


def stamped_dir(parent: Path, prefix: str) -> Path:
    folder = Path(parent) / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_reader_csv(directory: Path, rows: list[dict]) -> Path:
    """第2・最終読影の数値だけを書く。画像とマスクはコピーしない。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = [_measured_record(row) for row in rows]
    fields = [FILE_COL, LESION_PRESENT, QUALITY, "case_id", "date", "filename", *GA_PRIMARY_METRICS]
    path = directory / "grader.csv"
    write_csv(path, fields, records)
    return path


def write_grader_bundle(root: Path, rows: list[dict]) -> Path:
    """export/images と export/masks と grader.csv。第2は masks を使わない。"""
    root = Path(root)
    images = root / "export" / "images"
    masks = root / "export" / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict] = []
    for row in rows:
        source = Path(row["path"])
        filename = row.get("filename") or source.name
        shutil.copy2(source, images / filename)
        shape = tuple(row["shape"])
        mask = row.get("mask")
        if row.get("lesion_present") == "0" or mask is None:
            write_empty_mask(masks / f"{Path(filename).stem}.png", shape)
            csv_rows.append(absent_row(
                filename,
                case_id=row.get("case_id", ""),
                date=row.get("date", ""),
                filename=filename,
            ))
        else:
            _write_mask(masks / f"{Path(filename).stem}.png", np.asarray(mask))
            csv_rows.append(_measured_record(row))
    fields = [FILE_COL, LESION_PRESENT, QUALITY, "case_id", "date", "filename", *GA_PRIMARY_METRICS]
    path = root / "grader.csv"
    write_csv(path, fields, csv_rows)
    return path


def _measured_record(row: dict) -> dict:
    filename = row.get("filename") or Path(row["path"]).name
    record = {column: _blank_or_number(row.get(column)) for column in GA_PRIMARY_METRICS}
    record[FILE_COL] = filename
    record[LESION_PRESENT] = "1"
    record[QUALITY] = PRESENT_QUALITY
    record["case_id"] = row.get("case_id", "")
    record["date"] = row.get("date", "")
    record["filename"] = filename
    return record


def _blank_or_number(value) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.10g}"


def _write_mask(path: Path, mask: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if not cv2.imwrite(str(path), image):
        raise OSError(f"マスクを書けません: {path}")
