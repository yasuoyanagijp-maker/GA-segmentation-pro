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
    SCALE_EXPORT_COLUMNS,
)


def reading_root(case_id: str, downloads: Path | None = None) -> Path:
    base = downloads or (Path.home() / "Downloads" / "GA_optos_reading")
    return base / (case_id or "case")


def stamped_dir(parent: Path, prefix: str) -> Path:
    folder = Path(parent) / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def exclusion_log_text(
    n_images: int,
    n_measured: int,
    exclusions: list[tuple[str, str]],
) -> str:
    """外したファイルと理由。日付の重複や間隔では外さない、と残す。"""
    lines = [
        f"対象 {n_images} 枚。数値を書いた {n_measured} 枚。",
        f"外した {len(exclusions)} 枚。",
        "同じ日付や、短い間隔だけでは外しません。",
    ]
    lines.extend(f"- {name}: {reason}" for name, reason in exclusions)
    return "\n".join(lines) + "\n"


def write_exclusion_log(
    directory: Path,
    n_images: int,
    n_measured: int,
    exclusions: list[tuple[str, str]],
) -> Path:
    path = Path(directory) / "reading_exclusions.log"
    path.write_text(
        exclusion_log_text(n_images, n_measured, exclusions),
        encoding="utf-8",
    )
    return path


def write_reader_csv(
    directory: Path,
    rows: list[dict],
    *,
    n_images: int | None = None,
    exclusions: list[tuple[str, str]] | None = None,
    log_text: str | None = None,
) -> Path:
    """第2・最終読影の数値だけを書く。画像とマスクはコピーしない。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = [_measured_record(row) for row in rows]
    fields = grader_csv_fields()
    path = directory / "grader.csv"
    write_csv(path, fields, records)
    if log_text is not None:
        (directory / "reading_exclusions.log").write_text(log_text, encoding="utf-8")
    elif exclusions is not None:
        write_exclusion_log(directory, n_images if n_images is not None else len(rows), len(records), exclusions)
    return path


def write_grader_bundle(
    root: Path,
    rows: list[dict],
    *,
    source_images: list[Path] | None = None,
    exclusions: list[tuple[str, str]] | None = None,
    log_text: str | None = None,
) -> Path:
    """export/images と export/masks と grader.csv。第2は masks を使わない。

    source_images があれば、確定していない画像も export/images にコピーする。
    数値を書かなかった画像は exclusions と reading_exclusions.log に理由を残す。
    """
    root = Path(root)
    images = root / "export" / "images"
    masks = root / "export" / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict] = []
    copied: set[str] = set()
    for row in rows:
        source = Path(row["path"])
        filename = row.get("filename") or source.name
        shutil.copy2(source, images / filename)
        copied.add(Path(filename).name)
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
    for source in source_images or []:
        source = Path(source)
        if source.name in copied or not source.is_file():
            continue
        shutil.copy2(source, images / source.name)
        copied.add(source.name)
    fields = grader_csv_fields()
    path = root / "grader.csv"
    write_csv(path, fields, csv_rows)
    if log_text is not None:
        (root / "reading_exclusions.log").write_text(log_text, encoding="utf-8")
    elif exclusions is not None:
        write_exclusion_log(root, len(copied), len(csv_rows), exclusions)
    return path


def grader_csv_fields() -> list[str]:
    return [
        FILE_COL,
        LESION_PRESENT,
        QUALITY,
        "case_id",
        "date",
        "filename",
        *GA_PRIMARY_METRICS,
        "needs_review",
        *SCALE_EXPORT_COLUMNS,
    ]


def _measured_record(row: dict) -> dict:
    filename = row.get("filename") or Path(row["path"]).name
    record = {column: _blank_or_number(row.get(column)) for column in GA_PRIMARY_METRICS}
    record[FILE_COL] = filename
    record[LESION_PRESENT] = "1"
    record[QUALITY] = PRESENT_QUALITY
    record["case_id"] = row.get("case_id", "")
    record["date"] = row.get("date", "")
    record["filename"] = filename
    record["needs_review"] = row.get("needs_review", "")
    for column in SCALE_EXPORT_COLUMNS:
        record[column] = _scale_cell(column, row.get(column))
    return record


def _scale_cell(column: str, value) -> str:
    if value is None or value == "":
        return ""
    if column in {"mask_px", "scale_outlier"}:
        return str(int(round(float(value))))
    return f"{float(value):.4f}"


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
