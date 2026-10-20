"""病変なしは計測しない。数値は空欄、0 は書かない。空のマスクは残す。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ga_reading.metrics import ABSENT_QUALITY, FILE_COL, GA_PRIMARY_METRICS, LESION_PRESENT, QUALITY


def absent_row(filename: str, **meta: str) -> dict[str, str]:
    row = {column: "" for column in GA_PRIMARY_METRICS}
    row[FILE_COL] = filename
    row[LESION_PRESENT] = "0"
    row[QUALITY] = ABSENT_QUALITY
    for key, value in meta.items():
        row[key] = "" if value is None else str(value)
    return row


def write_empty_mask(path: Path, shape: tuple[int, ...]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    _write_png(path, mask)
    return path


def _write_png(path: Path, image: np.ndarray) -> None:
    import cv2

    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"マスクを書けません: {path}")
