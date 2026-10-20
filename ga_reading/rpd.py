"""相対差（RPD）と、20% 以内なら平均・超えまたは空欄なら NA。"""

from __future__ import annotations

import math
from typing import Any

from ga_reading.metrics import RPD_THRESHOLD_PCT

EPS = 1e-12


def to_float(value: Any) -> float | None:
    """空欄と NA は欠損。0 は 0 のまま残す。"""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.upper() in {"NA", "NAN", "NULL", "N/A"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def rpd_pct(a: float, b: float) -> float | None:
    """|A−B| / ((|A|+|B|)/2) × 100。両方 0 は 0。分母が 0 で差があるときは None。"""
    denom = (abs(a) + abs(b)) / 2.0
    if denom < EPS:
        return 0.0 if abs(a - b) < EPS else None
    return abs(a - b) / denom * 100.0


def adopt_pair(a: float | None, b: float | None, threshold: float = RPD_THRESHOLD_PCT) -> tuple[str, str, str]:
    """採用値、状態（OK / RECHECK / MISSING）、RPD の文字列。"""
    if a is None or b is None:
        return "NA", "MISSING", "NA"
    if abs(a) < EPS and abs(b) < EPS:
        return "0", "OK", "0.0000"
    score = rpd_pct(a, b)
    if score is None or score > threshold:
        return "NA", "RECHECK", "NA" if score is None else f"{score:.4f}"
    return f"{(a + b) / 2.0:.10g}", "OK", f"{score:.4f}"
