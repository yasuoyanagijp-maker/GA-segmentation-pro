"""眼ごとの固定 µm/px。時点ごとの乳頭–中心窩推定は計測には使わない。

Optos の時点スケールは、1枚目では DFD 4.76 mm、2枚目以降ではその比または
血管位置合わせの倍率から出る。同じ眼・同じ装置では光学倍率はほぼ一定で、
推定のぶれはランドマークと位置合わせの誤差である。面積換算には、眼の
推定値の中央値を使い、中央値から 3% を超える時点は印を付ける。
"""

from __future__ import annotations

import math
import statistics

# 中央値からこれ以上離れた時点推定は、面積換算には使わず印だけ残す。
SCALE_OUTLIER_RATIO = 0.03


def fixed_eye_scale(estimates: list[float]) -> float | None:
    """正の推定値の中央値。1つならその値。無ければ None。"""
    values = [float(value) for value in estimates if value is not None and float(value) > 0]
    if not values:
        return None
    return float(statistics.median(values))


def is_scale_outlier(
    estimate: float | None,
    fixed: float | None,
    *,
    ratio: float = SCALE_OUTLIER_RATIO,
) -> bool:
    """時点推定が、眼の固定スケールから ratio を超えて離れているか。"""
    if estimate is None or fixed is None or estimate <= 0 or fixed <= 0:
        return False
    return abs(float(estimate) - float(fixed)) / float(fixed) > float(ratio)


def area_mm2_from_mask_px(mask_px: int, scale_um_px: float) -> float:
    """マスク画素数と µm/px から mm²。"""
    if mask_px < 0:
        raise ValueError("mask_px は 0 以上が必要です")
    if scale_um_px <= 0:
        raise ValueError("scale_um_px は正の値が必要です")
    return float(mask_px) * (float(scale_um_px) ** 2) / 1e6


def recover_scale_um_px(area_mm2: float, mask_px: int) -> float:
    """報告面積とマスク画素数から、当時使った µm/px を逆算する。"""
    if mask_px <= 0:
        raise ValueError("mask_px は正の値が必要です")
    if area_mm2 < 0:
        raise ValueError("area_mm2 は 0 以上が必要です")
    return 1000.0 * math.sqrt(float(area_mm2) / float(mask_px))


def area_correction_factor(scale_original: float, scale_fixed: float) -> float:
    """面積に掛ける (s_fixed / s_original)²。"""
    if scale_original <= 0 or scale_fixed <= 0:
        raise ValueError("スケールは正の値が必要です")
    return (float(scale_fixed) / float(scale_original)) ** 2


def apply_area_correction(area_mm2: float, factor: float) -> float:
    return float(area_mm2) * float(factor)


def area_mm_sqrt(area_mm2: float) -> float:
    if area_mm2 <= 0:
        return 0.0
    return math.sqrt(float(area_mm2))
