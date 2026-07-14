"""
qc.py
-----
SOP 8章(異常判定)・10章(QCと承認)・11章(記録)に対応。
面積の時系列異常検出に加え、Dice(前フレームとの形状整合性)や
登録信頼度(cc)も含めた多面的なQC判定を行う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


def mask_area(mask: np.ndarray) -> int:
    return int(mask.astype(bool).sum())


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a, b = mask_a.astype(bool), mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0


@dataclass
class AnomalyThresholds:
    rel_drop: float = 0.05        # 前フレーム比でこれ以上縮小したら中間フレーム異常候補
    rel_jump: float = 0.20        # 次フレーム比でこれ以上急増したら異常候補
    first_ratio: float = 1.5      # 初回フレームが2枚目よりこの倍率以上大きければ異常候補
    last_ratio: float = 1.5       # 最終フレームが直前よりこの倍率以上大きければ異常候補
    min_dice_with_prev: float = 0.5   # 登録後Diceがこれ未満なら形状不整合フラグ
    min_registration_cc: float = 0.5  # 登録信頼度がこれ未満ならレビュー対象


@dataclass
class FrameQCRecord:
    index: int
    date: Optional[str]
    raw_area: int
    corrected_area: int
    dice_with_prev_raw: Optional[float]
    dice_with_prev_corrected: Optional[float]
    registration_mode: Optional[str]
    registration_cc: Optional[float]
    area_anomaly: bool
    shape_anomaly: bool
    registration_low_confidence: bool
    needs_review: bool
    reason: str


def detect_area_anomaly(prev_area: Optional[int], curr_area: int, next_area: Optional[int],
                         th: AnomalyThresholds) -> bool:
    """SOP 8章のロジック(初回/中間/最終で判定基準を分ける)。"""
    if prev_area is None and next_area is not None:
        return curr_area > next_area * th.first_ratio
    if next_area is None and prev_area is not None:
        return curr_area > prev_area * th.last_ratio
    if prev_area is not None and curr_area < prev_area * (1 - th.rel_drop):
        return True
    if next_area is not None and curr_area > next_area * (1 + th.rel_jump):
        return True
    return False


def build_qc_records(dates, raw_areas, corrected_areas, dice_prev_raw, dice_prev_corr,
                      reg_modes, reg_ccs, th: AnomalyThresholds) -> list[FrameQCRecord]:
    n = len(raw_areas)
    records = []
    for i in range(n):
        prev_a = corrected_areas[i - 1] if i > 0 else None
        next_a = raw_areas[i + 1] if i < n - 1 else None
        area_anom = detect_area_anomaly(prev_a, raw_areas[i], next_a, th)

        shape_anom = False
        if dice_prev_corr[i] is not None and dice_prev_corr[i] < th.min_dice_with_prev:
            shape_anom = True

        reg_low_conf = False
        if reg_ccs[i] is not None and reg_ccs[i] < th.min_registration_cc:
            reg_low_conf = True

        reasons = []
        if area_anom:
            reasons.append("area_anomaly")
        if shape_anom:
            reasons.append("shape_anomaly(low_dice)")
        if reg_low_conf:
            reasons.append("low_registration_confidence")

        records.append(FrameQCRecord(
            index=i,
            date=dates[i] if dates else None,
            raw_area=raw_areas[i],
            corrected_area=corrected_areas[i],
            dice_with_prev_raw=dice_prev_raw[i],
            dice_with_prev_corrected=dice_prev_corr[i],
            registration_mode=reg_modes[i],
            registration_cc=reg_ccs[i],
            area_anomaly=area_anom,
            shape_anomaly=shape_anom,
            registration_low_confidence=reg_low_conf,
            needs_review=bool(reasons),
            reason=";".join(reasons) if reasons else "ok",
        ))
    return records


def records_to_dataframe(records: list[FrameQCRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in records])


def save_qc_log(records: list[FrameQCRecord], out_csv_path: str) -> pd.DataFrame:
    df = records_to_dataframe(records)
    df.to_csv(out_csv_path, index=False)
    return df
