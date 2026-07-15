"""
demo_position_shift.py
-----------------------
「面積は正常なのに位置が大きくずれたROI」を、前後フレーム誘導 +
単調性制約(position_guided_monotonic_refine)で補正できることを
実データなしで検証する合成デモ。

シナリオ:
  frame 2 (中間) : 半径は真値どおりだが中心を大きくずらしたROI
                   -> 面積ベースの異常検知(area_anomaly)では検出できない
  他フレーム     : 真値ROIをそのまま初期マスクとして与える

位置ずれ補正 OFF/ON での真値Diceを比較出力する。
"""

from __future__ import annotations

import numpy as np

from ga_roi_correction import correct_ga_series, PipelineConfig
from ga_roi_correction.qc import dice
from demo_synthetic import make_synthetic_frame, make_circle_mask


def run_demo():
    size = 300
    n_frames = 5
    center = (150, 150)
    true_radii = [30 + 5 * i for i in range(n_frames)]  # 徐々に拡大(真値)
    shifted_index = 2
    shift_px = (60, 25)  # frame2 のROIだけ大きく位置ずれさせる

    images, init_masks, true_masks = [], [], []
    for i in range(n_frames):
        images.append(make_synthetic_frame(size, center, true_radii[i], seed=i))
        true_masks.append(make_circle_mask(size, center, true_radii[i]))
        if i == shifted_index:
            bad_center = (center[0] + shift_px[0], center[1] + shift_px[1])
            init_masks.append(make_circle_mask(size, bad_center, true_radii[i]))
        else:
            init_masks.append(true_masks[i].copy())

    dates = [f"2026-{m:02d}-01" for m in range(1, n_frames + 1)]
    print(f"raw dice(frame{shifted_index}, ずれたROI vs 真値) = "
          f"{dice(init_masks[shifted_index], true_masks[shifted_index]):.3f}\n")

    results = {}
    for label, enabled in (("OFF", False), ("ON ", True)):
        config = PipelineConfig(
            reference_index=0,
            refine_method="band_adaptive_threshold",
            enable_position_guided_refine=enabled,
        )
        out = correct_ga_series(images, init_masks, dates=dates, config=config)
        dices = [round(dice(out["corrected_masks_original"][i], true_masks[i]), 3)
                 for i in range(n_frames)]
        results[label] = out
        print(f"位置ずれ補正 {label}: dice(補正後 vs 真値) = {dices}")

    print("\n=== QC ログ(位置ずれ補正 ON) ===")
    df = results["ON "]["qc_dataframe"]
    cols = ["index", "raw_area", "corrected_area",
            "dice_with_neighbor_prior", "position_anomaly", "area_anomaly", "reason"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    run_demo()
