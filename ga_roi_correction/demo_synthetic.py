"""
demo_synthetic.py
------------------
実データなしでパイプライン全体を検証するための合成データテスト。

シナリオ(添付SOPの3異常パターンを再現):
  frame 0 (初回) : 過大セグメンテーション(半径+40px)
  frame 2 (中間) : 縮小異常(半径-18px)
  frame 4 (最終) : 過大セグメンテーション(半径+40px)
  各フレームは真の萎縮円がわずかに成長し、撮影位置もわずかにずれる
  (画像登録の妥当性も同時に検証する)。
"""

from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path

from ga_roi_correction import correct_ga_series, PipelineConfig
from ga_roi_correction.qc import dice


def make_synthetic_frame(size: int, center: tuple[int, int], radius: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)

    background = 180 + 20 * np.sin(xx / 40) + 15 * np.cos(yy / 55)
    lesion_intensity = 60.0
    smooth_edge = 1 / (1 + np.exp((dist - radius) / 3.0))  # 0=外側,1=内側 のソフトエッジ
    img = background * (1 - smooth_edge) + lesion_intensity * smooth_edge
    img += rng.normal(0, 6, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_circle_mask(size: int, center: tuple[int, int], radius: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    return (dist <= radius).astype(np.uint8)


def run_demo():
    size = 300
    n_frames = 5
    base_center = (150, 150)
    true_radii = [30 + 5 * i for i in range(n_frames)]     # 徐々に拡大(真値)
    jitters = [(0, 0), (3, -2), (-4, 3), (2, 4), (-3, -3)]  # 撮影ごとの位置ずれ

    images, init_masks, true_masks = [], [], []
    for i in range(n_frames):
        center = (base_center[0] + jitters[i][0], base_center[1] + jitters[i][1])
        img = make_synthetic_frame(size, center, true_radii[i], seed=i)
        true_mask = make_circle_mask(size, center, true_radii[i])
        images.append(img)
        true_masks.append(true_mask)

        # わざと異常な初期マスクを作る
        if i == 0:
            init_masks.append(make_circle_mask(size, center, true_radii[i] + 40))   # 初回:過大
        elif i == 2:
            init_masks.append(make_circle_mask(size, center, true_radii[i] - 18))   # 中間:縮小異常
        elif i == n_frames - 1:
            init_masks.append(make_circle_mask(size, center, true_radii[i] + 40))   # 最終:過大
        else:
            init_masks.append(true_mask.copy())

    dates = [f"2026-{m:02d}-01" for m in range(1, n_frames + 1)]

    config = PipelineConfig(reference_index=0, refine_method="chan_vese")
    out_dir = str(Path(__file__).resolve().parent / "demo_output")
    result = correct_ga_series(images, init_masks, dates=dates, config=config, output_dir=out_dir)

    print("=== QC ログ ===")
    print(result["qc_dataframe"].to_string(index=False))

    print("\n=== 真値マスクとのDice比較(補正前 vs 補正後) ===")
    for i in range(n_frames):
        d_before = dice(init_masks[i], true_masks[i])
        d_after = dice(result["corrected_masks_original"][i], true_masks[i])
        flag = "  <-- 意図的異常フレーム" if i in (0, 2, n_frames - 1) else ""
        print(f"frame {i}: raw_dice={d_before:.3f}  corrected_dice={d_after:.3f}{flag}")

    Path(out_dir).mkdir(exist_ok=True, parents=True)
    for i in range(n_frames):
        cv2.imwrite(f"{out_dir}/frame_{i}_init_mask.png", init_masks[i] * 255)
        cv2.imwrite(f"{out_dir}/frame_{i}_true_mask.png", true_masks[i] * 255)
        cv2.imwrite(f"{out_dir}/frame_{i}_corrected_mask.png", result["corrected_masks_original"][i] * 255)
        cv2.imwrite(f"{out_dir}/frame_{i}_raw_image.png", images[i])

    print(f"\nQC CSV・オーバーレイ画像は {out_dir} に保存されました。")


if __name__ == "__main__":
    run_demo()
