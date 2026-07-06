#!/usr/bin/env python3
"""テスト画像で乳頭計測（スケール算出）を検証する。"""

from pathlib import Path

from engine import calibration
from engine.io_utils import load_gray

TEST_DIR = Path("/Users/yy/Desktop/名称未設定フォルダ")
DISC_DIAMETER_UM = 1800.0


def main() -> None:
    files = sorted(TEST_DIR.glob("*.gif")) + sorted(TEST_DIR.glob("*.png"))
    if not files:
        raise SystemExit(f"画像なし: {TEST_DIR}")

    path = files[0]
    gray = load_gray(path)
    h, w = gray.shape
    print(f"画像: {path.name}  サイズ: {w}x{h}")

    # 画像中央付近に水平2点（乳頭端の模擬クリック）
    cy = h * 0.45
    p1 = (w * 0.42, cy)
    p2 = (w * 0.58, cy)
    scale = calibration.compute_scale_um_per_px(p1, p2, DISC_DIAMETER_UM)
    d_px = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    print(f"乳頭2点: {p1} → {p2}")
    print(f"乳頭径: {DISC_DIAMETER_UM} μm, 計測距離: {d_px:.1f} px")
    print(f"スケール: {scale:.4f} μm/px")
    assert scale > 0
    print("OK: 乳頭計測ロジック正常")


if __name__ == "__main__":
    main()
