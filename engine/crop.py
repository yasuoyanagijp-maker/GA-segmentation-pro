"""crop.py – 画像クロップユーティリティ。"""

import numpy as np


def normalize_box(x0: float, y0: float, x1: float, y1: float,
                  width: int, height: int) -> tuple[int, int, int, int]:
    """2点クリックから有効なクロップ矩形 (x0, y0, x1, y1) を返す。

    x1, y1 は排他的上限（スライス用）として扱う。
    """
    xa, xb = sorted((int(round(x0)), int(round(x1))))
    ya, yb = sorted((int(round(y0)), int(round(y1))))

    xa = max(0, min(width - 1, xa))
    xb = max(0, min(width, xb))
    ya = max(0, min(height - 1, ya))
    yb = max(0, min(height, yb))

    if xb <= xa + 1 or yb <= ya + 1:
        raise ValueError('クロップ範囲が小さすぎます')

    return xa, ya, xb, yb


def apply_crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """矩形 box=(x0,y0,x1,y1) で画像を切り出す。"""
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1].copy()
