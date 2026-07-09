"""region_growing.py – リージョングローイング（MNV=暗部 / GA=白部・過分割）。"""

from __future__ import annotations

import cv2
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _local_stats(gray: np.ndarray, x: int, y: int, window_size: int) -> tuple[float, float]:
    h, w = gray.shape[:2]
    half_w = window_size // 2
    y_min, y_max = max(0, y - half_w), min(h, y + half_w + 1)
    x_min, x_max = max(0, x - half_w), min(w, x + half_w + 1)
    roi = gray[y_min:y_max, x_min:x_max]
    mu = float(np.mean(roi))
    sigma = float(np.maximum(np.std(roi), 0.5))
    return mu, sigma


def _flood_from_potential(potential_mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    x, y = seed
    h, w = potential_mask.shape[:2]
    if potential_mask[y, x] == 0:
        return np.zeros((h, w), dtype=np.uint8)

    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood_mask[1:-1, 1:-1] = np.where(potential_mask == 0, 1, 0).astype(np.uint8)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
    image_dummy = np.zeros((h, w), dtype=np.uint8)
    cv2.floodFill(image_dummy, flood_mask, seed, 0, 0, 0, flags)
    return np.where(flood_mask[1:-1, 1:-1] == 255, 255, 0).astype(np.uint8)


def fast_region_growing(
    img: np.ndarray,
    seed: tuple[int, int],
    a: float,
    window_size: int = 5,
) -> np.ndarray:
    """MNV 向け: シードと同程度の**暗部**へ μ±aσ で連結拡大（双方向）。"""
    gray = _to_gray(img)
    x, y = seed
    h, w = gray.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return np.zeros((h, w), dtype=np.uint8)

    mu, sigma = _local_stats(gray, x, y, window_size)
    low = float(np.clip(mu - a * sigma, 0, 255))
    high = float(np.clip(mu + a * sigma, 0, 255))
    potential_mask = cv2.inRange(gray, low, high)
    return _flood_from_potential(potential_mask, seed)


def fast_region_growing_bright(
    img: np.ndarray,
    seed: tuple[int, int],
    a: float,
    *,
    window_size: int = 5,
    dark_margin: float = 0.25,
) -> np.ndarray:
    """明るい画素へ拡大（シードが画像中央値より暗い場合は空）。"""
    gray = _to_gray(img)
    x, y = seed
    h, w = gray.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return np.zeros((h, w), dtype=np.uint8)

    seed_val = float(gray[y, x])
    if seed_val < float(np.median(gray)):
        return np.zeros((h, w), dtype=np.uint8)

    mu, sigma = _local_stats(gray, x, y, window_size)
    low = float(np.clip(max(mu - dark_margin * a * sigma, seed_val - dark_margin * a * sigma), 0, 255))
    high = float(np.clip(mu + a * sigma, 0, 255))
    potential_mask = cv2.inRange(gray, low, high)

    grown = _flood_from_potential(potential_mask, seed)
    if not np.any(grown):
        return grown

    bright_floor = float(np.clip(seed_val - dark_margin * a * sigma, 0, 255))
    grown[gray < bright_floor] = 0
    return grown


def erase_oversegmentation_from_mask(
    img_gray: np.ndarray,
    mask: np.ndarray,
    seed: tuple[int, int],
    a: float,
    *,
    window_size: int = 11,
) -> tuple[np.ndarray, int]:
    """FAF 向け: マスク内の明るい過分割を除去（元画像グレーのみ使用）。

    緑の半透明表示は参照しない。暗い GA 本体上のクリックでも、
    連結するより明るいマスク画素を長押しで拡大削除する。
    戻り値: (新マスク, 削除画素数)
    """
    gray = _to_gray(img_gray)
    x, y = int(seed[0]), int(seed[1])
    h, w = gray.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return mask, 0

    on_mask = mask[y, x] > 0
    seed_val = float(gray[y, x])
    mu, sigma = _local_stats(gray, x, y, window_size)
    peel_floor = float(np.clip(seed_val + 0.15 * a * sigma, seed_val + 1, 255))
    high = float(np.clip(seed_val + a * sigma * 1.5, peel_floor + 1, 255))

    potential = cv2.inRange(gray, int(seed_val), int(high))
    if on_mask:
        # マスク上クリック: 拡大は GA 領域内に限定
        potential[mask == 0] = 0
        if potential[y, x] == 0:
            return mask, 0
    elif not np.any((potential > 0) & (mask > 0)):
        # 緑外クリック: 強度連結でマスクに届かない場合は何もしない
        return mask, 0

    grown = _flood_from_potential(potential, (x, y))
    if not np.any(grown):
        return mask, 0

    mask_in_grown = (grown > 0) & (mask > 0)
    if not np.any(mask_in_grown):
        return mask, 0

    # 暗い GA 本体: より明るい縁だけ削除 / 明るい過分割: 連結領域を削除 / 均一領域: 何もしない
    if seed_val < mu - 0.25 * sigma:
        remove = mask_in_grown & (gray >= peel_floor)
    elif seed_val > mu + 0.25 * sigma:
        remove = mask_in_grown
    else:
        return mask, 0
    removed = int(np.sum(remove))
    if removed == 0:
        return mask, 0

    out = mask.copy()
    out[remove] = 0
    return out, removed


def expand_segmentation_from_mask(
    img_gray: np.ndarray,
    mask: np.ndarray,
    seed: tuple[int, int],
    a: float,
    *,
    window_size: int = 11,
) -> tuple[np.ndarray, int]:
    """FAF 向け: 暗い GA 領域をマスクに追加（erase の逆操作）。

    緑の内外どちらでも、暗部を長押しで連結拡大しマスクに加える。
    戻り値: (新マスク, 追加画素数)
    """
    gray = _to_gray(img_gray)
    x, y = int(seed[0]), int(seed[1])
    h, w = gray.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return mask, 0

    seed_val = float(gray[y, x])
    mu, sigma = _local_stats(gray, x, y, window_size)

    if seed_val > mu + 0.25 * sigma:
        return mask, 0

    grown = fast_region_growing(img_gray, (x, y), a, window_size=window_size)
    if not np.any(grown):
        return mask, 0

    add = (grown > 0) & (mask == 0)
    added = int(np.sum(add))
    if added == 0:
        return mask, 0

    out = mask.copy()
    out[add] = 255
    return out, added


def clear_mask_in_freehand_region(
    mask: np.ndarray,
    points: list[tuple[float, float]],
    *,
    outside: bool = False,
) -> tuple[np.ndarray, int]:
    """フリーハンドで囲んだ領域のマスクを消去する。

    outside=False: 範囲内を消去（従来どおり）
    outside=True: 範囲外を消去（囲んだ内側だけ残す）
    """
    h, w = mask.shape[:2]
    if len(points) < 3:
        return mask, 0
    pts = np.array([[int(x), int(y)] for x, y in points], dtype=np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    region = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(region, [pts], 255)
    if outside:
        remove = (mask > 0) & (region == 0)
    else:
        remove = (mask > 0) & (region > 0)
    removed = int(np.sum(remove))
    if removed == 0:
        return mask, 0
    out = mask.copy()
    out[remove] = 0
    return out, removed


# 後方互換
def erase_bright_from_mask(
    img_gray: np.ndarray,
    mask: np.ndarray,
    seed: tuple[int, int],
    a: float,
) -> np.ndarray:
    out, _ = erase_oversegmentation_from_mask(img_gray, mask, seed, a)
    return out
