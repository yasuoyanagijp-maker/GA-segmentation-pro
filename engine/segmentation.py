"""segmentation.py – GAセグメンテーションロジック（FAF / geographic atrophy）。

パイプライン:
    load_gray → CLAHE → 黄斑ROI内 Otsu（反転）→ Morphological Open/Close
             → remove_small_objects → 中心窩基準の近接フィルタ
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

# ── Parameters（サイドバー等から上書き可能） ─────────────────────
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
MIN_GA_PX = 500
MIN_SPECK_PX = 50          # Erase/Draw 後の孤立微小粒子除去
MIN_SIZE_RATIO = 0.05
MAX_DIST_FROM_MAIN = 0     # 0 = 主病変に接続した成分のみ
MACULA_OTSU_RADIUS_MM = 6.0


def _macula_roi_mask(
    shape: tuple[int, int],
    fovea_xy: tuple[float, float],
    scale_um_px: float,
    radius_mm: float = MACULA_OTSU_RADIUS_MM,
) -> np.ndarray:
    """中心窩を中心とした円形 ROI（Otsu 推定用）。"""
    h, w = shape
    fx, fy = fovea_xy
    radius_px = radius_mm * 1000.0 / scale_um_px
    yy, xx = np.ogrid[:h, :w]
    return ((xx - fx) ** 2 + (yy - fy) ** 2) <= radius_px ** 2


def _threshold_dark_ga(
    enhanced: np.ndarray,
    *,
    fovea_xy: tuple[float, float] | None = None,
    scale_um_px: float | None = None,
    macula_radius_mm: float = MACULA_OTSU_RADIUS_MM,
) -> np.ndarray:
    """低輝光（GA）を 255 にする二値マスク。

    fovea + scale があれば黄斑 ROI 内の画素だけで Otsu 閾値を推定し、
    全画像に適用（外周の明るさに閾値が引っ張られるのを防ぐ）。
    """
    roi = None
    if (
        fovea_xy is not None
        and scale_um_px is not None
        and scale_um_px > 0
    ):
        roi = _macula_roi_mask(enhanced.shape, fovea_xy, scale_um_px, macula_radius_mm)

    if roi is not None and int(np.count_nonzero(roi)) >= 64:
        vals = enhanced[roi].astype(np.uint8).reshape(-1, 1)
        thresh, _ = cv2.threshold(
            vals, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        mask = np.where(enhanced <= thresh, 255, 0).astype(np.uint8)
    else:
        _, mask = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
    return mask


def remove_small_regions(
    mask: np.ndarray,
    *,
    min_size: int | None = None,
) -> np.ndarray:
    """連結成分のうち min_size 未満の領域を除去（0/255 uint8 マスク）。"""
    ms = int(MIN_GA_PX if min_size is None else min_size)
    if ms <= 0:
        return mask
    binary = mask > 0
    if not binary.any():
        return mask
    cleaned = remove_small_objects(binary, min_size=ms)
    return cleaned.astype(np.uint8) * 255


def cleanup_mask_after_edit(mask: np.ndarray) -> np.ndarray:
    """Erase / Draw / 範囲消去後に残る孤立微小粒子を除去。"""
    return remove_small_regions(mask, min_size=MIN_SPECK_PX)


def segment_ga(
    img: np.ndarray,
    *,
    use_clahe: bool = True,
    clahe_clip: float | None = None,
    fovea_xy: tuple[float, float] | None = None,
    scale_um_px: float | None = None,
    macula_radius_mm: float = MACULA_OTSU_RADIUS_MM,
    min_size: int | None = None,
) -> np.ndarray:
    """CLAHE → 黄斑ROI Otsu → morphology → 小領域除去 → 近接フィルタ → 二値マスク。

    use_clahe=False は局所コントラスト補正なし（過大マスクになりやすい）。
    fovea_xy + scale_um_px があれば Otsu 推定と近接フィルタの基準点に中心窩を使う。
    min_size は小領域除去の下限画素（None なら MIN_GA_PX）。
    """
    if use_clahe:
        clip = float(CLAHE_CLIP if clahe_clip is None else clahe_clip)
        clip = max(0.5, clip)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=CLAHE_TILE)
        enhanced = clahe.apply(img)
    else:
        enhanced = img

    mask = _threshold_dark_ga(
        enhanced,
        fovea_xy=fovea_xy,
        scale_um_px=scale_um_px,
        macula_radius_mm=macula_radius_mm,
    )

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    mask = remove_small_regions(mask, min_size=min_size)
    return _filter_proximity(mask, fovea_xy=fovea_xy)


def segment_ga_with_optional_cv(
    img: np.ndarray,
    *,
    auto_cv: bool = True,
    use_clahe: bool = True,
    clahe_clip: float | None = None,
    fovea_xy: tuple[float, float] | None = None,
    scale_um_px: float | None = None,
    macula_radius_mm: float = MACULA_OTSU_RADIUS_MM,
    min_size: int | None = None,
) -> np.ndarray:
    """Otsu 自動セグメンテーション → 任意で制約付き Chan-Vese 精緻化。

    auto_cv=True のとき ``refine_mask_chan_vese_light``（外周クリップ + 穴埋め）を適用。
    空マスクでは CV をスキップ。プレビュー / 確定で同じ経路・同じ iter を使う。
    """
    mask = segment_ga(
        img,
        use_clahe=use_clahe,
        clahe_clip=clahe_clip,
        fovea_xy=fovea_xy,
        scale_um_px=scale_um_px,
        macula_radius_mm=macula_radius_mm,
        min_size=min_size,
    )
    if not auto_cv or mask is None or not np.any(mask > 0):
        return mask
    from engine.region_growing import refine_mask_chan_vese_light

    return refine_mask_chan_vese_light(img, mask)


def _filter_proximity(
    mask: np.ndarray,
    *,
    fovea_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """主病変（中心窩に最も近い連結成分）に接続した成分のみ残す。"""
    labeled = label(mask > 0)
    props = regionprops(labeled)
    if not props:
        return mask

    h, w = mask.shape
    if fovea_xy is not None:
        anchor = np.array([fovea_xy[1], fovea_xy[0]], dtype=np.float64)
    else:
        anchor = np.array([h / 2, w / 2], dtype=np.float64)
    main = min(
        props,
        key=lambda r: np.linalg.norm(np.array(r.centroid) - anchor),
    )
    max_area = main.area

    main_bin = (labeled == main.label).astype(np.uint8)
    dist_map = distance_transform_edt(1 - main_bin)

    out = np.zeros_like(mask)
    for r in props:
        if r.area < max_area * MIN_SIZE_RATIO:
            continue
        pxs = np.argwhere(labeled == r.label)
        if dist_map[pxs[:, 0], pxs[:, 1]].min() > MAX_DIST_FROM_MAIN:
            continue
        out[labeled == r.label] = 255
    return out
