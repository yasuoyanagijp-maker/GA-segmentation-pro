"""segmentation.py – GAセグメンテーションロジック（FAF / geographic atrophy）。

パイプライン:
    load_gray → CLAHE → Otsu閾値（反転）→ Morphological Open/Close/Dilate
             → remove_small_objects → 中心部病変フィルタ（distance_transform_edt）
"""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

# ── Parameters（サイドバー等から上書き可能） ─────────────────────
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
MIN_GA_PX = 500
MIN_SIZE_RATIO = 0.05
MAX_DIST_FROM_MAIN = 0   # 0 = 中心病変のみ


def segment_ga(
    img: np.ndarray,
    *,
    use_clahe: bool = True,
    clahe_clip: float | None = None,
) -> np.ndarray:
    """CLAHE → Otsu → morphology → 近接フィルタ → 二値マスク。

    use_clahe=False は局所コントラスト補正なし（過大マスクになりやすい）。
    clahe_clip: None なら CLAHE_CLIP。大きいほど局所コントラストが強くなる。
    """
    if use_clahe:
        clip = float(CLAHE_CLIP if clahe_clip is None else clahe_clip)
        clip = max(0.5, clip)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=CLAHE_TILE)
        enhanced = clahe.apply(img)
    else:
        enhanced = img

    _, mask = cv2.threshold(enhanced, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k, iterations=1)

    mask = remove_small_objects(mask.astype(bool),
                                min_size=MIN_GA_PX).astype(np.uint8) * 255
    return _filter_proximity(mask)


def _filter_proximity(mask: np.ndarray) -> np.ndarray:
    labeled = label(mask > 0)
    props = regionprops(labeled)
    if not props:
        return mask

    h, w = mask.shape
    center = np.array([h / 2, w / 2])
    main = min(props,
               key=lambda r: np.linalg.norm(np.array(r.centroid) - center))
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
