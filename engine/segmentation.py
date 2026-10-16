"""segmentation.py – GAセグメンテーションロジック（FAF / geographic atrophy）。

パイプライン:
    load_gray → CLAHE → 黄斑ROI内 Otsu（反転）→ Morphological Open/Close
             → remove_small_objects → 中心窩基準の近接フィルタ

    または Seeded: 中心窩近傍の暗部コア（多巣）を検出し、拡大は手動成長に任せる
             → remove_small → 任意で制約付き CV
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

# Seeded auto-seg: 多巣の小円形コアを拾い、拡大は手動に任せる（自動は小さく）
SEG_AUTO_MODE_OTSU = "otsu"
SEG_AUTO_MODE_SEEDED = "seeded"
SEED_DARK_PERCENTILE = 18.0       # 暗側パーセンタイル（高すぎると灰縁を拾う）
SEED_MEDIAN_OFFSET = 25.0         # thr ≤ med - offset（40 は mottled FAF でコア欠落）
SEED_MEDIAN_FRAC = 0.65           # thr ≤ med * frac（0.5 は暗め画像で厳しすぎ）
SEED_MIN_COMPONENT_PX = 12
SEED_MAX_COMPONENTS = 24
SEEDED_CORE_DILATE_PX = 1         # コアにごく薄い縁だけ（成長はしない）
# UI の seg_min_size(既定500) は Otsu 用。Seeded はシード検出フロアのみ再適用
SEEDED_MIN_SIZE_PX = SEED_MIN_COMPONENT_PX
SEEDED_MIN_SIZE_CAP_PX = SEED_MIN_COMPONENT_PX


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


def _enhance(img: np.ndarray, *, use_clahe: bool, clahe_clip: float | None) -> np.ndarray:
    if use_clahe:
        clip = float(CLAHE_CLIP if clahe_clip is None else clahe_clip)
        clip = max(0.5, clip)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=CLAHE_TILE)
        return clahe.apply(img)
    return img


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
    enhanced = _enhance(img, use_clahe=use_clahe, clahe_clip=clahe_clip)

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


def _dark_seed_mask_and_points(
    enhanced: np.ndarray,
    fovea_xy: tuple[float, float],
    scale_um_px: float,
    macula_radius_mm: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """中心窩近傍 ROI 内の暗いクラスタからシードマスクと代表点を作る。"""
    roi = _macula_roi_mask(
        enhanced.shape, fovea_xy, scale_um_px, macula_radius_mm,
    )
    if int(np.count_nonzero(roi)) < 64:
        return np.zeros(enhanced.shape, dtype=np.uint8), []

    vals = enhanced[roi]
    med = float(np.median(vals))
    below = vals[vals < med]
    if below.size < 32:
        return np.zeros(enhanced.shape, dtype=np.uint8), []
    # 中央値より暗い側のコア（灰縁はパーセンタイル＋med キャップで抑える）
    thr = float(np.percentile(below, SEED_DARK_PERCENTILE))
    thr = min(thr, med - float(SEED_MEDIAN_OFFSET), med * float(SEED_MEDIAN_FRAC))
    dark = (enhanced <= thr) & roi
    if not np.any(dark):
        return np.zeros(enhanced.shape, dtype=np.uint8), []

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_u8 = np.where(dark, 255, 0).astype(np.uint8)
    dark_u8 = cv2.morphologyEx(dark_u8, cv2.MORPH_OPEN, k, iterations=1)

    labeled = label(dark_u8 > 0)
    props = regionprops(labeled)
    if not props:
        return np.zeros(enhanced.shape, dtype=np.uint8), []

    fx, fy = float(fovea_xy[0]), float(fovea_xy[1])
    ranked = sorted(
        props,
        key=lambda r: (
            (r.centroid[1] - fx) ** 2 + (r.centroid[0] - fy) ** 2,
            -r.area,
        ),
    )
    seed_mask = np.zeros(enhanced.shape, dtype=np.uint8)
    seeds: list[tuple[int, int]] = []
    for r in ranked:
        if r.area < SEED_MIN_COMPONENT_PX:
            continue
        seed_mask[labeled == r.label] = 255
        cy, cx = r.centroid  # row, col
        seeds.append((int(round(cx)), int(round(cy))))
        if len(seeds) >= SEED_MAX_COMPONENTS:
            break

    return seed_mask, seeds


def segment_ga_seeded(
    img: np.ndarray,
    *,
    use_clahe: bool = True,
    clahe_clip: float | None = None,
    fovea_xy: tuple[float, float] | None = None,
    scale_um_px: float | None = None,
    macula_radius_mm: float = MACULA_OTSU_RADIUS_MM,
    min_size: int | None = None,
    grow_a: float | None = None,
    use_entropy_stop: bool = True,
) -> tuple[np.ndarray, dict]:
    """中心窩近傍の暗部コア（多巣）をマスク化。自動では大きく伸ばさない。

    成長（region growing）は手動の標準安全成長に任せ、ここではコア＋ごく薄い縁のみ。
    ``grow_a`` / ``use_entropy_stop`` は互換のため残すが未使用。
    シード不能時は Otsu ``segment_ga`` にフォールバックする。
    """
    del grow_a, use_entropy_stop
    info: dict = {
        "mode_used": SEG_AUTO_MODE_SEEDED,
        "fallback": False,
        "n_seeds": 0,
        "message": "",
    }
    if (
        fovea_xy is None
        or scale_um_px is None
        or scale_um_px <= 0
    ):
        mask = segment_ga(
            img,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            fovea_xy=fovea_xy,
            scale_um_px=scale_um_px,
            macula_radius_mm=macula_radius_mm,
            min_size=min_size,
        )
        info.update(
            mode_used=SEG_AUTO_MODE_OTSU,
            fallback=True,
            message="Seeded不可（中心窩/スケール不足）→ Otsu にフォールバック",
        )
        return mask, info

    enhanced = _enhance(img, use_clahe=use_clahe, clahe_clip=clahe_clip)
    seed_mask, seeds = _dark_seed_mask_and_points(
        enhanced, fovea_xy, float(scale_um_px), macula_radius_mm,
    )
    info["n_seeds"] = len(seeds)

    if not seeds or not np.any(seed_mask > 0):
        mask = segment_ga(
            img,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            fovea_xy=fovea_xy,
            scale_um_px=scale_um_px,
            macula_radius_mm=macula_radius_mm,
            min_size=min_size,
        )
        info.update(
            mode_used=SEG_AUTO_MODE_OTSU,
            fallback=True,
            message="暗部シードなし → Otsu にフォールバック",
        )
        return mask, info

    # コアのみ（LSF成長なし）。1px 程度の close で穴・ギザを軽く整える。
    mask = seed_mask.copy()
    k = max(1, int(SEEDED_CORE_DILATE_PX) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ROI 外への漏れを抑える（黄斑内の多巣はすべて残す）
    roi = _macula_roi_mask(
        enhanced.shape, fovea_xy, float(scale_um_px), macula_radius_mm,
    )
    mask = np.where(roi, mask, 0).astype(np.uint8)

    # Otsu 用の大きな min_size（UI 既定 500）で多巣コアを消さない
    req = int(SEEDED_MIN_SIZE_PX if min_size is None else min_size)
    ms = max(SEED_MIN_COMPONENT_PX, min(req, SEEDED_MIN_SIZE_CAP_PX))
    cleaned = remove_small_regions(mask, min_size=ms)
    if np.any(cleaned > 0):
        mask = cleaned
    elif np.any(seed_mask > 0):
        # 後処理で空になっても暗コアがあればコアを返す（Otsu 巨大マスクより優先）
        mask = np.where(roi, seed_mask, 0).astype(np.uint8)
        info["message"] = (
            f"Seededコア（後処理縮小回避・シード {info['n_seeds']}）"
        )
        n_comp = int(label(mask > 0).max()) if np.any(mask > 0) else 0
        info["n_components"] = n_comp
        return mask, info
    else:
        mask = segment_ga(
            img,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            fovea_xy=fovea_xy,
            scale_um_px=scale_um_px,
            macula_radius_mm=macula_radius_mm,
            min_size=min_size,
        )
        info.update(
            mode_used=SEG_AUTO_MODE_OTSU,
            fallback=True,
            message="Seeded後処理で空 → Otsu にフォールバック",
        )
        return mask, info

    n_comp = int(label(mask > 0).max()) if np.any(mask > 0) else 0
    info["n_components"] = n_comp
    info["message"] = f"Seededコア（シード {info['n_seeds']}・成分 {n_comp}）"
    return mask, info


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
    mode: str = SEG_AUTO_MODE_OTSU,
    info: dict | None = None,
) -> np.ndarray:
    """自動セグメンテーション → 任意で制約付き Chan-Vese 精緻化。

    mode:
        ``otsu`` — 現行黄斑ROI Otsu 経路
        ``seeded`` — 中心窩近傍の暗部コア検出（多巣・自動では伸ばさない；失敗時 Otsu）

    auto_cv=True のとき ``refine_mask_chan_vese_light``（外周クリップ + 穴埋め）を適用。
    空マスクでは CV をスキップ。プレビュー / 確定で同じ経路・同じ iter を使う。
    info に dict を渡すと mode_used / fallback / message / n_seeds を書き込む。
    """
    meta: dict = {
        "mode_used": mode,
        "fallback": False,
        "n_seeds": 0,
        "message": "",
    }
    if mode == SEG_AUTO_MODE_SEEDED:
        mask, seeded_info = segment_ga_seeded(
            img,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            fovea_xy=fovea_xy,
            scale_um_px=scale_um_px,
            macula_radius_mm=macula_radius_mm,
            min_size=min_size,
            use_entropy_stop=True,
        )
        meta.update(seeded_info)
    else:
        mask = segment_ga(
            img,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            fovea_xy=fovea_xy,
            scale_um_px=scale_um_px,
            macula_radius_mm=macula_radius_mm,
            min_size=min_size,
        )
        meta["mode_used"] = SEG_AUTO_MODE_OTSU
        meta["message"] = "Otsu（黄斑ROI）"

    if auto_cv and mask is not None and np.any(mask > 0):
        from engine.region_growing import refine_mask_chan_vese_light

        mask = refine_mask_chan_vese_light(img, mask)

    if info is not None:
        info.clear()
        info.update(meta)
    return mask


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
