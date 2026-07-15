"""
refine.py
---------
非AI(非深層学習)のセグメンテーション境界改善手法 5種。
すべて「既存ROI(前フレームや自動抽出の初期マスク)」を出発点として
境界だけを補正する設計。ゼロからの自動抽出ではなく、SOPの「補正」
というタスクに合わせている。

含まれる手法:
    1. band_adaptive_threshold_refine  - 境界帯の局所適応閾値化
    2. chan_vese_refine                - Active Contour (Morphological Chan-Vese)
    3. geodesic_prior_reconstruction   - 形態学的再構成による時系列シェイプ事前分布補正
    4. grabcut_refine                  - GraphCut (GrabCut) によるエネルギー最小化補正
    5. slic_boundary_snap              - SLICスーパーピクセルによる境界スナッピング
"""

from __future__ import annotations

import numpy as np
import cv2
from skimage.segmentation import morphological_chan_vese, slic
from skimage.morphology import dilation, erosion, disk, remove_small_objects, remove_small_holes
from skimage.morphology import reconstruction
from skimage.filters import threshold_local


def _to_bool(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool)


def _clean(mask_bool: np.ndarray, hole_max: int = 64, obj_max: int = 32) -> np.ndarray:
    """穴埋め・微小領域除去の共通後処理(skimage 0.22: area_threshold / min_size)。"""
    m = remove_small_holes(mask_bool, area_threshold=hole_max)
    m = remove_small_objects(m, min_size=obj_max)
    return m


# ---------------------------------------------------------------------------
# 1. 境界帯の局所適応閾値化
# ---------------------------------------------------------------------------
def band_adaptive_threshold_refine(image_u8: np.ndarray, mask: np.ndarray,
                                    band_px: int = 12, block_size: int = 35,
                                    offset: float = 5.0, dark_lesion: bool = True) -> np.ndarray:
    """
    現マスク境界の外周±band_px の「帯」領域のみ、局所適応閾値で再判定する。
    画像全体に閾値処理をかけると背景ムラの影響を受けやすいが、帯領域限定なら
    (a) 大域的な誤抽出を防ぎつつ (b) 境界のギザギザ・食い込みを滑らかにできる。
    GA(萎縮部)はFAF上で低自発蛍光(暗い)として写ることが一般的なため
    dark_lesion=True をデフォルトとする(明るい病変を扱う場合はFalseに)。
    """
    mask_b = _to_bool(mask)
    outer = dilation(mask_b, disk(band_px))
    inner = erosion(mask_b, disk(band_px))
    band = outer & ~inner

    local_thresh = threshold_local(image_u8, block_size=block_size, offset=offset)
    if dark_lesion:
        candidate = image_u8 < local_thresh
    else:
        candidate = image_u8 > local_thresh

    refined = mask_b.copy()
    refined[band] = candidate[band]
    refined = _clean(refined)
    return refined.astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Active Contour (Morphological Chan-Vese)
# ---------------------------------------------------------------------------
def chan_vese_refine(image_u8: np.ndarray, mask: np.ndarray,
                      num_iter: int = 12, smoothing: int = 1) -> np.ndarray:
    """
    現マスクを初期輪郭として Chan-Vese 型 active contour を数反復かけ、
    領域内外の輝度均質性エネルギーに基づいて境界を局所最適化する。
    RegionFinder系ソフトのcorrection toolや文献のGA/OCT境界検出でも
    採用される古典的な手法で、学習不要かつ境界の滑らかさを担保しやすい。
    """
    image_f = image_u8.astype(np.float64) / 255.0
    init_ls = _to_bool(mask).astype(np.uint8)
    if init_ls.sum() == 0:
        return mask.astype(np.uint8)

    evolved = morphological_chan_vese(
        image_f, num_iter=num_iter, init_level_set=init_ls, smoothing=smoothing,
    )
    evolved = _clean(evolved.astype(bool))
    return evolved.astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. 形態学的再構成による時系列シェイプ事前分布補正
# ---------------------------------------------------------------------------
def geodesic_prior_reconstruction(curr_mask: np.ndarray, prev_mask_registered: np.ndarray,
                                   growth_margin_px: int = 6) -> np.ndarray:
    """
    「前フレームROI(登録済み)を marker、現フレーム候補+前フレームを
    growth_margin_px だけ膨張させた領域を mask」として形態学的再構成
    (reconstruction by dilation) を行う。単純な論理和(union)と異なり、
    marker と連結していない離れた領域は再構成されないため、
    無関係な孤立領域を誤って結合しない。局所的な過小抽出(縮小異常)の
    復元に適している。
    """
    curr_b = _to_bool(curr_mask)
    prev_b = _to_bool(prev_mask_registered)
    if not prev_b.any():
        return curr_mask.astype(np.uint8)

    candidate = dilation(curr_b | prev_b, disk(growth_margin_px))
    marker = prev_b & candidate

    rec = reconstruction(marker.astype(np.float64), candidate.astype(np.float64), method="dilation")
    fixed = rec > 0.5
    fixed = fixed | curr_b  # 現フレームで明確に検出された部分は必ず保持
    fixed = remove_small_holes(fixed, area_threshold=64)
    return fixed.astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. GraphCut (GrabCut) によるエネルギー最小化補正
# ---------------------------------------------------------------------------
def grabcut_refine(image_u8: np.ndarray, mask: np.ndarray,
                    band_px: int = 15, iterations: int = 5) -> np.ndarray:
    """
    現マスクから "確実な前景/背景/不確実帯" を作り、GrabCutで
    グラフカット最適化する。画素の色(輝度)類似度と空間的な滑らかさを
    同時に考慮したエネルギー最小化のため、局所閾値だけでは残る
    ノイズ状の穴や突起を抑えやすい。皮膚病変や医用画像の境界補正で
    広く使われる古典的グラフベース手法。
    """
    mask_b = _to_bool(mask)
    if mask_b.sum() == 0:
        return mask.astype(np.uint8)

    sure_fg = erosion(mask_b, disk(max(band_px // 2, 1)))
    sure_bg = ~dilation(mask_b, disk(band_px))

    gc_mask = np.full(mask_b.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[mask_b] = cv2.GC_PR_FGD
    gc_mask[sure_fg] = cv2.GC_FGD
    gc_mask[sure_bg] = cv2.GC_BGD

    img_bgr = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask.astype(np.uint8)

    refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(bool)
    refined = _clean(refined)
    return refined.astype(np.uint8)


# ---------------------------------------------------------------------------
# 5. SLICスーパーピクセルによる境界スナッピング
# ---------------------------------------------------------------------------
def slic_boundary_snap(image_u8: np.ndarray, mask: np.ndarray,
                        n_segments: int = 600, compactness: float = 8.0,
                        overlap_thresh: float = 0.5) -> np.ndarray:
    """
    画像をSLICでスーパーピクセルに過分割し、各スーパーピクセルの
    現マスクとの重なり率が overlap_thresh を超えるものを病変側に採用する。
    ピクセル単位のギザギザ境界を、画像の自然な輝度エッジに沿った
    滑らかな境界に「スナップ」させる後処理として有効(GrabCutの
    前処理としても使われる古典的な組み合わせ)。
    """
    mask_b = _to_bool(mask)
    img_rgb = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2RGB)
    segments = slic(img_rgb, n_segments=n_segments, compactness=compactness,
                     start_label=1, channel_axis=-1)

    refined = np.zeros_like(mask_b)
    for seg_id in np.unique(segments):
        sp = segments == seg_id
        overlap = (sp & mask_b).sum() / max(sp.sum(), 1)
        if overlap >= overlap_thresh:
            refined[sp] = True

    refined = _clean(refined)
    return refined.astype(np.uint8)


METHODS = {
    "band_adaptive_threshold": band_adaptive_threshold_refine,
    "chan_vese": chan_vese_refine,
    "geodesic_prior_reconstruction": geodesic_prior_reconstruction,  # 特殊シグネチャ(prev_maskが必要)
    "grabcut": grabcut_refine,
    "slic_boundary_snap": slic_boundary_snap,
}
