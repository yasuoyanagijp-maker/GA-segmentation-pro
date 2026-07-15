"""
refine.py
---------
非AI(非深層学習)のセグメンテーション境界改善手法 5種。
すべて「既存ROI(前フレームや自動抽出の初期マスク)」を出発点として
境界だけを補正する設計。ゼロからの自動抽出ではなく、SOPの「補正」
というタスクに合わせている。

含まれる手法:
    1. band_adaptive_threshold_refine    - 境界帯の局所適応閾値化
    1b. position_guided_monotonic_refine - 位置ずれフレーム向け: 前後フレーム誘導 + 単調性制約
    2. chan_vese_refine                  - Active Contour (Morphological Chan-Vese)
    3. geodesic_prior_reconstruction     - 形態学的再構成による時系列シェイプ事前分布補正
    4. grabcut_refine                    - GraphCut (GrabCut) によるエネルギー最小化補正
    5. slic_boundary_snap                - SLICスーパーピクセルによる境界スナッピング
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage
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
                                    offset: float = 5.0, dark_lesion: bool = True,
                                    base_mask: np.ndarray | None = None) -> np.ndarray:
    """
    帯(マスク境界の外周±band_px)の領域のみ、局所適応閾値で再判定する。
    画像全体に閾値処理をかけると背景ムラの影響を受けやすいが、帯領域限定なら
    (a) 大域的な誤抽出を防ぎつつ (b) 境界のギザギザ・食い込みを滑らかにできる。
    GA(萎縮部)はFAF上で低自発蛍光(暗い)として写ることが一般的なため
    dark_lesion=True をデフォルトとする(明るい病変を扱う場合はFalseに)。

    base_mask: 帯の「中心」をどこに置くかを決めるマスク。省略時は mask
    (=現フレーム自身のROI)を使う。現フレームのROIが大きく位置ずれして
    いる疑いがある場合は、前後フレームから作った prior(位置の事前分布)
    を base_mask として渡すことで、間違った位置ではなく妥当な位置の
    周辺で局所閾値を走らせられる。band内の判定自体は常に image_u8 の
    実際の輝度に基づくので、base_maskがある程度ずれていても帯が
    真の境界にかかっていれば結果は追従する。
    """
    base_b = _to_bool(base_mask if base_mask is not None else mask)
    outer = dilation(base_b, disk(band_px))
    inner = erosion(base_b, disk(band_px))
    band = outer & ~inner

    local_thresh = threshold_local(image_u8, block_size=block_size, offset=offset)
    if dark_lesion:
        candidate = image_u8 < local_thresh
    else:
        candidate = image_u8 > local_thresh

    refined = base_b.copy()
    refined[band] = candidate[band]
    refined = _clean(refined)
    return refined.astype(np.uint8)


# ---------------------------------------------------------------------------
# 1b. 位置ずれフレーム向け: 前後フレーム誘導 + 単調性制約つき境界補正
# ---------------------------------------------------------------------------
def interpolate_prior_mask(prev_mask: np.ndarray | None, next_mask: np.ndarray | None,
                            t: float = 0.5) -> np.ndarray | None:
    """
    Shape-based interpolation(符号付き距離変換による形状補間)。
    登録済みの前後フレームROIから、現フレーム時点でのおおよその位置・
    形状を推定する。現フレーム自身のマスクは「位置」の手がかりとして
    信頼できない(大きくずれている)前提のときに使う prior を作る関数。

    前後どちらか一方しかない場合はそれをそのまま返す(簡易外挿)。
    両方無ければ None。
    """
    if prev_mask is None and next_mask is None:
        return None
    if prev_mask is None:
        return next_mask.astype(np.uint8)
    if next_mask is None:
        return prev_mask.astype(np.uint8)

    prev_b, next_b = _to_bool(prev_mask), _to_bool(next_mask)
    d_prev = ndimage.distance_transform_edt(prev_b) - ndimage.distance_transform_edt(~prev_b)
    d_next = ndimage.distance_transform_edt(next_b) - ndimage.distance_transform_edt(~next_b)
    d_interp = (1 - t) * d_prev + t * d_next
    return (d_interp > 0).astype(np.uint8)


def trusted_prior_mask(prev_mask: np.ndarray | None, next_mask: np.ndarray | None,
                        t: float = 0.5,
                        min_neighbor_agreement: float = 0.3) -> np.ndarray | None:
    """
    interpolate_prior_mask のガード付き版。両隣が存在する場合、
    prev と next が互いに min_neighbor_agreement 以上重なっている
    (=補間priorとして信頼できる)ときだけ prior を返す。

    これが無いと、位置ずれフレームの「隣」の正常フレームを検査するとき、
    ずれたフレームを含む prior との不一致で正常フレームまで
    position_anomaly と誤判定し、連鎖的に破壊してしまう。
    """
    if prev_mask is not None and next_mask is not None:
        from .qc import dice
        if dice(prev_mask, next_mask) < min_neighbor_agreement:
            return None
    return interpolate_prior_mask(prev_mask, next_mask, t=t)


def position_guided_monotonic_refine(image_u8: np.ndarray, curr_mask: np.ndarray,
                                      prev_mask: np.ndarray | None = None,
                                      next_mask: np.ndarray | None = None,
                                      band_px: int = 12, block_size: int = 35,
                                      offset: float = 5.0, dark_lesion: bool = True,
                                      enforce_monotonic: bool = True,
                                      next_margin_px: int = 3) -> np.ndarray:
    """
    現フレームのROIが大きく位置ずれしている疑いがあるときの補正手順。

      1) 前後フレーム(登録済み)から shape-based interpolation で
         位置の prior を作る(現マスクの位置は信頼しない)。
      2) その prior を band_adaptive_threshold_refine の base_mask として
         渡し、実際の画像輝度で境界を帯領域だけフィットさせる。
      3) GA(萎縮)は基本的に非退行性(時間とともに拡大)という仮定に
         基づき、単調性を強制する:
             result ⊇ prev_mask        (前フレームより小さくしない)
             result ⊆ dilate(next_mask, next_margin_px)
                                        (次フレームの拡張分+マージンを超えない)
         next_margin_px は次フレーム自体の境界誤差を吸収するための
         許容幅(px)。厳密に "next以下" にしたい場合は 0 にする。

    前後フレームが片方しか無い場合はその一方だけで prior・制約を作る。
    両方無い場合は通常の band_adaptive_threshold_refine と同じ挙動になる。
    """
    prior = interpolate_prior_mask(prev_mask, next_mask, t=0.5)
    base = prior if prior is not None else curr_mask

    refined = band_adaptive_threshold_refine(
        image_u8, curr_mask, band_px=band_px, block_size=block_size,
        offset=offset, dark_lesion=dark_lesion, base_mask=base,
    )

    if enforce_monotonic:
        refined_b = _to_bool(refined)
        if prev_mask is not None:
            refined_b = refined_b | _to_bool(prev_mask)  # 下限: 前フレームを必ず含む
        if next_mask is not None:
            upper = _to_bool(next_mask)
            if next_margin_px > 0:
                upper = dilation(upper, disk(next_margin_px))
            refined_b = refined_b & upper                 # 上限: 次フレーム(+マージン)を超えない
        refined = _clean(refined_b).astype(np.uint8)

    return refined


# ---------------------------------------------------------------------------
# 1c. 画像拘束付き単調封筒 (∪prev ∩ dilate(last) + band fit)
# ---------------------------------------------------------------------------
def image_guided_monotonic_envelope(
    image_u8: np.ndarray,
    curr_mask: np.ndarray,
    *,
    prev_mask: np.ndarray | None = None,
    last_mask: np.ndarray | None = None,
    last_margin_px: int = 3,
    band_px: int = 12,
    block_size: int = 35,
    offset: float = 5.0,
    dark_lesion: bool = True,
) -> np.ndarray:
    """
    案B: 論理的な単調封筒のあと、境界帯だけ画像輝度でフィットする。

      1) envelope = curr ∪ prev          （前フレームより縮めない）
      2) envelope ∩= dilate(last, m)     （最終フレームを超えない）
      3) band_adaptive_threshold(base=envelope) で輝度に境界を合わせる
      4) 再度 ⊇ prev かつ ⊆ dilate(last) を強制（band で崩れた制約を戻す）

    prev / last が無い場合はその制約をスキップする。
    """
    curr_b = _to_bool(curr_mask)
    envelope = curr_b.copy()
    if prev_mask is not None:
        envelope = envelope | _to_bool(prev_mask)

    upper = None
    if last_mask is not None:
        upper = _to_bool(last_mask)
        if last_margin_px > 0:
            upper = dilation(upper, disk(int(last_margin_px)))
        envelope = envelope & upper

    refined = band_adaptive_threshold_refine(
        image_u8, curr_mask,
        band_px=band_px, block_size=block_size, offset=offset,
        dark_lesion=dark_lesion, base_mask=envelope.astype(np.uint8),
    )
    refined_b = _to_bool(refined)
    if prev_mask is not None:
        refined_b = refined_b | _to_bool(prev_mask)
    if upper is not None:
        refined_b = refined_b & upper
    return _clean(refined_b).astype(np.uint8)


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
    "position_guided_monotonic": position_guided_monotonic_refine,  # 特殊シグネチャ(prev/next_maskが必要)
    "image_guided_monotonic": image_guided_monotonic_envelope,  # 特殊シグネチャ(prev/last_maskが必要)
    "chan_vese": chan_vese_refine,
    "geodesic_prior_reconstruction": geodesic_prior_reconstruction,  # 特殊シグネチャ(prev_maskが必要)
    "grabcut": grabcut_refine,
    "slic_boundary_snap": slic_boundary_snap,
}
