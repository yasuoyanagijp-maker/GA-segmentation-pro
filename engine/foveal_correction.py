"""foveal_correction.py – 中心窩黄斑色素ブロック（低蛍光）の半径方向補正。

Blue-FAF では中心窩が滑らかな同心円状に暗くなるが、完全な GA は急峻な低蛍光として
保護し、補正は低周波の中心窩成分に限定する。
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter1d
from skimage.measure import label, regionprops


MACULA_RADIUS_MM = 3.0
FOVEAL_CORRECTION_RADIUS_MM = 2.0
PROFILE_STEP_MM = 0.05
GA_PROTECT_FRACTION = 0.40

# 健常網膜の参照輝度を取る同心円リング（補正半径に対する比）
REF_RING_INNER_FRAC = 0.6
REF_RING_OUTER_FRAC = 1.6
REF_PERCENTILE = 60.0
# 鋭い暗縁（真の GA 境界）を持ち上げから保護する閾値
SHARP_EDGE_BLUR_MM = 0.12
SHARP_EDGE_DROP = 22.0


def _dist_mm_grid(
    shape: tuple[int, int],
    fovea_xy: tuple[float, float],
    scale_um_px: float,
) -> np.ndarray:
    h, w = shape
    fx, fy = fovea_xy
    yy, xx = np.ogrid[:h, :w]
    dist_px = np.sqrt((xx - fx) ** 2 + (yy - fy) ** 2)
    return dist_px * scale_um_px / 1000.0


def _annulus_profile(
    gray: np.ndarray,
    dist_mm: np.ndarray,
    *,
    inner_mm: float,
    outer_mm: float,
    bright_pct: float = 65.0,
) -> float | None:
    ring = (dist_mm >= inner_mm) & (dist_mm < outer_mm)
    if not np.any(ring):
        return None
    vals = gray[ring].astype(np.float32)
    return float(np.percentile(vals, bright_pct))


def _build_expected_field(
    gray: np.ndarray,
    dist_mm: np.ndarray,
) -> np.ndarray:
    """半径方向プロファイルから滑らかな期待輝度場を構築。"""
    edges = np.arange(0.0, MACULA_RADIUS_MM + PROFILE_STEP_MM, PROFILE_STEP_MM)
    centers = (edges[:-1] + edges[1:]) / 2.0
    profile: list[float] = []
    for i in range(len(edges) - 1):
        val = _annulus_profile(gray, dist_mm, inner_mm=edges[i], outer_mm=edges[i + 1])
        profile.append(val if val is not None else np.nan)

    arr = np.array(profile, dtype=np.float64)
    valid = np.isfinite(arr)
    if valid.sum() < 5:
        return gray.astype(np.float32)

    ref_mask = (centers >= 1.0) & (centers <= 2.2) & valid
    reference = float(np.median(arr[ref_mask])) if ref_mask.any() else float(np.median(arr[valid]))

    filled = arr.copy()
    filled[~valid] = reference
    smoothed = gaussian_filter1d(filled, sigma=2.0, mode="nearest")

    # 中心窩の滑らかな減衰のみをモデル化（外周は reference に漸近）
    outer_blend = np.clip((centers - 0.4) / 1.0, 0.0, 1.0)
    expected_1d = reference * (1.0 - outer_blend) + smoothed * outer_blend
    expected_1d = np.minimum(expected_1d, reference)

    return np.interp(
        dist_mm.ravel(),
        centers,
        expected_1d,
    ).reshape(dist_mm.shape).astype(np.float32)


def build_ga_protection_mask(
    gray: np.ndarray,
    dist_mm: np.ndarray,
    expected_field: np.ndarray,
    *,
    ga_protect_fraction: float = GA_PROTECT_FRACTION,
) -> np.ndarray:
    """急峻な低蛍光（GA 候補）を補正対象から除外。"""
    obs = gray.astype(np.float32)
    protect = np.zeros(obs.shape, dtype=bool)
    central = dist_mm <= 2.0

    # 真の GA 低蛍光: 滑らかな中心窩ブロックよりさらに暗い画素
    deficit = expected_field - obs
    protect |= central & (obs < 55.0) & (deficit > 35.0)

    if central.any():
        dark_thresh = float(np.percentile(obs[central], 22))
        blob_seed = (obs <= dark_thresh) & central
        labeled = label(blob_seed)
        for region in regionprops(labeled):
            if region.area < 9:
                continue
            mask = labeled == region.label
            region_mean = float(obs[mask].mean())
            cy, cx = (int(region.centroid[0]), int(region.centroid[1]))
            local_expected = float(expected_field[cy, cx])
            if region_mean >= local_expected * 0.58:
                continue
            dilated = binary_dilation(mask, iterations=2)
            ring = dilated & ~mask
            if not ring.any():
                continue
            contrast = float(obs[ring].mean()) - region_mean
            if contrast >= 22.0:
                protect[mask] = True

    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    protect_u8 = protect.astype(np.uint8) * 255
    protect_u8 = cv2.morphologyEx(protect_u8, cv2.MORPH_CLOSE, struct, iterations=1)
    protect_u8 = cv2.dilate(protect_u8, struct, iterations=1)
    return protect_u8 > 0


def _radial_reference(obs: np.ndarray, dist_mm: np.ndarray, radius_mm: float) -> float:
    """健常網膜（中心窩ブロックの外側）の代表輝度。

    セグメンテーション閾値は健常網膜レベルで決まるため、持ち上げ目標も
    「近傍プロファイルの中央値」ではなく実際の健常レベルに合わせる必要がある。
    """
    inner = radius_mm * REF_RING_INNER_FRAC
    outer = radius_mm * REF_RING_OUTER_FRAC
    ring = (dist_mm >= inner) & (dist_mm <= outer)
    if int(ring.sum()) >= 200:
        return float(np.percentile(obs[ring], REF_PERCENTILE))
    # リングが画像外に出る小画像はフォールバック（全体の明部）
    return float(np.percentile(obs, 80.0))


def _sharp_edge_protect(obs: np.ndarray, scale_um_px: float) -> np.ndarray:
    """局所的に急峻な低蛍光（真の GA 境界）を持ち上げから保護。"""
    sigma_px = max(SHARP_EDGE_BLUR_MM * 1000.0 / scale_um_px, 4.0)
    smooth = cv2.GaussianBlur(obs, (0, 0), sigmaX=sigma_px)
    return (smooth - obs) > SHARP_EDGE_DROP


def _radial_deficit_field(
    obs: np.ndarray,
    dist_mm: np.ndarray,
    reference: float,
    radius_mm: float,
) -> np.ndarray:
    """同心円（半径方向に対称）成分だけの輝度不足量を再構成する。

    各同心円の *中央値* がどれだけ reference より暗いかを不足量とする。
    局所的な GA は円環の一部しか占めないため中央値をほとんど下げず、
    構造的に持ち上げ対象から外れる（= 保護される）。滑らかな中心窩ブロックは
    円環全体を暗くするため中央値が下がり、持ち上げ対象になる。
    """
    edges = np.arange(0.0, radius_mm + PROFILE_STEP_MM, PROFILE_STEP_MM)
    centers = (edges[:-1] + edges[1:]) / 2.0
    med = np.full(len(centers), np.nan, dtype=np.float64)
    for i in range(len(centers)):
        ring = (dist_mm >= edges[i]) & (dist_mm < edges[i + 1])
        if ring.any():
            med[i] = float(np.median(obs[ring]))
    valid = np.isfinite(med)
    if valid.sum() < 3:
        return np.zeros_like(obs)
    med[~valid] = reference
    med_s = gaussian_filter1d(med, sigma=2.0, mode="nearest")
    deficit_1d = np.clip(reference - med_s, 0.0, None)
    radial_1d = np.clip(1.0 - centers / radius_mm, 0.0, 1.0)
    lift_1d = deficit_1d * radial_1d
    return np.interp(
        dist_mm.ravel(), centers, lift_1d, right=0.0,
    ).reshape(dist_mm.shape).astype(np.float32)


def apply_foveal_block_correction(
    gray: np.ndarray,
    fovea_xy: tuple[float, float],
    scale_um_px: float,
    strength: float,
    *,
    foveal_radius_mm: float = FOVEAL_CORRECTION_RADIUS_MM,
    ga_protect_fraction: float = GA_PROTECT_FRACTION,
) -> np.ndarray:
    """中心窩の同心円状ブロックを補正した uint8 画像を返す。

    strength: 0=補正なし, 1=最大補正

    方針（同心円フラット化）:
      1. 健常網膜レベル reference を推定（持ち上げ目標）。
      2. 半径方向中央値プロファイルの不足量だけを持ち上げる → 同心円成分のみ除去。
         局所 GA は円環中央値をほとんど動かさないため構造的に保護される。
      3. 鋭い暗縁（真の GA 境界）は追加で保護。
      4. reference を超える過補正はしない。
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0 or scale_um_px <= 0:
        return gray.copy()

    obs = gray.astype(np.float32)
    dist_mm = _dist_mm_grid(gray.shape, fovea_xy, scale_um_px)
    reference = _radial_reference(obs, dist_mm, foveal_radius_mm)

    lift = _radial_deficit_field(obs, dist_mm, reference, foveal_radius_mm) * strength
    lift[_sharp_edge_protect(obs, scale_um_px)] = 0.0
    # reference を超えて明るくしない（過補正防止）
    lift = np.minimum(lift, np.clip(reference - obs, 0.0, None))
    lift = np.clip(lift, 0.0, None)

    return np.clip(obs + lift, 0, 255).astype(np.uint8)


def correction_effect_stats(
    original: np.ndarray,
    corrected: np.ndarray,
    fovea_xy: tuple[float, float],
    scale_um_px: float,
    *,
    foveal_radius_mm: float = FOVEAL_CORRECTION_RADIUS_MM,
    mask_raw: np.ndarray | None = None,
    mask_corr: np.ndarray | None = None,
) -> dict:
    """補正の実効性（表示ラベル用）。

    画素輝度の変化に加え、セグメント面積が縮小した場合も effective=True。
    （中心窩が GA 保護されて輝度が不変でも、周辺補正でマスクが小さくなることがある）
    """
    diff = np.abs(corrected.astype(np.int16) - original.astype(np.int16))
    h, w = original.shape[:2]
    fx, fy = int(fovea_xy[0]), int(fovea_xy[1])
    fx = max(0, min(w - 1, fx))
    fy = max(0, min(h - 1, fy))
    r_px = foveal_radius_mm * 1000.0 / scale_um_px
    yy, xx = np.ogrid[:h, :w]
    central = ((xx - fx) ** 2 + (yy - fy) ** 2) <= r_px ** 2
    center_lift = int(corrected[fy, fx]) - int(original[fy, fx])
    central_lift_mean = float(
        corrected[central].astype(np.float32).mean()
        - original[central].astype(np.float32).mean()
    ) if central.any() else 0.0

    area_raw = int(np.sum(mask_raw > 0)) if mask_raw is not None else None
    area_corr = int(np.sum(mask_corr > 0)) if mask_corr is not None else None
    area_reduction_pct = None
    if area_raw is not None and area_corr is not None and area_raw > 0:
        area_reduction_pct = 100.0 * (area_raw - area_corr) / area_raw

    intensity_effective = bool(
        center_lift > 0
        or (central_lift_mean > 3.0 and float(diff.mean()) > 0.15)
    )
    mask_effective = bool(
        area_reduction_pct is not None and area_reduction_pct >= 1.0
    )
    return {
        "changed_px": int(np.sum(diff > 0)),
        "mean_diff": float(diff.mean()),
        "center_lift": center_lift,
        "central_lift_mean": central_lift_mean,
        "area_raw": area_raw,
        "area_corr": area_corr,
        "area_reduction_pct": area_reduction_pct,
        "effective": intensity_effective or mask_effective,
        "mask_effective": mask_effective,
        "intensity_effective": intensity_effective,
    }


def gray_for_segmentation(
    gray_working: np.ndarray,
    gray_original: np.ndarray | None,
    fovea_xy: tuple[float, float] | None,
    scale_um_px: float | None,
    *,
    correction_applied: bool,
    correction_skipped: bool,
    correction_strength: float,
) -> np.ndarray:
    """セグメンテーション入力画像を決定する（原画像 + 補正設定から毎回算出）。"""
    base = gray_original if gray_original is not None else gray_working
    if (
        correction_applied
        and not correction_skipped
        and fovea_xy is not None
        and scale_um_px is not None
        and correction_strength > 0
    ):
        return apply_foveal_block_correction(
            base, fovea_xy, scale_um_px, correction_strength,
        )
    return base.copy()
