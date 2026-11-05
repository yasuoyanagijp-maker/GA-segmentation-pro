"""Optos 緑 AF 向けの初期マスクと計測。

現行の ``segment_ga`` は呼ばない。黄斑窓の中で乳頭を埋め、多巣の暗コアを
縁まで成長させ、面積は直径 6 mm（半径 3 mm）の円の中だけを mm² にする。
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np
from skimage.measure import label, regionprops

from engine.calibration import (
    DEFAULT_DFD_UM,
    DEFAULT_DISC_DIAMETER_UM,
    compute_scale_um_per_px_from_dfd,
)
from engine.region_growing import (
    GROW_MODE_STANDARD,
    expand_segmentation_from_mask,
)
from engine.segmentation import (
    _dark_seed_mask_and_points,
    _enhance,
    _macula_roi_mask,
    remove_small_regions,
)
from engine.stats import ga_stats

logger = logging.getLogger("ga.optos.segment")

# Spectralis 約 10.5 µm/px での 7 px / 64 px を物理サイズに固定する。
SPECTRALIS_UM_PER_PX = 10.5
MORPH_UM = 7.0 * SPECTRALIS_UM_PER_PX
DISC_RADIUS_MM = 0.9
# シアン円と面積の範囲。6 mm は直径で、GAWAIN の黄斑（半径 3 mm）と同じ。
QUANT_RADIUS_MM = 3.0
WINDOW_RADIUS_MM = 8.0
# 半径 4 mm・追加 15 mm² に広げると、後の時点で暗い網膜まで広がり、
# 面積が約 3 倍になり提案した中心窩まで届く。停止は明るさ（a）で行う。
AUTO_GROW_RADIUS_MM = 1.5
AUTO_GROW_MAX_MM2 = 3.0
MIN_COMPONENT_MM2 = 0.01
BRUSH_RADIUS_MM = 0.15
MAX_AUTO_COMPONENTS = 12


def px_from_mm(mm: float, scale_um_px: float) -> int:
    if scale_um_px <= 0:
        raise ValueError("scale_um_px は正の値が必要です")
    return max(1, int(round(float(mm) * 1000.0 / float(scale_um_px))))


def px_from_mm2(mm2: float, scale_um_px: float) -> int:
    if scale_um_px <= 0:
        raise ValueError("scale_um_px は正の値が必要です")
    return max(1, int(round(float(mm2) * 1e6 / (float(scale_um_px) ** 2))))


def scale_from_dfd(
    disc_xy: tuple[float, float],
    fovea_xy: tuple[float, float],
    dfd_um: float = DEFAULT_DFD_UM,
) -> float:
    return compute_scale_um_per_px_from_dfd(fovea_xy, disc_xy, dfd_um)


# 乳頭の縁が読めないときの予備。この比率未満のずれはクリックのばらつきとして据え置く。
_DFD_HOLD_RATIO = 1.08


def snap_disc_center(
    gray: np.ndarray,
    disc_xy: tuple[float, float],
    *,
    search_px: int = 100,
    accept_px: float = 16.0,
) -> tuple[float, float]:
    """クリックが乳頭中心から十数画素以内なら、暗い円の重心へ戻す。

    探索窓は乳頭の縁が見える広さにする。重心がクリックから accept_px
    より遠ければ、別の場所をクリックしたとみなして動かさない。
    """
    h, w = gray.shape[:2]
    cx, cy = float(disc_xy[0]), float(disc_xy[1])
    if not (0 <= cx < w and 0 <= cy < h):
        return cx, cy
    radius = int(search_px)
    x0, x1 = max(0, int(round(cx)) - radius), min(w, int(round(cx)) + radius + 1)
    y0, y1 = max(0, int(round(cy)) - radius), min(h, int(round(cy)) + radius + 1)
    crop = gray[y0:y1, x0:x1]
    if crop.size < 16:
        return cx, cy
    border = np.concatenate([crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]])
    background = float(np.median(border))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= float(radius * radius)
    dark = (crop.astype(np.float32) <= background - 25.0) & inside
    if int(dark.sum()) < 20:
        return cx, cy
    weight = np.where(dark, np.clip(background - crop.astype(np.float32), 0, None), 0.0)
    total = float(weight.sum())
    if total <= 0:
        return cx, cy
    sx = float((weight * xx).sum() / total)
    sy = float((weight * yy).sum() / total)
    if math.hypot(sx - cx, sy - cy) > float(accept_px):
        return cx, cy
    return sx, sy


def stable_disc_radius(
    gray: np.ndarray,
    disc_xy: tuple[float, float],
) -> tuple[tuple[float, float], float | None]:
    """クリックを乳頭の重心へ寄せ、縁までの半径（px）の中央値を返す。"""
    center = snap_disc_center(gray, disc_xy)
    radii: list[float] = []
    for dx, dy in ((0.0, 0.0), (3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0)):
        radius = estimate_disc_radius_px(gray, (center[0] + dx, center[1] + dy))
        if radius is not None and radius >= 1.0:
            radii.append(float(radius))
    if not radii:
        return center, None
    return center, float(np.median(radii))


def stable_rim_scale(
    gray: np.ndarray,
    disc_xy: tuple[float, float],
    disc_diameter_um: float = DEFAULT_DISC_DIAMETER_UM,
) -> tuple[tuple[float, float], float | None]:
    """乳頭径 1.8 mm の仮スケール。中心窩を仮置きするときだけ使う。"""
    center, radius = stable_disc_radius(gray, disc_xy)
    if radius is None or radius < 1.0:
        return center, None
    return center, float(disc_diameter_um) / (2.0 * radius)


def series_scale_from_dfd(
    anchor_scale_um_px: float,
    anchor_disc_xy: tuple[float, float],
    anchor_fovea_xy: tuple[float, float],
    disc_xy: tuple[float, float],
    fovea_xy: tuple[float, float],
) -> float:
    """2枚目以降の µm/px。1枚目と、この画像の乳頭–中心窩の画素距離の比。"""
    dfd_anchor = math.hypot(
        float(anchor_fovea_xy[0]) - float(anchor_disc_xy[0]),
        float(anchor_fovea_xy[1]) - float(anchor_disc_xy[1]),
    )
    dfd_here = math.hypot(
        float(fovea_xy[0]) - float(disc_xy[0]),
        float(fovea_xy[1]) - float(disc_xy[1]),
    )
    if anchor_scale_um_px <= 0 or dfd_anchor < 1.0 or dfd_here < 1.0:
        raise ValueError("乳頭–中心窩の画素距離とスケールは正の値が必要です")
    return float(anchor_scale_um_px) * dfd_anchor / dfd_here


def series_scale_um_px(
    anchor_scale_um_px: float,
    anchor_radius_px: float,
    radius_px: float,
) -> float:
    """1枚目の µm/px を、乳頭半径の比でこの画像のズームに合わせる。"""
    if anchor_scale_um_px <= 0 or anchor_radius_px <= 0 or radius_px <= 0:
        raise ValueError("乳頭半径とスケールは正の値が必要です")
    return float(anchor_scale_um_px) * float(anchor_radius_px) / float(radius_px)


def fovea_from_reference(
    disc_xy: tuple[float, float],
    vector_xy: tuple[float, float],
    anchor_radius_px: float,
    radius_px: float,
    shape: tuple[int, int],
) -> tuple[float, float]:
    """1枚目の乳頭→中心窩ベクトルを、この画像の乳頭の大きさで画素に戻す。"""
    if anchor_radius_px <= 0 or radius_px <= 0:
        raise ValueError("乳頭半径は正の値が必要です")
    factor = float(radius_px) / float(anchor_radius_px)
    h, w = shape
    fx = float(disc_xy[0]) + float(vector_xy[0]) * factor
    fy = float(disc_xy[1]) + float(vector_xy[1]) * factor
    return float(np.clip(fx, 0, w - 1)), float(np.clip(fy, 0, h - 1))


def measurement_scale(
    gray: np.ndarray,
    disc_xy: tuple[float, float],
    fovea_xy: tuple[float, float] | None = None,
    *,
    previous: float | None = None,
) -> tuple[tuple[float, float], float | None, str]:
    """1枚目用。戻り値は (乳頭中心, µm/px, 由来)。

    由来は dfd / dfd-held / none。中心窩をクリックした距離を 4.76 mm にする。
    直前の換算から 8% 以内のずれは、クリックのばらつきとして据え置く。
    """
    center = snap_disc_center(gray, disc_xy)
    if fovea_xy is None:
        return center, None, "none"
    dfd = scale_from_dfd(center, fovea_xy)
    if previous is not None and previous > 0:
        ratio = max(dfd, float(previous)) / min(dfd, float(previous))
        if ratio < _DFD_HOLD_RATIO:
            return center, float(previous), "dfd-held"
    return center, dfd, "dfd"


def estimate_disc_radius_px(
    gray: np.ndarray,
    disc_xy: tuple[float, float],
    *,
    r_min: int = 24,
    r_max: int = 160,
) -> float | None:
    """クリックした乳頭中心から、暗い乳頭の外縁までの半径（px）。

    画面のズームが変わるとこの半径が変わる。読めなければ None。
    """
    h, w = gray.shape[:2]
    cx, cy = float(disc_xy[0]), float(disc_xy[1])
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    limit = int(min(r_max, cx - 1, cy - 1, w - 2 - cx, h - 2 - cy))
    if limit <= r_min + 8:
        return None
    radii = np.arange(r_min, limit + 1)
    theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    xs = cx + radii[:, None] * np.cos(theta)[None, :]
    ys = cy + radii[:, None] * np.sin(theta)[None, :]
    samples = gray[np.rint(ys).astype(int), np.rint(xs).astype(int)].astype(np.float32)
    profile = np.median(samples, axis=1)
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    smooth = np.convolve(profile, kernel, mode="same")
    grad = np.diff(smooth)
    if grad.size < 8:
        return None
    search = grad.copy()
    search[:4] = -1e9
    search[-3:] = -1e9
    edge = int(np.argmax(search))
    if float(search[edge]) < 1.5:
        return None
    inside = float(np.median(smooth[: max(3, edge // 2)]))
    outside_end = min(len(smooth), edge + 18)
    outside = float(np.median(smooth[edge + 3: outside_end])) if edge + 6 < len(smooth) else float(smooth[-1])
    if outside < inside + 12:
        return None
    return float(radii[edge] + 1)


def provisional_scale_from_disc(
    gray: np.ndarray,
    disc_xy: tuple[float, float],
    disc_diameter_um: float = DEFAULT_DISC_DIAMETER_UM,
) -> float | None:
    """乳頭径 1.8 mm から、この画像だけの仮スケール（µm/px）。中心窩を置くため。"""
    radius = estimate_disc_radius_px(gray, disc_xy)
    if radius is None or radius < 1:
        return None
    return float(disc_diameter_um) / (2.0 * radius)


def propose_fovea_from_disc(
    disc_xy: tuple[float, float],
    shape: tuple[int, int],
    *,
    scale_um_px: float | None = None,
) -> tuple[float, float]:
    """乳頭から画像中心へ向かう仮の中心窩。

    スケールが既知なら距離は DFD 4.76 mm。未知なら短辺の 18% で、
    同じ画素サイズの画面でもズーム差は入らない。ズームが違う画像では
    ``provisional_scale_from_disc`` を渡す。下方向へ 0.3 mm
    （スケール未知なら短辺の 2%）ずらす。
    """
    h, w = shape
    dx, dy = float(disc_xy[0]), float(disc_xy[1])
    vx, vy = (w - 1) / 2.0 - dx, (h - 1) / 2.0 - dy
    norm = math.hypot(vx, vy)
    if norm < 1.0:
        vx, vy, norm = -1.0, 0.35, math.hypot(-1.0, 0.35)
    if scale_um_px is not None and scale_um_px > 0:
        dist = DEFAULT_DFD_UM / float(scale_um_px)
        drop = 300.0 / float(scale_um_px)
    else:
        dist = 0.18 * float(min(h, w))
        drop = 0.02 * float(min(h, w))
    fx = dx + vx / norm * dist
    fy = dy + vy / norm * dist + drop
    fx = float(np.clip(fx, 0, w - 1))
    fy = float(np.clip(fy, 0, h - 1))
    return fx, fy


def disc_circle_mask(
    shape: tuple[int, int],
    disc_xy: tuple[float, float],
    scale_um_px: float,
    radius_mm: float = DISC_RADIUS_MM,
) -> np.ndarray:
    h, w = shape
    radius = px_from_mm(radius_mm, scale_um_px)
    yy, xx = np.ogrid[:h, :w]
    return (xx - disc_xy[0]) ** 2 + (yy - disc_xy[1]) ** 2 <= radius ** 2


def central_roi_mask(
    shape: tuple[int, int],
    fovea_xy: tuple[float, float],
    scale_um_px: float,
    radius_mm: float = QUANT_RADIUS_MM,
) -> np.ndarray:
    h, w = shape
    radius = px_from_mm(radius_mm, scale_um_px)
    yy, xx = np.ogrid[:h, :w]
    return (xx - fovea_xy[0]) ** 2 + (yy - fovea_xy[1]) ** 2 <= radius ** 2


def window_box(
    shape: tuple[int, int],
    fovea_xy: tuple[float, float],
    scale_um_px: float,
    radius_mm: float = WINDOW_RADIUS_MM,
) -> tuple[int, int, int, int]:
    """中心窩中心の正方形 (x0, y0, x1, y1)。x1/y1 はスライスの排他端。"""
    h, w = shape
    radius = px_from_mm(radius_mm, scale_um_px)
    x0 = max(0, int(math.floor(fovea_xy[0] - radius)))
    y0 = max(0, int(math.floor(fovea_xy[1] - radius)))
    x1 = min(w, int(math.ceil(fovea_xy[0] + radius)) + 1)
    y1 = min(h, int(math.ceil(fovea_xy[1] + radius)) + 1)
    if x1 <= x0 + 1:
        x1 = min(w, x0 + 2)
    if y1 <= y0 + 1:
        y1 = min(h, y0 + 2)
    return x0, y0, x1, y1


def shift_point(
    xy: tuple[float, float] | None,
    origin_xy: tuple[int, int],
) -> tuple[float, float] | None:
    if xy is None:
        return None
    return float(xy[0] - origin_xy[0]), float(xy[1] - origin_xy[1])


def flatten_illumination(gray: np.ndarray, scale_um_px: float) -> np.ndarray:
    """低周波の周辺減光だけを持ち上げる。中心窩の色素補正はしない。"""
    sigma = float(px_from_mm(2.0, scale_um_px))
    src = gray.astype(np.float32)
    blur = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma, sigmaY=sigma)
    med = float(np.median(blur))
    lifted = src + (med - blur)
    return np.clip(lifted, 0, 255).astype(np.uint8)


def vessel_barrier(gray: np.ndarray) -> np.ndarray:
    """細い暗線を成長の壁にする。大きい暗塊（萎縮そのもの）は壁にしない。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    if int(hat.max()) < 8:
        return np.zeros(gray.shape, dtype=np.uint8)
    thr = float(np.percentile(hat, 97))
    thin = (hat >= max(thr, 8.0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    out = np.zeros(gray.shape, dtype=np.uint8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        long_side = max(bw, bh)
        short_side = max(1, min(bw, bh))
        if area < 12 or area > 800:
            continue
        if long_side / short_side < 2.5:
            continue
        out[labels == i] = 255
    return out


def _fill_disc(
    gray: np.ndarray,
    disc_xy: tuple[float, float],
    scale_um_px: float,
) -> np.ndarray:
    disc = disc_circle_mask(gray.shape, disc_xy, scale_um_px)
    if not np.any(disc):
        return gray
    radius = px_from_mm(DISC_RADIUS_MM, scale_um_px)
    yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
    ring = (
        ((xx - disc_xy[0]) ** 2 + (yy - disc_xy[1]) ** 2 > radius ** 2)
        & ((xx - disc_xy[0]) ** 2 + (yy - disc_xy[1]) ** 2 <= (radius + 4) ** 2)
    )
    fill = float(np.median(gray[ring])) if np.any(ring) else float(np.median(gray))
    out = gray.copy()
    out[disc] = np.clip(int(round(fill)), 0, 255)
    return out


def _cores(
    enhanced: np.ndarray,
    fovea_xy: tuple[float, float],
    scale_um_px: float,
    disc: np.ndarray,
    min_lesion_mm2: float = MIN_COMPONENT_MM2,
) -> np.ndarray:
    seed_mask, _seeds = _dark_seed_mask_and_points(
        enhanced, fovea_xy, scale_um_px, QUANT_RADIUS_MM,
    )
    roi = _macula_roi_mask(enhanced.shape, fovea_xy, scale_um_px, QUANT_RADIUS_MM)
    if not np.any(seed_mask > 0):
        vals = enhanced[roi & ~disc]
        if vals.size < 32:
            return np.zeros(enhanced.shape, dtype=np.uint8)
        thr = float(np.percentile(vals, 30))
        seed_mask = np.where((enhanced <= thr) & roi, 255, 0).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, k, iterations=1)
    seed_mask = np.where(disc, 0, seed_mask).astype(np.uint8)
    seed_mask = np.where(roi, seed_mask, 0).astype(np.uint8)
    min_px = px_from_mm2(min_lesion_mm2, scale_um_px)
    return remove_small_regions(seed_mask, min_size=min_px)


def _keep_nearest_components(
    mask: np.ndarray,
    fovea_xy: tuple[float, float],
    limit: int = MAX_AUTO_COMPONENTS,
) -> np.ndarray:
    labeled = label(mask > 0)
    props = regionprops(labeled)
    if len(props) <= limit:
        return mask
    fx, fy = float(fovea_xy[0]), float(fovea_xy[1])
    ranked = sorted(
        props,
        key=lambda r: (r.centroid[1] - fx) ** 2 + (r.centroid[0] - fy) ** 2,
    )
    out = np.zeros_like(mask)
    for r in ranked[:limit]:
        out[labeled == r.label] = 255
    return out


def drop_border_touching(mask: np.ndarray, margin: int = 8) -> np.ndarray:
    """画像の端に接する塊は萎縮にしない。瞼の影や周辺減光が縁から入る。"""
    labeled = label(mask > 0)
    if labeled.max() == 0:
        return mask
    h, w = mask.shape[:2]
    out = mask.copy()
    m = max(0, int(margin))
    for region in regionprops(labeled):
        min_r, min_c, max_r, max_c = region.bbox
        if min_r <= m or min_c <= m or max_r >= h - m or max_c >= w - m:
            out[labeled == region.label] = 0
    return out


def scale_disagreement_note(
    dfd_scale: float | None,
    rim_scale: float | None,
    *,
    max_ratio: float = 1.4,
) -> str:
    """乳頭径と DFD の µm/px が離れすぎていれば、確認を促す文。一致なら空。"""
    if not dfd_scale or not rim_scale or dfd_scale <= 0 or rim_scale <= 0:
        return ""
    ratio = max(dfd_scale, rim_scale) / min(dfd_scale, rim_scale)
    if ratio < max_ratio:
        return ""
    return (
        f"乳頭の大きさでは {rim_scale:.1f} µm/px、"
        f"乳頭–中心窩では {dfd_scale:.1f} µm/px です。"
        "ずれが大きいので、中心窩の位置を確認してください。"
    )


# 中心クロップの半径。Optos の周辺は歪みが大きく、全画面で合わせると
# 相似変換が周辺に引っ張られて中心がずれる。乳頭(4.76mm)と血管アーケードが入る 8mm。
REGISTRATION_CROP_RADIUS_MM = 8.0


def register_fovea_from_prior(
    prior_gray: np.ndarray,
    prior_fovea_xy: tuple[float, float],
    gray: np.ndarray,
    *,
    scale_um_px: float | None = None,
    center_hint: tuple[float, float] | None = None,
    min_inliers: int = 12,
) -> tuple[tuple[float, float], float] | None:
    """前時点の確定済み中心窩を今の画像へ写し、(中心窩, 倍率) を返す。

    乳頭ベクトルからの提案は撮影間の回転（±数度 ≒ 数百 µm）を無視するので、
    回転込みの相似変換（AKAZE + RANSAC）で写す。まず全画面で合わせ、
    scale_um_px があれば写した中心窩の周り 8mm だけで合わせ直す（周辺の歪み対策）。
    全画面が失敗したら center_hint（手動クリック）を中心に中心だけで再試行する。
    倍率 zoom は「前画像の 1px が今の画像で zoom px」。µm/px は前時点の値 ÷ zoom。
    信用できないときは None を返し、呼び出し側がベクトル提案に戻す。
    """
    detector = cv2.AKAZE_create()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    kp_prior, desc_prior = detector.detectAndCompute(clahe.apply(prior_gray), None)
    kp_now, desc_now = detector.detectAndCompute(clahe.apply(gray), None)
    if desc_prior is None or desc_now is None:
        return None
    if len(kp_prior) < min_inliers or len(kp_now) < min_inliers:
        return None

    def _fit(idx_prior: list[int], idx_now: list[int]):
        if len(idx_prior) < min_inliers or len(idx_now) < min_inliers:
            return None
        pts_prior = [kp_prior[i] for i in idx_prior]
        pts_now = [kp_now[i] for i in idx_now]
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(desc_prior[idx_prior], desc_now[idx_now], k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < min_inliers:
            return None
        src = np.float32([pts_prior[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([pts_now[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        warp, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        )
        if warp is None or inliers is None or int(inliers.sum()) < min_inliers:
            return None
        zoom = float(np.hypot(warp[0, 0], warp[1, 0]))
        rotation = abs(float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))))
        if not (0.8 <= zoom <= 1.25) or rotation > 15.0:
            return None
        return warp, zoom

    everything_prior = list(range(len(kp_prior)))
    everything_now = list(range(len(kp_now)))
    fit = _fit(everything_prior, everything_now)
    if fit is not None:
        x, y = fit[0] @ np.array([float(prior_fovea_xy[0]), float(prior_fovea_xy[1]), 1.0])
        center_now: tuple[float, float] | None = (float(x), float(y))
    elif center_hint is not None:
        center_now = (float(center_hint[0]), float(center_hint[1]))
    else:
        return None

    if scale_um_px and scale_um_px > 0:
        radius_px = REGISTRATION_CROP_RADIUS_MM * 1000.0 / float(scale_um_px)

        def _near(points, center) -> list[int]:
            cx, cy = float(center[0]), float(center[1])
            return [
                i for i, kp in enumerate(points)
                if np.hypot(kp.pt[0] - cx, kp.pt[1] - cy) <= radius_px
            ]

        refined = _fit(_near(kp_prior, prior_fovea_xy), _near(kp_now, center_now))
        if refined is not None:
            fit = refined
    if fit is None:
        return None
    warp, zoom = fit
    x, y = warp @ np.array([float(prior_fovea_xy[0]), float(prior_fovea_xy[1]), 1.0])
    h, w = gray.shape[:2]
    if not (0.0 <= x < w and 0.0 <= y < h):
        return None
    return (float(x), float(y)), zoom


def robust_foveal_gap_um(
    mask: np.ndarray,
    fovea_xy: tuple[float, float],
    scale_um_px: float,
    *,
    open_um: float = 50.0,
) -> float | None:
    """細い突起を落としてから、中心窩から萎縮縁までの距離（µm）。

    1 画素の凹凸で隙間が潰れるのを避ける。病巣そのものは残す。
    中心窩がマスクの中なら 0。病変がなければ None。
    """
    if scale_um_px <= 0 or mask.size == 0 or not np.any(mask):
        return None
    kpx = max(1, int(round(open_um / float(scale_um_px))))
    if kpx % 2 == 0:
        kpx += 1
    if kpx > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kpx, kpx))
        kept = cv2.morphologyEx((mask > 0).astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    else:
        kept = (mask > 0).astype(np.uint8) * 255
    if not np.any(kept):
        return None
    h, w = kept.shape[:2]
    x = int(np.clip(round(fovea_xy[0]), 0, w - 1))
    y = int(np.clip(round(fovea_xy[1]), 0, h - 1))
    if kept[y, x]:
        return 0.0
    inv = np.where(kept > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    return float(dist[y, x]) * float(scale_um_px)


def intact_within_mm2(
    mask: np.ndarray,
    fovea_xy: tuple[float, float],
    scale_um_px: float,
    *,
    radius_mm: float = 1.0,
) -> float:
    """中心窩から radius_mm 以内で、萎縮でない面積（mm²）。"""
    circle = central_roi_mask(mask.shape, fovea_xy, scale_um_px, radius_mm)
    intact = circle & (mask == 0)
    return float(np.sum(intact)) * (float(scale_um_px) ** 2) / 1e6


def growth_image(
    gray: np.ndarray,
    disc_xy: tuple[float, float],
    scale_um_px: float,
) -> np.ndarray:
    """縁までの成長に使う輝度。初期マスクと同じ乳頭埋めと照明平坦化。"""
    return flatten_illumination(_fill_disc(gray, disc_xy, scale_um_px), scale_um_px)


def grow_to_edge(
    gray: np.ndarray,
    mask: np.ndarray,
    seed: tuple[int, int],
    scale_um_px: float,
    *,
    a: float = 2.2,
) -> tuple[np.ndarray, int]:
    """1 回で低蛍光の縁まで伸ばす。長押しのパワー上昇はしない。"""
    radius_px = px_from_mm(AUTO_GROW_RADIUS_MM, scale_um_px)
    max_delta = px_from_mm2(AUTO_GROW_MAX_MM2, scale_um_px)
    barrier = vessel_barrier(gray)
    return expand_segmentation_from_mask(
        gray,
        mask,
        seed,
        a,
        radius_px=radius_px,
        max_delta_px=max_delta,
        mode=GROW_MODE_STANDARD,
        barrier=barrier,
    )


def similarity_matrix(
    src_a: tuple[float, float],
    src_b: tuple[float, float],
    dst_a: tuple[float, float],
    dst_b: tuple[float, float],
) -> np.ndarray:
    """2点を2点へ写す相似変換（移動・回転・倍率）。点が近すぎるときは移動だけ。"""
    ax, ay = float(src_a[0]), float(src_a[1])
    bx, by = float(src_b[0]), float(src_b[1])
    cx, cy = float(dst_a[0]), float(dst_a[1])
    dx, dy = float(dst_b[0]), float(dst_b[1])
    vx, vy = bx - ax, by - ay
    wx, wy = dx - cx, dy - cy
    src_len = math.hypot(vx, vy)
    dst_len = math.hypot(wx, wy)
    if src_len < 1.0 or dst_len < 1.0:
        return np.array([[1.0, 0.0, cx - ax], [0.0, 1.0, cy - ay]], dtype=np.float32)
    scale = dst_len / src_len
    angle = math.atan2(wy, wx) - math.atan2(vy, vx)
    cos_s = math.cos(angle) * scale
    sin_s = math.sin(angle) * scale
    return np.array([
        [cos_s, -sin_s, cx - cos_s * ax + sin_s * ay],
        [sin_s, cos_s, cy - sin_s * ax - cos_s * ay],
    ], dtype=np.float32)


def _compose_affine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left = np.vstack([np.asarray(first, dtype=np.float64), [0.0, 0.0, 1.0]])
    right = np.vstack([np.asarray(second, dtype=np.float64), [0.0, 0.0, 1.0]])
    return (right @ left)[:2].astype(np.float32)


def _vessel_feature(gray: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    return cv2.GaussianBlur(hat.astype(np.float32), (0, 0), 1.2)


def refine_translation_ecc(warped_prior: np.ndarray, current: np.ndarray) -> np.ndarray:
    """相似変換のあと、血管像の平行移動だけを足す。失敗したら動かさない。"""
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if warped_prior.shape[:2] != current.shape[:2] or warped_prior.size < 64:
        return identity
    template = _vessel_feature(current)
    moving = _vessel_feature(warped_prior)
    if float(template.max()) < 1.0 or float(moving.max()) < 1.0:
        return identity
    warp = identity.copy()
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
    try:
        found = cv2.findTransformECC(
            template, moving, warp, cv2.MOTION_TRANSLATION, criteria,
        )
    except cv2.error:
        return identity
    if isinstance(found, tuple) and len(found) > 1 and found[1] is not None:
        warp = np.asarray(found[1], dtype=np.float32)
    # この OpenCV が返す平行移動は、今の画像を前の画像へ戻す向き。
    # warpAffine で前の画像を今の画像へ載せるには、符号を反転する。
    warp = warp.copy()
    warp[0, 2] *= -1.0
    warp[1, 2] *= -1.0
    if abs(float(warp[0, 2])) > 40.0 or abs(float(warp[1, 2])) > 40.0:
        return identity
    return warp


def translate_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """マスクを画像座標で (dx, dy) 平行移動する。はみ出した画素は捨てる。"""
    h, w = mask.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    if mask.size == 0:
        return out
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        out[mask > 0] = 255
        return out
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    warped = cv2.warpAffine(
        (mask > 0).astype(np.uint8) * 255,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    out[warped > 0] = 255
    return out


def align_prior_mask(
    mask: np.ndarray,
    prior_disc_xy: tuple[float, float],
    prior_fovea_xy: tuple[float, float],
    disc_xy: tuple[float, float],
    fovea_xy: tuple[float, float],
    shape: tuple[int, int],
    *,
    prior_gray: np.ndarray | None = None,
    gray: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """前の時点のマスクを、乳頭と中心窩の相似変換で今の画像へ合わせる。

    両方の画像があれば、合わせた血管像で平行移動を微調整する。
    戻り値は (マスク, 今の画像の大きさに合わせた前の画像)。
    """
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    if mask.size == 0:
        return out, None
    matrix = similarity_matrix(prior_disc_xy, prior_fovea_xy, disc_xy, fovea_xy)
    preview = None
    if prior_gray is not None and gray is not None:
        rough = cv2.warpAffine(
            prior_gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0,
        )
        matrix = _compose_affine(matrix, refine_translation_ecc(rough, gray))
    if prior_gray is not None:
        preview = cv2.warpAffine(
            prior_gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0,
        )
    warped = cv2.warpAffine(
        (mask > 0).astype(np.uint8) * 255,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    out[warped > 0] = 255
    return out, preview


def alignment_overlay(current: np.ndarray, warped_prior: np.ndarray) -> np.ndarray:
    """今の画像を赤と青、合わせた前の画像を緑に置いて、ずれを色で見せる。"""
    h, w = current.shape[:2]
    prior = warped_prior
    if prior.shape[:2] != (h, w):
        prior = cv2.resize(prior, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = current
    rgb[..., 1] = prior
    rgb[..., 2] = current
    return rgb


def apply_contrast_gain(gray: np.ndarray, gain: float) -> np.ndarray:
    """中央値を保ったままコントラストだけを、基準時点の倍率で伸縮する。"""
    mid = float(np.median(gray))
    out = mid + (gray.astype(np.float32) - mid) * float(gain)
    return np.clip(out, 0, 255).astype(np.uint8)


def baseline_appearance(
    gray: np.ndarray,
    fovea_xy: tuple[float, float],
    disc_xy: tuple[float, float],
    scale_um_px: float,
) -> tuple[float, float]:
    """基準時点のコントラスト倍率と、ROI 内での低蛍光パーセンタイル。

    しきい値の絶対値は持たない。後の時点は、その画像の ROI 分布の同じ割合を使う。
    """
    flat = growth_image(gray, disc_xy, scale_um_px)
    roi = central_roi_mask(flat.shape, fovea_xy, scale_um_px, QUANT_RADIUS_MM)
    vals = flat[roi]
    if vals.size < 32:
        return 1.0, 20.0
    spread = float(np.percentile(vals, 95) - np.percentile(vals, 5))
    gain = 1.0 if spread < 1.0 else float(np.clip(80.0 / spread, 0.5, 3.0))
    adjusted = apply_contrast_gain(flat, gain)
    kept = adjusted[roi]
    med = float(np.median(kept))
    below = kept[kept < med]
    if below.size < 16:
        thr = med * 0.8
    else:
        thr = min(float(np.percentile(below, 40)), med - 5.0)
    percentile = float(np.mean(kept <= thr) * 100.0)
    return gain, float(np.clip(percentile, 1.0, 45.0))


def drop_small_lesions(mask: np.ndarray, scale_um_px: float, min_mm2: float) -> np.ndarray:
    """min_mm2 未満の塊は病変に数えない。"""
    if min_mm2 <= 0 or not np.any(mask):
        return mask
    return remove_small_regions(mask, min_size=px_from_mm2(min_mm2, scale_um_px))


def lid_lash_mask(shape: tuple[int, int], *, fraction: float = 0.08) -> np.ndarray:
    """上下の瞼と左右のまつ毛が入りやすい帯。"""
    h, w = shape
    out = np.zeros(shape, dtype=bool)
    band = max(1, int(round(h * fraction)))
    side = max(1, int(round(w * 0.04)))
    out[:band, :] = True
    out[h - band:, :] = True
    out[:, :side] = True
    out[:, w - side:] = True
    return out


def peripapillary_mask(
    shape: tuple[int, int],
    disc_xy: tuple[float, float],
    scale_um_px: float,
    *,
    extra_mm: float = 0.5,
) -> np.ndarray:
    """乳頭の縁から extra_mm まで。乳頭そのものは別の円で除く。"""
    outer = disc_circle_mask(shape, disc_xy, scale_um_px, DISC_RADIUS_MM + extra_mm)
    inner = disc_circle_mask(shape, disc_xy, scale_um_px, DISC_RADIUS_MM)
    return outer & ~inner


def reading_exclusion_mask(
    gray: np.ndarray,
    disc_xy: tuple[float, float],
    scale_um_px: float,
) -> np.ndarray:
    """血管、乳頭周囲、まつ毛と瞼。True の画素は病変にしない。"""
    excluded = np.zeros(gray.shape[:2], dtype=bool)
    excluded |= vessel_barrier(gray) > 0
    excluded |= peripapillary_mask(gray.shape[:2], disc_xy, scale_um_px)
    excluded |= lid_lash_mask(gray.shape[:2])
    excluded |= disc_circle_mask(gray.shape[:2], disc_xy, scale_um_px)
    return excluded


def apply_reading_exclusions(
    mask: np.ndarray,
    gray: np.ndarray,
    disc_xy: tuple[float, float],
    scale_um_px: float,
) -> np.ndarray:
    out = mask.copy()
    out[reading_exclusion_mask(gray, disc_xy, scale_um_px)] = 0
    return out


def _cores_at_threshold(
    enhanced: np.ndarray,
    fovea_xy: tuple[float, float],
    scale_um_px: float,
    disc: np.ndarray,
    hypo_percentile: float,
    min_lesion_mm2: float,
) -> np.ndarray:
    """hypo_percentile は、この画像の ROI の中での割合。絶対輝度ではない。"""
    roi = _macula_roi_mask(enhanced.shape, fovea_xy, scale_um_px, QUANT_RADIUS_MM)
    usable = roi & ~disc
    vals = enhanced[usable]
    n_before = int(vals.size)
    if vals.size < 16:
        logger.info("しきい値前 %d px、しきい値後 0 px（ROI が狭い）", n_before)
        return np.zeros(enhanced.shape, dtype=np.uint8)
    level = float(np.percentile(vals, float(hypo_percentile)))
    seed = np.where((enhanced <= level) & usable, 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, k, iterations=1)
    kept = remove_small_regions(seed, min_size=px_from_mm2(min_lesion_mm2, scale_um_px))
    logger.info(
        "しきい値前 %d px、しきい値後 %d px（この画像の %.1f%% = %.1f）",
        n_before,
        int(np.count_nonzero(kept)),
        float(hypo_percentile),
        level,
    )
    return kept


def segment_optos(
    gray: np.ndarray,
    fovea_xy: tuple[float, float],
    disc_xy: tuple[float, float],
    scale_um_px: float,
    *,
    min_lesion_mm2: float = MIN_COMPONENT_MM2,
    contrast_gain: float | None = None,
    hypo_percentile: float | None = None,
    exclude_artifacts: bool = False,
) -> np.ndarray:
    """乳頭を除いた多巣マスク。コアから暗部の縁まで伸ばす。"""
    flat = growth_image(gray, disc_xy, scale_um_px)
    if contrast_gain is not None:
        flat = apply_contrast_gain(flat, contrast_gain)
    enhanced = _enhance(flat, use_clahe=contrast_gain is None, clahe_clip=None)
    disc = disc_circle_mask(enhanced.shape, disc_xy, scale_um_px)
    if hypo_percentile is None:
        cores = _cores(enhanced, fovea_xy, scale_um_px, disc, min_lesion_mm2)
    else:
        cores = _cores_at_threshold(
            enhanced, fovea_xy, scale_um_px, disc, hypo_percentile, min_lesion_mm2,
        )
    cores = drop_border_touching(_keep_nearest_components(cores, fovea_xy))
    barrier = vessel_barrier(flat)
    radius_px = px_from_mm(AUTO_GROW_RADIUS_MM, scale_um_px)
    max_delta = px_from_mm2(AUTO_GROW_MAX_MM2, scale_um_px)
    out = cores.copy()
    labeled = label(cores > 0)
    for r in regionprops(labeled):
        cy, cx = r.centroid
        seed = (int(round(cx)), int(round(cy)))
        out, _added = expand_segmentation_from_mask(
            flat,
            out,
            seed,
            2.2,
            radius_px=radius_px,
            max_delta_px=max_delta,
            mode=GROW_MODE_STANDARD,
            barrier=barrier,
        )
    kpx = max(3, int(round(MORPH_UM / scale_um_px)))
    if kpx % 2 == 0:
        kpx += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kpx, kpx))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    out[disc] = 0
    roi = _macula_roi_mask(out.shape, fovea_xy, scale_um_px, WINDOW_RADIUS_MM)
    out = np.where(roi, out, 0).astype(np.uint8)
    out = drop_border_touching(out)
    out = drop_small_lesions(out, scale_um_px, min_lesion_mm2)
    if exclude_artifacts:
        out = apply_reading_exclusions(out, gray, disc_xy, scale_um_px)
    return out


def paint_brush(
    mask: np.ndarray,
    xy: tuple[float, float],
    scale_um_px: float,
    *,
    erase: bool,
    radius_mm: float = BRUSH_RADIUS_MM,
) -> np.ndarray:
    out = mask.copy()
    radius = px_from_mm(radius_mm, scale_um_px)
    color = 0 if erase else 255
    cv2.circle(out, (int(round(xy[0])), int(round(xy[1]))), radius, color, thickness=-1)
    return out


# ビューワーの既定に合わせた球体。網膜面の半径はその半分。
GLOBE_DIAMETER_MM = 24.0


def area_mm2_on_24mm_globe(
    mask: np.ndarray,
    fovea_xy: tuple[float, float],
    scale_um_px: float,
    *,
    globe_diameter_mm: float = GLOBE_DIAMETER_MM,
) -> float | None:
    """中心窩を投影中心とした stereographic 投影で、球面上の面積を足す。

    接平面の mm/px は、すでに決めた µm/px を使う。
    中心窩を少し動かしても倍率は変わらず、
    各画素の球面面積だけが中心からの距離で 1 / (1 + (ρ/2R)²)² になる。R は 12 mm。
    """
    radius = float(globe_diameter_mm) / 2.0
    if scale_um_px <= 0 or radius <= 0.0:
        return None
    mm_per_px = float(scale_um_px) / 1000.0
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        return 0.0
    rho = np.hypot(xs - float(fovea_xy[0]), ys - float(fovea_xy[1])) * mm_per_px
    weight = 1.0 / (1.0 + (rho / (2.0 * radius)) ** 2) ** 2
    return float(np.sum(weight) * (mm_per_px ** 2))


def stats_central(
    mask: np.ndarray,
    fovea_xy: tuple[float, float],
    scale_um_px: float,
    disc_xy: tuple[float, float] | None = None,
) -> dict:
    """面積・中心窩距離は直径 6 mm（半径 3 mm）の内側だけ。GAWAIN も同じ半径。"""
    roi = central_roi_mask(mask.shape, fovea_xy, scale_um_px, QUANT_RADIUS_MM)
    clipped = np.where(roi, mask, 0).astype(np.uint8)
    stats = ga_stats(clipped, scale_um_px, fovea_xy)
    stats["outside_quant_px"] = int(np.sum((mask > 0) & ~roi))
    stats["quant_radius_mm"] = QUANT_RADIUS_MM
    globe = area_mm2_on_24mm_globe(clipped, fovea_xy, scale_um_px)
    stats["area_mm2_24mm"] = None if globe is None else round(globe, 3)
    return stats
