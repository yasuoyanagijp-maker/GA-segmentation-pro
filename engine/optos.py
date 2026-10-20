"""Optos 緑 AF 向けの初期マスクと計測。

現行の ``segment_ga`` は呼ばない。黄斑窓の中で乳頭を埋め、多巣の暗コアを
縁まで成長させ、面積は中心 6 mm だけを mm² にする。
"""

from __future__ import annotations

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

# Spectralis 約 10.5 µm/px での 7 px / 64 px を物理サイズに固定する。
SPECTRALIS_UM_PER_PX = 10.5
MORPH_UM = 7.0 * SPECTRALIS_UM_PER_PX
DISC_RADIUS_MM = 0.9
QUANT_RADIUS_MM = 6.0
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
    min_px = px_from_mm2(MIN_COMPONENT_MM2, scale_um_px)
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


def segment_optos(
    gray: np.ndarray,
    fovea_xy: tuple[float, float],
    disc_xy: tuple[float, float],
    scale_um_px: float,
) -> np.ndarray:
    """乳頭を除いた多巣マスク。コアから暗部の縁まで伸ばす。"""
    flat = growth_image(gray, disc_xy, scale_um_px)
    enhanced = _enhance(flat, use_clahe=True, clahe_clip=None)
    disc = disc_circle_mask(enhanced.shape, disc_xy, scale_um_px)
    cores = drop_border_touching(
        _keep_nearest_components(
            _cores(enhanced, fovea_xy, scale_um_px, disc),
            fovea_xy,
        )
    )
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
    return drop_border_touching(out)


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
    disc_xy: tuple[float, float],
    *,
    dfd_mm: float = DEFAULT_DFD_UM / 1000.0,
    globe_diameter_mm: float = GLOBE_DIAMETER_MM,
) -> float | None:
    """中心窩を投影中心とした stereographic 投影で、球面上の面積を足す。

    乳頭–中心窩 4.76 mm は球面上の弧とする。画像上の距離は接平面の
    ρ = 2R tan(θ/2) に対応させ、各画素の球面面積は接平面の面積に
    1 / (1 + (ρ/2R)²)² を掛けたもの。R は 12 mm。
    """
    radius = float(globe_diameter_mm) / 2.0
    d_px = math.hypot(float(disc_xy[0]) - float(fovea_xy[0]), float(disc_xy[1]) - float(fovea_xy[1]))
    if d_px < 1.0 or radius <= 0.0 or dfd_mm <= 0.0:
        return None
    theta = float(dfd_mm) / radius
    if theta >= math.pi:
        return None
    mm_per_px = (2.0 * radius * math.tan(theta / 2.0)) / d_px
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
    """面積・GAWAIN・中心窩距離は中心 6 mm の内側だけ。"""
    roi = central_roi_mask(mask.shape, fovea_xy, scale_um_px, QUANT_RADIUS_MM)
    clipped = np.where(roi, mask, 0).astype(np.uint8)
    stats = ga_stats(clipped, scale_um_px, fovea_xy)
    stats["outside_quant_px"] = int(np.sum((mask > 0) & ~roi))
    stats["quant_radius_mm"] = QUANT_RADIUS_MM
    globe = None if disc_xy is None else area_mm2_on_24mm_globe(clipped, fovea_xy, disc_xy)
    stats["area_mm2_24mm"] = None if globe is None else round(globe, 3)
    return stats
