"""stats.py – GA面積・中心窩距離などの計測値算出。"""

import math

import numpy as np
from skimage.measure import label, regionprops


def _lesion_distance(fovea_xy: tuple[float, float],
                     ga_pts: np.ndarray,
                     scale_um_px: float) -> tuple[float | None, tuple[int, int] | None]:
    if len(ga_pts) == 0:
        return None, None
    fx, fy = fovea_xy
    dists = np.sqrt((ga_pts[:, 1] - fx) ** 2 + (ga_pts[:, 0] - fy) ** 2)
    idx = dists.argmin()
    nr, nc = ga_pts[idx]
    return float(dists[idx] * scale_um_px), (int(nc), int(nr))


def ga_lesion_stats(mask: np.ndarray, scale_um_px: float,
                    fovea_xy: tuple[float, float] | None) -> list[dict]:
    """各連結GA病変ごとの面積・中心窩距離を返す。"""
    labeled = label(mask > 0)
    lesions: list[dict] = []

    for idx, region in enumerate(regionprops(labeled), start=1):
        area_px = int(region.area)
        area_mm2 = area_px * (scale_um_px ** 2) / 1e6
        cy, cx = region.centroid

        dist_um = None
        nearest_xy = None
        if fovea_xy is not None:
            ga_pts = np.argwhere(labeled == region.label)
            dist_um, nearest_xy = _lesion_distance(fovea_xy, ga_pts, scale_um_px)

        lesions.append({
            'id': idx,
            'area_px': area_px,
            'area_mm2': round(area_mm2, 3),
            'centroid_xy': (round(float(cx), 1), round(float(cy), 1)),
            'dist_um': round(dist_um, 1) if dist_um is not None else None,
            'nearest_xy': nearest_xy,
        })

    return lesions


def ga_stats(mask: np.ndarray, scale_um_px: float,
             fovea_xy: tuple[float, float] | None) -> dict:
    """マスク全体と病変別の面積・fovea距離統計を返す。"""
    lesions = ga_lesion_stats(mask, scale_um_px, fovea_xy)

    area_px = int((mask > 0).sum())
    area_mm2 = area_px * (scale_um_px ** 2) / 1e6

    dist_um = None
    nearest_xy = None
    if lesions:
        with_dist = [l for l in lesions if l['dist_um'] is not None]
        if with_dist:
            closest = min(with_dist, key=lambda l: l['dist_um'])
            dist_um = closest['dist_um']
            nearest_xy = closest['nearest_xy']
    elif fovea_xy is not None:
        ga_pts = np.argwhere(mask > 0)
        dist_um, nearest_xy = _lesion_distance(fovea_xy, ga_pts, scale_um_px)
        if dist_um is not None:
            dist_um = round(dist_um, 1)

    return {
        'area_px': area_px,
        'area_mm2': round(area_mm2, 3),
        'area_mm_sqrt': round(math.sqrt(area_mm2), 2) if area_mm2 > 0 else 0.0,
        'dist_um': dist_um,
        'nearest_xy': nearest_xy,
        'lesions': lesions,
        'lesion_count': len(lesions),
        'gawain': _compute_gawain_score(mask, scale_um_px, fovea_xy),
    }


def _compute_gawain_score(mask: np.ndarray, scale_um_px: float,
                          fovea_xy: tuple[float, float] | None) -> float | None:
    if fovea_xy is None or scale_um_px <= 0:
        return None
    from engine.gawain import compute_gawain
    return compute_gawain(mask, fovea_xy, scale_um_px)['gawain']
