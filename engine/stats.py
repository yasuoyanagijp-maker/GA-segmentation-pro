"""stats.py – GA面積・中心窩距離などの計測値算出。"""

import numpy as np


def ga_stats(mask: np.ndarray, scale_um_px: float,
             fovea_xy: tuple[float, float] | None
             ) -> dict:
    """マスクから面積とfovea距離の統計を返す。"""
    area_px = int((mask > 0).sum())
    area_mm2 = area_px * (scale_um_px ** 2) / 1e6

    dist_um = None
    nearest_xy = None
    if fovea_xy is not None:
        ga_pts = np.argwhere(mask > 0)
        if len(ga_pts):
            fx, fy = fovea_xy          # x=col, y=row
            dists = np.sqrt((ga_pts[:, 1] - fx) ** 2 +
                            (ga_pts[:, 0] - fy) ** 2)
            idx = dists.argmin()
            dist_um = float(dists[idx] * scale_um_px)
            nr, nc = ga_pts[idx]
            nearest_xy = (int(nc), int(nr))

    return {
        'area_px':    area_px,
        'area_mm2':   round(area_mm2, 3),
        'dist_um':    round(dist_um, 1) if dist_um is not None else None,
        'nearest_xy': nearest_xy,
    }
