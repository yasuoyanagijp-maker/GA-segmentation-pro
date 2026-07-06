"""gawain.py – Geographic Atrophy Weighted-by-Acuity Index (GAWAIN).

中心窩を中心に直径6mm（半径3mm）の黄斑を60個の同心円環（各0.05mm幅）に分割し、
各環の健全網膜占有率に論文由来の重みを掛け合わせて0–100の指数を算出する。

重みは Leonlevde/GAWAIN リポジトリの Excel（JAMA Ophthalmology 2025 付属）に基づく。
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

MACULA_RADIUS_MM = 3.0
NUM_ANNULI = 60
ANNULUS_WIDTH_MM = 0.05
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / 'data' / 'gawain_weights.csv'


@lru_cache(maxsize=1)
def load_annulus_weights() -> list[dict]:
    """annulus, inner_mm, outer_mm, area_mm2, weight を読み込む。"""
    rows: list[dict] = []
    with WEIGHTS_PATH.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({
                'annulus': int(row['annulus']),
                'inner_mm': float(row['inner_mm']),
                'outer_mm': float(row['outer_mm']),
                'area_mm2': float(row['area_mm2']),
                'weight': float(row['weight']),
            })
    if len(rows) != NUM_ANNULI:
        raise ValueError(f'GAWAIN weights: expected {NUM_ANNULI} annuli, got {len(rows)}')
    return rows


def _mm_to_px(mm: float, scale_um_px: float) -> float:
    return mm * 1000.0 / scale_um_px


def macula_radius_px(scale_um_px: float) -> float:
    """黄斑解析半径（3mm）をピクセル換算する。"""
    return _mm_to_px(MACULA_RADIUS_MM, scale_um_px)


def compute_gawain(mask: np.ndarray,
                   fovea_xy: tuple[float, float],
                   scale_um_px: float,
                   *,
                   return_annuli: bool = False) -> dict:
    """GA二値マスクと中心窩位置から GAWAIN 指数を算出する。

    Args:
        mask: GA領域が255のuint8二値マスク。
        fovea_xy: 中心窩座標 (x=col, y=row)。
        scale_um_px: 較正済み μm/px。
        return_annuli: True なら環ごとの詳細も返す。

    Returns:
        {'gawain': float, 'annuli': list[dict] | None}
    """
    if scale_um_px <= 0:
        raise ValueError('scale_um_px must be positive')

    h, w = mask.shape
    fx, fy = fovea_xy
    yy, xx = np.ogrid[:h, :w]
    dist_px = np.sqrt((xx - fx) ** 2 + (yy - fy) ** 2)
    dist_mm = dist_px * scale_um_px / 1000.0

    ga = (mask > 0).astype(np.float32)
    macula = dist_mm <= MACULA_RADIUS_MM

    total = 0.0
    annuli_out: list[dict] = []

    for spec in load_annulus_weights():
        ring = macula & (dist_mm >= spec['inner_mm']) & (dist_mm < spec['outer_mm'])
        n_ring = int(ring.sum())
        if n_ring == 0:
            ga_fraction = 0.0
        else:
            ga_fraction = float(ga[ring].mean())

        healthy_pct = 100.0 * (1.0 - ga_fraction)
        contribution = healthy_pct * spec['area_mm2'] * spec['weight'] / 100.0
        total += contribution

        if return_annuli:
            annuli_out.append({
                'annulus': spec['annulus'],
                'inner_mm': spec['inner_mm'],
                'outer_mm': spec['outer_mm'],
                'ga_fraction': round(ga_fraction, 4),
                'healthy_pct': round(healthy_pct, 2),
                'contribution': round(contribution, 4),
            })

    return {
        'gawain': round(total, 2),
        'annuli': annuli_out if return_annuli else None,
    }
