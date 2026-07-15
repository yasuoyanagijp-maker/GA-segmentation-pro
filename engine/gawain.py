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
GAWAIN_FULL_NAME = "Geographic Atrophy Weighted-by-Acuity Index (GAWAIN)"
GAWAIN_VALUE_RANGE = "[0–100]"
GAWAIN_LABEL = f"{GAWAIN_FULL_NAME} {GAWAIN_VALUE_RANGE}"
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / 'data' / 'gawain_weights.csv'


def format_gawain_display(gawain: float | None) -> str:
    """GAWAIN 値とレンジを常に [] 付きで返す。"""
    if gawain is None:
        return f"— {GAWAIN_VALUE_RANGE}"
    return f"{gawain:.1f} {GAWAIN_VALUE_RANGE}"


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

    黄斑（半径3mm）のバウンディングボックスに切り出し、環インデックスを
    1パスで集計する（環ごとに全画面マスクを作る旧実装と同値・高速）。

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

    specs = load_annulus_weights()
    inner = np.array([s['inner_mm'] for s in specs], dtype=np.float64)
    outer = np.array([s['outer_mm'] for s in specs], dtype=np.float64)

    h, w = mask.shape
    fx, fy = fovea_xy
    r_px = macula_radius_px(scale_um_px)
    x0 = max(0, int(np.floor(fx - r_px)))
    x1 = min(w, int(np.ceil(fx + r_px)) + 1)
    y0 = max(0, int(np.floor(fy - r_px)))
    y1 = min(h, int(np.ceil(fy + r_px)) + 1)

    n_ring = np.zeros(NUM_ANNULI, dtype=np.int64)
    ga_ring = np.zeros(NUM_ANNULI, dtype=np.float64)
    if x0 < x1 and y0 < y1:
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist_mm = (
            np.sqrt((xx - fx) ** 2 + (yy - fy) ** 2) * scale_um_px / 1000.0
        )
        d = dist_mm.ravel()
        ga = (mask[y0:y1, x0:x1].ravel() > 0)
        # 環 i の条件: dist <= 3mm（黄斑内）かつ inner_i <= dist < outer_i
        idx = np.searchsorted(outer, d, side='right')
        ok = (idx < NUM_ANNULI) & (d <= MACULA_RADIUS_MM)
        idx_ok = idx[ok]
        keep = d[ok] >= inner[idx_ok]
        idx_ok = idx_ok[keep]
        ga_ok = ga[ok][keep]
        n_ring = np.bincount(idx_ok, minlength=NUM_ANNULI)
        ga_ring = np.bincount(
            idx_ok, weights=ga_ok.astype(np.float64), minlength=NUM_ANNULI,
        )

    total = 0.0
    annuli_out: list[dict] = []
    for i, spec in enumerate(specs):
        if n_ring[i] == 0:
            ga_fraction = 0.0
        else:
            ga_fraction = float(ga_ring[i] / n_ring[i])
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
