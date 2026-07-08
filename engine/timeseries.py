"""timeseries.py – 時系列画像の日付管理・解析・進行速度推定。"""

from __future__ import annotations

import io
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from engine import segmentation
from engine.io_utils import load_gray
from engine.registration import register_stack
from engine.stats import ga_stats


_DATE_PATTERNS = (
    re.compile(r'(20\d{2})[-_./]?(\d{2})[-_./]?(\d{2})'),
    re.compile(r'(20\d{2})(\d{2})(\d{2})'),
)


def guess_image_date(path: Path) -> date:
    """EXIF → ファイル名 → 更新日時の順で撮影日を推定する。"""
    path = Path(path)

    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if exif:
                raw = exif.get(36867) or exif.get(306)
                if raw:
                    return datetime.strptime(str(raw)[:10], '%Y:%m:%d').date()
    except Exception:
        pass

    name = path.stem
    for pattern in _DATE_PATTERNS:
        match = pattern.search(name)
        if match:
            y, m, d = map(int, match.groups())
            try:
                return date(y, m, d)
            except ValueError:
                continue

    return datetime.fromtimestamp(path.stat().st_mtime).date()


def resize_to_reference(images: list[np.ndarray]) -> list[np.ndarray]:
    """先頭画像の解像度に合わせる。"""
    if not images:
        return images
    h, w = images[0].shape[:2]
    out = [images[0]]
    for img in images[1:]:
        if img.shape[:2] == (h, w):
            out.append(img)
        else:
            import cv2
            out.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR))
    return out


def analyze_time_series(
    images: list[np.ndarray],
    paths: list[Path],
    dates: list[date],
    scale_um_px: float,
    fovea_xy: tuple[float, float],
    *,
    register: bool = True,
    progress_callback=None,
) -> list[dict]:
    """各時点でセグメンテーションと指標を算出する。"""
    if len(images) != len(paths) or len(images) != len(dates):
        raise ValueError('images, paths, dates の長さが一致しません')
    if not images:
        return []

    images = resize_to_reference(images)
    stack = np.stack(images, axis=0)

    if register and len(images) > 1:
        stack, _ = register_stack(stack, progress_callback=progress_callback)

    order = sorted(range(len(dates)), key=lambda i: dates[i])
    results: list[dict] = []

    for rank, idx in enumerate(order):
        if progress_callback:
            progress_callback(rank / len(order), f'Analyzing timepoint {rank + 1}/{len(order)}...')
        img = stack[idx]
        mask = segmentation.segment_ga(img)
        stats = ga_stats(mask, scale_um_px, fovea_xy)
        results.append({
            'index': idx,
            'path': str(paths[idx]),
            'filename': paths[idx].name,
            'date': dates[idx].isoformat(),
            'image': img,
            'mask': mask,
            'stats': stats,
        })
    return results


def _slope_per_year_from_results(
    results: list[dict],
    getter,
) -> float | None:
    """基準日（先頭時点）からの経過年数に対する線形回帰の傾きを返す。"""
    if len(results) < 2:
        return None

    d0 = datetime.fromisoformat(results[0]['date']).date()
    pts: list[tuple[float, float]] = []
    for row in results:
        y = getter(row)
        if y is None:
            continue
        d1 = datetime.fromisoformat(row['date']).date()
        years = (d1 - d0).days / 365.25
        pts.append((years, float(y)))

    if len(pts) < 2:
        return None

    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    if np.allclose(xs, xs[0]):
        return None

    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def estimate_progression_rate_um_per_year(results: list[dict]) -> float | None:
    """中心窩距離の時系列から進行速度（μm/年）を線形回帰で推定。"""
    slope = _slope_per_year_from_results(
        results, lambda r: r['stats']['dist_um'],
    )
    if slope is None:
        return None
    # 距離が縮む（進行）方向を正にする
    return round(abs(slope), 1)


def estimate_growth_mm2_per_year(results: list[dict]) -> float | None:
    """GA面積の時系列から拡大速度（mm²/年）を線形回帰で推定。"""
    slope = _slope_per_year_from_results(
        results, lambda r: r['stats']['area_mm2'],
    )
    if slope is None:
        return None
    return round(slope, 3)


def estimate_growth_mm_sqrt_per_year(results: list[dict]) -> float | None:
    """√変換GA面積の時系列から拡大速度（mm/年, SQRT scale）を線形回帰で推定。"""
    slope = _slope_per_year_from_results(
        results, lambda r: r['stats']['area_mm_sqrt'],
    )
    if slope is None:
        return None
    return round(slope, 2)


def build_series_from_saved(
    per_file: dict[str, dict],
    files: list[Path],
    series_dates: dict[str, str],
    *,
    register: bool = True,
) -> list[dict]:
    """メモリ上の個別解析結果から時系列表示用データを構築（再セグメントなし）。"""
    from engine.registration import apply_transformations_to_stack, register_stack

    ordered = sorted(
        [f for f in files if f.name in per_file],
        key=lambda f: series_dates.get(f.name, ""),
    )
    if len(ordered) < 2:
        raise ValueError("確定済み画像が2枚未満です")

    snapshots = [per_file[f.name] for f in ordered]
    images = resize_to_reference([s["image"] for s in snapshots])
    masks = resize_to_reference([s["mask"] for s in snapshots])
    img_stack = np.stack(images, axis=0)
    mask_stack = np.stack(masks, axis=0)

    if register and len(images) > 1:
        img_stack, matrices = register_stack(img_stack)
        mask_stack = apply_transformations_to_stack(mask_stack, matrices)

    results: list[dict] = []
    for i, (snap, f) in enumerate(zip(snapshots, ordered)):
        img = img_stack[i]
        mask = mask_stack[i]
        scale = snap.get("scale_um_px")
        fovea = snap.get("fovea_xy")
        if scale and fovea:
            stats = ga_stats(mask, scale, fovea)
        else:
            stats = snap["stats"]
        results.append({
            "index": i,
            "path": snap["path"],
            "filename": f.name,
            "date": snap["date"],
            "image": img,
            "mask": mask,
            "stats": stats,
            "fovea_xy": fovea,
            "scale_um_px": scale,
        })
    return results


def overlays_to_gif_bytes(overlays: list[np.ndarray], duration_ms: int = 600) -> bytes:
    """RGBオーバーレイ列からGIFバイト列を生成する。"""
    if not overlays:
        raise ValueError('overlays is empty')

    frames = [Image.fromarray(frame) for frame in overlays]
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return buf.getvalue()
