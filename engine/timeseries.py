"""timeseries.py – 時系列画像の日付管理・解析・進行速度推定。"""

from __future__ import annotations

import io
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from engine.stats import ga_stats


_DATE_PATTERNS = (
    re.compile(r'(20\d{2})[-_./]?(\d{2})[-_./]?(\d{2})'),
    re.compile(r'(20\d{2})(\d{2})(\d{2})'),
)

_EXIF_DATETIME_FORMATS = (
    '%Y:%m:%d %H:%M:%S',
    '%Y:%m:%d',
)

_PLOT_EPOCH = datetime(1970, 1, 1)


def guess_image_datetime(path: Path) -> datetime:
    """EXIF → ファイル名 → 更新日時の順で撮影日時を推定する。"""
    path = Path(path)

    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if exif:
                raw = exif.get(36867) or exif.get(306)
                if raw:
                    text = str(raw).strip()
                    for fmt in _EXIF_DATETIME_FORMATS:
                        try:
                            return datetime.strptime(text[:19], fmt)
                        except ValueError:
                            continue
    except Exception:
        pass

    name = path.stem
    for pattern in _DATE_PATTERNS:
        match = pattern.search(name)
        if match:
            y, m, d = map(int, match.groups())
            try:
                return datetime(y, m, d)
            except ValueError:
                continue

    return datetime.fromtimestamp(path.stat().st_mtime)


def guess_image_date(path: Path) -> date:
    """EXIF → ファイル名 → 更新日時の順で撮影日を推定する。"""
    return guess_image_datetime(path).date()


def datetime_to_iso(dt: datetime) -> str:
    """ISO 文字列（時刻が 00:00:00 なら日付のみ）。"""
    if (
        dt.hour == 0 and dt.minute == 0
        and dt.second == 0 and dt.microsecond == 0
    ):
        return dt.date().isoformat()
    return dt.isoformat(sep='T', timespec='seconds')


def parse_series_datetime(value: str) -> datetime:
    """時系列行の date 文字列を datetime にパース。"""
    return datetime.fromisoformat(value)


def datetime_to_plot_x(dt: datetime, *, epoch: datetime | None = None) -> float:
    """グラフ X 軸用の連続値（epoch からの経過日数, float）。"""
    origin = epoch or _PLOT_EPOCH
    return (dt - origin).total_seconds() / 86400.0


def parse_user_datetime_input(raw: str) -> datetime:
    """UI 手入力の撮影日時文字列を datetime にパース。"""
    text = (raw or "").strip()
    if not text:
        raise ValueError("日時が空です")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return parse_series_datetime(text)


def sync_date_to_snapshots(
    per_file: dict[str, dict],
    filename: str,
    iso: str,
) -> None:
    """確定スナップショットの date を series_dates の最新値へ同期。"""
    snap = per_file.get(filename)
    if snap is not None:
        snap["date"] = iso


def refresh_series_dates_for_files(
    files: list[Path],
    filenames: set[str],
    *,
    skip_manual: set[str] | None = None,
) -> dict[str, str]:
    """対象ファイルの撮影日時を EXIF / ファイル名 / mtime から再取得。

    skip_manual に含まれるファイル名は上書きしない（手動入力を保持）。
    """
    skip = skip_manual or set()
    refreshed: dict[str, str] = {}
    for f in files:
        if f.name in filenames and f.name not in skip:
            refreshed[f.name] = datetime_to_iso(guess_image_datetime(f))
    return refreshed


def resize_to_reference(images: list[np.ndarray]) -> list[np.ndarray]:
    """先頭画像の解像度に合わせる。"""
    resized, _ = resize_to_reference_with_scales(images)
    return resized


def resize_to_reference_with_scales(
    images: list[np.ndarray],
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """先頭画像の解像度に合わせ、各画像の (scale_x, scale_y) も返す。"""
    if not images:
        return images, []
    h, w = images[0].shape[:2]
    out = [images[0]]
    scales: list[tuple[float, float]] = [(1.0, 1.0)]
    for img in images[1:]:
        ih, iw = img.shape[:2]
        if (ih, iw) == (h, w):
            out.append(img)
            scales.append((1.0, 1.0))
        else:
            import cv2
            out.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR))
            scales.append((w / iw, h / ih))
    return out, scales


def scale_landmarks(
    fovea_xy: tuple[float, float] | None,
    disc_pts: list[tuple[float, float]] | None,
    disc_xy: tuple[float, float] | None,
    scale_x: float,
    scale_y: float,
) -> dict:
    """画像リサイズに合わせて固定点座標をスケールする。"""
    fovea = None
    if fovea_xy is not None:
        fovea = (fovea_xy[0] * scale_x, fovea_xy[1] * scale_y)
    pts = None
    if disc_pts and len(disc_pts) == 2:
        pts = [(p[0] * scale_x, p[1] * scale_y) for p in disc_pts]
    dxy = None
    if disc_xy is not None:
        dxy = (disc_xy[0] * scale_x, disc_xy[1] * scale_y)
    elif pts:
        dxy = disc_xy_from_pts(pts)
    return {"fovea_xy": fovea, "disc_pts": pts, "disc_xy": dxy}


def _slope_per_year_from_results(
    results: list[dict],
    getter,
) -> float | None:
    """基準日（先頭時点）からの経過年数に対する線形回帰の傾きを返す。"""
    if len(results) < 2:
        return None

    d0 = parse_series_datetime(results[0]['date'])
    pts: list[tuple[float, float]] = []
    for row in results:
        y = getter(row)
        if y is None:
            continue
        d1 = parse_series_datetime(row['date'])
        years = (d1 - d0).total_seconds() / (86400.0 * 365.25)
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


def _compute_row_stats(mask, scale, fovea, fallback_stats):
    if scale and fovea:
        return ga_stats(mask, scale, fovea)
    return fallback_stats


def series_metric_regressions(results: list[dict]) -> list[str]:
    """Detect implausible metric trends across chronological timepoints.

    Returns short Japanese alert strings (empty if no regression).
    Area / SQRT / GAWAIN should increase; dist_um should decrease toward fovea.
    """
    if len(results) < 2:
        return []
    # (key, label, expect_increase)
    checks = (
        ("area_mm2", "GA面積", True),
        ("area_mm_sqrt", "SQRT面積", True),
        ("gawain", "GAWAIN", True),
        ("dist_um", "中心窩距離", False),
    )
    alerts: list[str] = []
    for key, label, expect_increase in checks:
        prev = None
        for row in results:
            stats = row.get("stats") or {}
            val = stats.get(key)
            if val is None:
                continue
            v = float(val)
            if prev is not None:
                if expect_increase and v < prev - 1e-9:
                    alerts.append(
                        f"{label}が時間経過で減少しています（再解析を推奨）"
                    )
                    break
                if not expect_increase and v >= prev - 1e-9:
                    alerts.append(
                        f"{label}が時間経過で縮まっていません（再解析を推奨）"
                    )
                    break
            prev = v
    return alerts


def disc_xy_from_pts(
    disc_pts: list | tuple | None,
) -> tuple[float, float] | None:
    """乳頭計測ライン（2端点）の中央座標 — registration 固定点用。"""
    if not disc_pts or len(disc_pts) != 2:
        return None
    p1, p2 = disc_pts[0], disc_pts[1]
    return (
        (float(p1[0]) + float(p2[0])) / 2.0,
        (float(p1[1]) + float(p2[1])) / 2.0,
    )


def landmarks_from_snapshot(snap: dict) -> dict:
    """確定スナップショットから画像固有の固定点を取り出す（コピー）。"""
    fovea = snap.get("fovea_xy")
    if fovea is not None:
        fovea = (float(fovea[0]), float(fovea[1]))
    disc_pts = snap.get("disc_pts")
    if disc_pts and len(disc_pts) == 2:
        disc_pts = [(float(p[0]), float(p[1])) for p in disc_pts]
    else:
        disc_pts = None
    disc_xy = snap.get("disc_xy")
    if disc_xy is not None:
        disc_xy = (float(disc_xy[0]), float(disc_xy[1]))
    elif disc_pts:
        disc_xy = disc_xy_from_pts(disc_pts)
    else:
        disc_xy = None
    return {"fovea_xy": fovea, "disc_pts": disc_pts, "disc_xy": disc_xy}


def build_series_from_saved(
    per_file: dict[str, dict],
    files: list[Path],
    series_dates: dict[str, str],
    *,
    register: bool = True,
    selected: set[str] | None = None,
    methods: dict[str, str] | None = None,
    prior_matrices: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    """メモリ上の個別解析結果から時系列表示用データを構築（再セグメントなし）。

    - selected: 時系列対象に含めるファイル名の集合（None なら確定済み全件）。
    - methods: ファイル名 → 位置合わせ手法（redo 用。既定 "affine"）。
    - prior_matrices: ファイル名 → 既知のアフィン行列。あれば再計算せず適用する。

    先頭時点（最古の撮影日）を基準に、各画像・マスク・中心窩を同じアフィン
    行列で整列する。中心窩は必ず基準の中心窩へ一致させる（fovea anchoring）。
    面積等は基準画像の scale_um_px で全時点を再計測する（各画像固有スケールは使わない）。
    """
    from engine.series_registration import (
        default_method_for,
        register_moving_to_reference,
        warp_image,
        warp_mask,
        warp_point,
    )

    methods = methods or {}
    prior_matrices = prior_matrices or {}

    ordered = sorted(
        [
            f for f in files
            if f.name in per_file
            and (selected is None or f.name in selected)
        ],
        key=lambda f: series_dates.get(f.name, ""),
    )
    if len(ordered) < 2:
        raise ValueError("時系列対象の確定済み画像が2枚未満です")

    snapshots = [per_file[f.name] for f in ordered]
    raw_images = [s["image"] for s in snapshots]
    raw_masks = [s["mask"] for s in snapshots]
    images, img_scales = resize_to_reference_with_scales(raw_images)
    masks, _ = resize_to_reference_with_scales(raw_masks)

    ref_img = images[0]
    ref_snap = snapshots[0]
    ref_lm_raw = landmarks_from_snapshot(ref_snap)
    ref_lm = scale_landmarks(
        ref_lm_raw["fovea_xy"], ref_lm_raw["disc_pts"], ref_lm_raw["disc_xy"],
        img_scales[0][0], img_scales[0][1],
    )
    ref_fovea = ref_lm["fovea_xy"]
    ref_disc = ref_lm["disc_xy"]
    # Pixel→physical: always the reference image's scale for every timepoint.
    ref_scale = ref_snap.get("scale_um_px")
    if ref_scale is None:
        raise ValueError("基準画像に scale_um_px がありません")

    results: list[dict] = []
    for i, (snap, f) in enumerate(zip(snapshots, ordered)):
        fname = f.name
        date_str = series_dates.get(fname) or snap.get("date", "")
        lm_raw = landmarks_from_snapshot(snap)
        sx, sy = img_scales[i]
        lm = scale_landmarks(
            lm_raw["fovea_xy"], lm_raw["disc_pts"], lm_raw["disc_xy"], sx, sy,
        )
        moving_fovea = lm["fovea_xy"]
        moving_disc = lm["disc_xy"]
        source_fovea = lm_raw["fovea_xy"]
        scale = float(ref_scale)
        if i == 0 or not register:
            img = images[i]
            mask = masks[i]
            fovea = moving_fovea
            disc_xy = moving_disc
            disc_pts = lm["disc_pts"]
            matrix = np.eye(2, 3, dtype=np.float32)
            method = "reference" if i == 0 else methods.get(fname, "affine")
        elif fname in prior_matrices:
            method = methods.get(fname) or "affine"
            matrix = np.asarray(prior_matrices[fname], dtype=np.float32)
            h, w = ref_img.shape[:2]
            img = warp_image(images[i], matrix, (w, h))
            mask = warp_mask(masks[i], matrix, (w, h))
            fovea = warp_point(moving_fovea, matrix) or ref_fovea
            disc_xy = warp_point(moving_disc, matrix) or ref_disc
            disc_pts = lm["disc_pts"]
        else:
            method = methods.get(fname) or default_method_for(
                moving_fovea, moving_disc, ref_fovea, ref_disc,
            )
            reg = register_moving_to_reference(
                images[i], masks[i], ref_img,
                method=method,
                moving_fovea=moving_fovea,
                ref_fovea=ref_fovea,
                moving_disc=moving_disc,
                ref_disc=ref_disc,
                anchor_fovea=True,
            )
            img = reg["image"]
            mask = reg["mask"]
            fovea = reg["fovea_xy"] or ref_fovea
            disc_xy = reg["disc_xy"] or ref_disc
            disc_pts = lm["disc_pts"]
            matrix = reg["matrix"]

        stats = _compute_row_stats(mask, scale, fovea, snap["stats"])
        results.append({
            "index": i,
            "path": snap["path"],
            "filename": fname,
            "date": date_str,
            "image": img,
            "mask": mask,
            "stats": stats,
            "fovea_xy": fovea,
            "source_fovea_xy": source_fovea,
            "disc_pts": disc_pts,
            "disc_xy": disc_xy,
            "scale_um_px": scale,
            "reg_method": method,
            "reg_matrix": matrix,
            "reg_ok": (i == 0) or (fname in prior_matrices),
            "is_reference": (i == 0),
        })
    return results


def reregister_timepoint(
    results: list[dict],
    time_idx: int,
    per_file: dict[str, dict],
    *,
    method: str,
) -> dict:
    """既存 series の 1 時点だけを別手法で再位置合わせして更新行を返す。"""
    from engine.series_registration import register_moving_to_reference

    if not results:
        raise ValueError("時系列がありません")
    if time_idx <= 0 or time_idx >= len(results):
        raise ValueError("基準時点は再位置合わせできません")

    ref_row = results[0]
    row = results[time_idx]
    ref_snap = per_file[ref_row["filename"]]
    snap = per_file[row["filename"]]
    ref_lm_raw = landmarks_from_snapshot(ref_snap)
    lm_raw = landmarks_from_snapshot(snap)

    raw_images = [ref_snap["image"], snap["image"]]
    raw_masks = [ref_snap["mask"], snap["mask"]]
    images, img_scales = resize_to_reference_with_scales(raw_images)
    masks, _ = resize_to_reference_with_scales(raw_masks)
    ref_img = images[0]
    ref_lm = scale_landmarks(
        ref_lm_raw["fovea_xy"], ref_lm_raw["disc_pts"], ref_lm_raw["disc_xy"],
        img_scales[0][0], img_scales[0][1],
    )
    lm = scale_landmarks(
        lm_raw["fovea_xy"], lm_raw["disc_pts"], lm_raw["disc_xy"],
        img_scales[1][0], img_scales[1][1],
    )
    ref_fovea = ref_lm["fovea_xy"]
    ref_disc = ref_lm["disc_xy"]
    ref_scale = ref_snap.get("scale_um_px")
    if ref_scale is None:
        raise ValueError("基準画像に scale_um_px がありません")
    scale = float(ref_scale)

    reg = register_moving_to_reference(
        images[1], masks[1], ref_img,
        method=method,
        moving_fovea=lm["fovea_xy"],
        ref_fovea=ref_fovea,
        moving_disc=lm["disc_xy"],
        ref_disc=ref_disc,
        anchor_fovea=True,
    )
    fovea = reg["fovea_xy"] or ref_fovea
    disc_xy = reg["disc_xy"] or ref_disc
    stats = _compute_row_stats(reg["mask"], scale, fovea, snap["stats"])

    updated = dict(row)
    updated.update({
        "image": reg["image"],
        "mask": reg["mask"],
        "stats": stats,
        "fovea_xy": fovea,
        "source_fovea_xy": lm_raw["fovea_xy"],
        "disc_pts": lm["disc_pts"],
        "disc_xy": disc_xy,
        "scale_um_px": scale,
        "reg_method": method,
        "reg_matrix": reg["matrix"],
        "reg_ok": True,
    })
    return updated


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
