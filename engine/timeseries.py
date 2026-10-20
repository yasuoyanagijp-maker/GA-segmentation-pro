"""timeseries.py – 時系列画像の日付管理・解析・進行速度推定。"""

from __future__ import annotations

import io
import logging
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from engine.stats import ga_stats

_perf_log = logging.getLogger("ga.reg.timing")


_DATE_PATTERNS = (
    re.compile(r'(20\d{2})[-_./]?(\d{2})[-_./]?(\d{2})'),
    re.compile(r'(20\d{2})(\d{2})(\d{2})'),
)

_EXIF_DATETIME_FORMATS = (
    '%Y:%m:%d %H:%M:%S',
    '%Y:%m:%d',
)

_PLOT_EPOCH = datetime(1970, 1, 1)


def extract_image_datetime(path: Path) -> tuple[datetime | None, str]:
    """撮影日時を推定し (datetime|None, source) を返す。

    source: ``exif`` | ``filename`` | ``mtime`` | ``missing``
    EXIF / ファイル名が取れない場合は mtime を候補として返す（確定扱いしない）。
    """
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
                            return datetime.strptime(text[:19], fmt), "exif"
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
                return datetime(y, m, d), "filename"
            except ValueError:
                continue

    try:
        return datetime.fromtimestamp(path.stat().st_mtime), "mtime"
    except OSError:
        return None, "missing"


def guess_image_datetime(path: Path) -> datetime:
    """EXIF → ファイル名 → 更新日時の順で撮影日時を推定する。"""
    dt, source = extract_image_datetime(path)
    if dt is not None:
        return dt
    # フォールバック（通常は mtime で返る）
    return datetime.fromtimestamp(Path(path).stat().st_mtime)


def is_datetime_source_confirmed(source: str | None) -> bool:
    """Registration ゲート用: EXIF / ファイル名 / 手動入力は確定扱い。"""
    return source in ("exif", "filename", "manual")


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


def refresh_series_datetimes_for_files(
    files: list[Path],
    filenames: set[str],
    *,
    skip_manual: set[str] | None = None,
) -> dict[str, tuple[str, str]]:
    """対象ファイルの (iso, source) を再取得。skip_manual は上書きしない。"""
    skip = skip_manual or set()
    refreshed: dict[str, tuple[str, str]] = {}
    for f in files:
        if f.name in filenames and f.name not in skip:
            dt, source = extract_image_datetime(f)
            if dt is None:
                continue
            refreshed[f.name] = (datetime_to_iso(dt), source)
    return refreshed


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


# 緑 AF の中心窩クリックは、これより近い隙間を年数にすると誤差が年数になる。
FOVEAL_GAP_MIN_UM = 250.0


def describe_foveal_approach(
    dist_um: float | None,
    rate_um_per_year: float | None,
) -> str:
    """中心窩までの距離を、年数にしてよいかを分けて一文にする。

    隙間が FOVEAL_GAP_MIN_UM 未満のときは、位置合わせしても
    到達年数にはしない。その症例の進行は面積と中心 1 mm の残存で見る。
    """
    if dist_um is None:
        return "中心窩までの距離は測れていません"
    if dist_um < FOVEAL_GAP_MIN_UM:
        return (
            f"中心窩まで {dist_um:.0f} µm。"
            "縁に達しているので、到達年数は出しません"
        )
    if rate_um_per_year and rate_um_per_year > 0:
        years = float(dist_um) / float(rate_um_per_year)
        return (
            f"中心窩まで {dist_um:.0f} µm"
            f" · 接近速度 {rate_um_per_year} µm/年"
            f" · 最新時点から中心窩まで {years:.2f} 年"
        )
    return f"中心窩まで {dist_um:.0f} µm"


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


def estimate_intact_mm2_per_year(results: list[dict]) -> float | None:
    """中心 1 mm の残存面積の傾き（mm²/年）。減っていれば負。"""
    slope = _slope_per_year_from_results(
        results, lambda r: r['stats'].get('intact_1mm_mm2'),
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


def landmarks_ready_for_registration(snap: dict | None) -> bool:
    """Registration 前ゲート。

    乳頭と中心窩が揃っていれば可。HRA の幅指定は乳頭が写らないので、
    中心窩と正の scale_um_px（calibration_source=width）で可。
    """
    if not snap:
        return False
    lm = landmarks_from_snapshot(snap)
    if lm.get("fovea_xy") is None:
        return False
    if lm.get("disc_xy") is not None:
        return True
    if snap.get("calibration_source") != "width":
        return False
    scale = snap.get("scale_um_px")
    try:
        return scale is not None and float(scale) > 0
    except (TypeError, ValueError):
        return False


def _maybe_resegment_aligned(
    img: np.ndarray,
    warped_mask: np.ndarray,
    fovea,
    scale: float,
    *,
    enabled: bool,
    fname: str,
) -> tuple[np.ndarray, str]:
    """位置合わせ後画像で再segmentation。失敗時は warp マスクへフォールバック。"""
    if not enabled:
        return warped_mask, "warp"
    from engine.segmentation import segment_ga

    try:
        reseged = segment_ga(img, fovea_xy=fovea, scale_um_px=scale)
    except Exception as exc:
        _perf_log.warning("resegment failed for %s: %s — keep warped mask", fname, exc)
        return warped_mask, "warp_fallback_error"
    if reseged is None or not np.any(reseged):
        _perf_log.warning(
            "resegment empty for %s — keep warped mask", fname,
        )
        return warped_mask, "warp_fallback_empty"
    return reseged, "resegment"


def build_series_from_saved(
    per_file: dict[str, dict],
    files: list[Path],
    series_dates: dict[str, str],
    *,
    register: bool = True,
    selected: set[str] | None = None,
    methods: dict[str, str] | None = None,
    prior_matrices: dict[str, np.ndarray] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    resegment_after_reg: bool = True,
) -> list[dict]:
    """メモリ上の個別解析結果から時系列表示用データを構築する。

    - selected: 時系列対象に含めるファイル名の集合（None なら確定済み全件）。
    - methods: ファイル名 → 位置合わせ手法（redo 用。既定 "affine"）。
    - prior_matrices: ファイル名 → 既知のアフィン行列。あれば再計算せず適用する。
    - progress_callback: (done, total, label) — 移動側時点の進捗（UI 用）。
    - resegment_after_reg: True なら位置合わせ後の画像で GA を再segmentation
      （失敗時は warp マスクへフォールバック）。False なら従来どおりマスクを warp。

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
    # 時系列本線（reg→seg）では prep 時点で mask が無いことがある → 空マスク
    raw_masks = []
    for s, img in zip(snapshots, raw_images):
        m = s.get("mask")
        if m is None:
            raw_masks.append(np.zeros(img.shape[:2], dtype=np.uint8))
        else:
            raw_masks.append(m)
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

    # 各時点の固定点を先に解決（並列登録と組み立ての両方で使う）
    row_lms: list[tuple[dict, dict]] = []
    for i, snap in enumerate(snapshots):
        lm_raw = landmarks_from_snapshot(snap)
        sx, sy = img_scales[i]
        lm = scale_landmarks(
            lm_raw["fovea_xy"], lm_raw["disc_pts"], lm_raw["disc_xy"], sx, sy,
        )
        row_lms.append((lm_raw, lm))

    # 位置合わせが必要な時点は互いに独立（全て基準時点へ整列する）なので
    # 並列に計算する。結果は直列実行と同一（時点間に依存なし・決定的）。
    reg_methods: dict[int, str] = {}
    for i, f in enumerate(ordered):
        fname = f.name
        if i == 0 or not register or fname in prior_matrices:
            continue
        lm = row_lms[i][1]
        reg_methods[i] = methods.get(fname) or default_method_for(
            lm["fovea_xy"], lm["disc_xy"], ref_fovea, ref_disc,
        )

    total_moving = max(len(ordered) - 1, 0) if register else 0
    progress_done = {"n": 0}

    def _report_progress(label: str) -> None:
        if progress_callback is None or total_moving <= 0:
            return
        progress_done["n"] += 1
        progress_callback(progress_done["n"], total_moving, label)

    def _register_one(i: int) -> tuple[int, dict]:
        fname = ordered[i].name
        method = reg_methods[i]
        lm = row_lms[i][1]
        t0 = time.perf_counter()
        reg = register_moving_to_reference(
            images[i], masks[i], ref_img,
            method=method,
            moving_fovea=lm["fovea_xy"],
            ref_fovea=ref_fovea,
            moving_disc=lm["disc_xy"],
            ref_disc=ref_disc,
            anchor_fovea=True,
        )
        _perf_log.info(
            "timepoint %d/%d %s (method=%s): %.1f s",
            i + 1, len(ordered), fname, method,
            time.perf_counter() - t0,
        )
        return i, reg

    reg_out: dict[int, dict] = {}
    if reg_methods:
        # メモリ（1時点あたり前処理キャッシュ数十MB）を抑えるため上限4並列
        max_workers = min(len(reg_methods), 4, os.cpu_count() or 1)
        if max_workers <= 1:
            for i in sorted(reg_methods):
                reg_out[i] = _register_one(i)[1]
                _report_progress(f"計算 {ordered[i].name}")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(_register_one, i): i for i in sorted(reg_methods)
                }
                for fut in as_completed(futures):
                    i, reg = fut.result()
                    reg_out[i] = reg
                    _report_progress(f"計算 {ordered[i].name}")

    results: list[dict] = []
    for i, (snap, f) in enumerate(zip(snapshots, ordered)):
        fname = f.name
        date_str = series_dates.get(fname) or snap.get("date", "")
        lm_raw, lm = row_lms[i]
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
            mask_source = "reference" if i == 0 else "native"
        elif fname in prior_matrices:
            method = methods.get(fname) or "affine"
            matrix = np.asarray(prior_matrices[fname], dtype=np.float32)
            h, w = ref_img.shape[:2]
            img = warp_image(images[i], matrix, (w, h))
            mask = warp_mask(masks[i], matrix, (w, h))
            fovea = warp_point(moving_fovea, matrix) or ref_fovea
            disc_xy = warp_point(moving_disc, matrix) or ref_disc
            disc_pts = lm["disc_pts"]
            mask, mask_source = _maybe_resegment_aligned(
                img, mask, fovea, scale,
                enabled=resegment_after_reg, fname=fname,
            )
            _report_progress(f"適用 {fname}")
        else:
            method = reg_methods[i]
            reg = reg_out[i]
            img = reg["image"]
            mask = reg["mask"]
            fovea = reg["fovea_xy"] or ref_fovea
            disc_xy = reg["disc_xy"] or ref_disc
            disc_pts = lm["disc_pts"]
            matrix = reg["matrix"]
            mask, mask_source = _maybe_resegment_aligned(
                img, mask, fovea, scale,
                enabled=resegment_after_reg, fname=fname,
            )

        stats = _compute_row_stats(mask, scale, fovea, snap.get("stats"))
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
            "calibration_source": ref_snap.get("calibration_source"),
            "mask_source": mask_source,
            "case_id": snap.get("case_id") or ref_snap.get("case_id"),
            "reg_method": method,
            "reg_matrix": matrix,
            "reg_ok": (i == 0) or (fname in prior_matrices),
            "is_reference": (i == 0),
        })
    return results


def segment_aligned_series(
    results: list[dict],
    *,
    allow_warp_fallback: bool = True,
) -> list[dict]:
    """整列済み時系列の各時点で GA を segmentation し、同一 scale で再計測する。

    時系列本線: Registration → 整列QC → **この関数** → 測定 → 任意マスクQC。
    再seg が空/失敗で、かつ warp 済みマスクに画素がある場合のみフォールバック。
    """
    from engine.segmentation import segment_ga

    if not results:
        return results

    out: list[dict] = []
    for row in results:
        updated = dict(row)
        img = row["image"]
        fovea = row.get("fovea_xy")
        scale = row.get("scale_um_px")
        prior_mask = row.get("mask")
        mask_source = "resegment"
        mask = None
        try:
            mask = segment_ga(img, fovea_xy=fovea, scale_um_px=scale)
        except Exception as exc:
            _perf_log.warning(
                "aligned seg failed for %s: %s", row.get("filename"), exc,
            )
            mask = None
            mask_source = "warp_fallback_error"

        if mask is None or not np.any(mask):
            if (
                allow_warp_fallback
                and prior_mask is not None
                and np.any(prior_mask)
            ):
                mask = prior_mask
                if mask_source == "resegment":
                    mask_source = "warp_fallback_empty"
            else:
                h, w = img.shape[:2]
                mask = (
                    prior_mask
                    if prior_mask is not None
                    else np.zeros((h, w), dtype=np.uint8)
                )
                if mask_source == "resegment":
                    mask_source = "empty"

        if row.get("is_reference") and mask_source == "resegment":
            mask_source = "reference"

        stats = _compute_row_stats(mask, scale, fovea, row.get("stats"))
        updated["mask"] = mask
        updated["stats"] = stats
        updated["mask_source"] = mask_source
        out.append(updated)
    return out


def reregister_timepoint(
    results: list[dict],
    time_idx: int,
    per_file: dict[str, dict],
    *,
    method: str,
    resegment_after_reg: bool = True,
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
    raw_masks = []
    for s, img in zip((ref_snap, snap), raw_images):
        m = s.get("mask")
        if m is None:
            raw_masks.append(np.zeros(img.shape[:2], dtype=np.uint8))
        else:
            raw_masks.append(m)
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
    mask, mask_source = _maybe_resegment_aligned(
        reg["image"], reg["mask"], fovea, scale,
        enabled=resegment_after_reg, fname=row["filename"],
    )
    stats = _compute_row_stats(mask, scale, fovea, snap["stats"])

    updated = dict(row)
    updated.update({
        "image": reg["image"],
        "mask": mask,
        "stats": stats,
        "fovea_xy": fovea,
        "source_fovea_xy": lm_raw["fovea_xy"],
        "disc_pts": lm["disc_pts"],
        "disc_xy": disc_xy,
        "scale_um_px": scale,
        "calibration_source": ref_snap.get("calibration_source"),
        "mask_source": mask_source,
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
