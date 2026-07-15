"""アプリから参照されなくなったヘルパーの保管庫（詳細は archive/README.md）。

アプリ・テストからは import されない。復活させる場合は元のモジュールへ
戻すこと。現行 API（guess_image_datetime / resize_to_reference_with_scales /
generate_pdf_report_bytes / render_gawain_figure_bytes）へ委譲する薄い
ラッパーが中心のため、engine パッケージが import 可能な環境なら
そのまま動作する。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
from skimage.measure import label, regionprops

from engine.export import generate_pdf_report_bytes
from engine.gawain_figure import render_gawain_figure_bytes
from engine.timeseries import (
    guess_image_datetime,
    resize_to_reference_with_scales,
)


# --- 旧: engine/timeseries.py -------------------------------------------

def guess_image_date(path: Path) -> date:
    """EXIF → ファイル名 → 更新日時の順で撮影日を推定する（旧 date 版）。"""
    return guess_image_datetime(path).date()


def resize_to_reference(images: list[np.ndarray]) -> list[np.ndarray]:
    """先頭画像のサイズに合わせてリサイズ（旧・倍率なし版）。"""
    resized, _ = resize_to_reference_with_scales(images)
    return resized


# --- 旧: engine/export.py -----------------------------------------------

def generate_pdf_report(
    *,
    filename: str,
    date: str,
    stats: dict,
    overlay_rgb: np.ndarray,
    output_path: Path,
    gawain_figure_png: bytes | None = None,
    scale_um_px: float | None = None,
    calibration_source: str | None = None,
    field_width_mm: float | None = None,
    fovea_xy: tuple[float, float] | None = None,
    foveal_correction_applied: bool | None = None,
    foveal_correction_skipped: bool | None = None,
    foveal_correction_strength: float | None = None,
) -> Path:
    """1時点の解析結果 PDF を保存（ファイル書き出し版）。"""
    pdf_bytes = generate_pdf_report_bytes(
        filename=filename,
        date=date,
        stats=stats,
        overlay_rgb=overlay_rgb,
        gawain_figure_png=gawain_figure_png,
        scale_um_px=scale_um_px,
        calibration_source=calibration_source,
        field_width_mm=field_width_mm,
        fovea_xy=fovea_xy,
        foveal_correction_applied=foveal_correction_applied,
        foveal_correction_skipped=foveal_correction_skipped,
        foveal_correction_strength=foveal_correction_strength,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pdf_bytes)
    return output_path


# --- 旧: engine/gawain_figure.py ----------------------------------------

def split_optos_dual_panel(img: np.ndarray, eye: str = "R") -> np.ndarray:
    """Optos 左右2眼並列画像から片眼パネルを切り出す。

    eye: 'R' = 左半分, 'L' = 右半分（画像上の配置）
    """
    h, w = img.shape[:2]
    if w < h * 1.35:
        return img
    mid = w // 2
    if eye.upper() == "L":
        return img[:, mid:].copy()
    return img[:, :mid].copy()


def crop_display_margins(img: np.ndarray, margin_frac: float = 0.07) -> np.ndarray:
    """上下左右の UI テキスト余白を除去。"""
    h, w = img.shape[:2]
    dy = int(h * margin_frac)
    dx = int(w * margin_frac)
    return img[dy : h - dy, dx : w - dx].copy()


def estimate_fovea_from_mask(mask: np.ndarray) -> tuple[float, float] | None:
    """最大 GA 連結成分の重心を中心窩近似とする。"""
    labeled = label(mask > 0)
    props = regionprops(labeled)
    if not props:
        return None
    main = max(props, key=lambda r: r.area)
    cy, cx = main.centroid
    return float(cx), float(cy)


def render_gawain_figure(
    img_gray: np.ndarray,
    mask: np.ndarray,
    fovea_xy: tuple[float, float],
    scale_um_px: float,
    output_path: Path | str,
    *,
    title: str = "",
    dpi: int = 150,
) -> dict:
    """Figure 1 型（A–E）の GAWAIN 説明 PNG を出力する（ファイル書き出し版）。"""
    output_path = Path(output_path)
    png_bytes, meta = render_gawain_figure_bytes(
        img_gray, mask, fovea_xy, scale_um_px, title=title, dpi=dpi,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png_bytes)
    meta["output_path"] = str(output_path)
    return meta
