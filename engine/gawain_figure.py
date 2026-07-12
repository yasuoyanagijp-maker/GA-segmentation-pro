"""gawain_figure.py – 論文 Figure 1 型の GAWAIN 説明図を生成する。"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from skimage.measure import label, regionprops

from engine.gawain import (
    MACULA_RADIUS_MM,
    NUM_ANNULI,
    compute_gawain,
    load_annulus_weights,
    macula_radius_px,
)
from engine.overlay import blend_mask_overlay


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


def draw_annuli_overlay(
    img_gray: np.ndarray,
    fovea_xy: tuple[float, float],
    scale_um_px: float,
    *,
    step: int = 5,
) -> np.ndarray:
    """中心窩＋60 同心円環（step おきに描画）の RGB。"""
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    fx, fy = int(fovea_xy[0]), int(fovea_xy[1])
    cv2.drawMarker(rgb, (fx, fy), (34, 211, 238), cv2.MARKER_CROSS, 14, 1)
    weights = load_annulus_weights()
    for spec in weights:
        r_px = int(round(spec["outer_mm"] * 1000.0 / scale_um_px))
        color = (167, 139, 250) if spec["annulus"] % step == 0 else (120, 100, 200)
        thickness = 1 if spec["annulus"] % step else 2
        cv2.circle(rgb, (fx, fy), r_px, color, thickness)
    outer_px = int(round(macula_radius_px(scale_um_px)))
    cv2.circle(rgb, (fx, fy), outer_px, (250, 204, 21), 2)
    return rgb


def render_gawain_figure_bytes(
    img_gray: np.ndarray,
    mask: np.ndarray,
    fovea_xy: tuple[float, float],
    scale_um_px: float,
    *,
    title: str = "",
    dpi: int = 150,
) -> tuple[bytes, dict]:
    """Figure 1 型（A–E）の GAWAIN 説明 PNG をバイト列で返す。"""
    gaw = compute_gawain(mask, fovea_xy, scale_um_px, return_annuli=True)
    annuli = gaw["annuli"] or []
    weights = load_annulus_weights()

    panel_a = img_gray
    panel_b = blend_mask_overlay(img_gray, mask)
    panel_c = draw_annuli_overlay(img_gray, fovea_xy, scale_um_px)

    mean_radii = [(w["inner_mm"] + w["outer_mm"]) / 2.0 for w in weights]
    w_vals = [w["weight"] for w in weights]
    ga_fracs = [a["ga_fraction"] * 100.0 for a in annuli]
    annulus_ids = [a["annulus"] for a in annuli]

    area_px = int((mask > 0).sum())
    area_mm2 = area_px * (scale_um_px ** 2) / 1e6

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.15, 0.85], hspace=0.28, wspace=0.22)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1:])

    for ax, img, label in (
        (ax_a, panel_a, "A  FAF"),
        (ax_b, panel_b, "B  GA segmentation"),
        (ax_c, panel_c, "C  60 annuli (6 mm)"),
    ):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(label, fontsize=11, fontweight="bold", loc="left")
        ax.axis("off")

    ax_d.plot(mean_radii, w_vals, color="#7c3aed", linewidth=2)
    ax_d.set_xlabel("Mean annulus radius (mm)")
    ax_d.set_ylabel("Empirical weight")
    ax_d.set_title("D  Annulus weights (JAMA / Leonlevde)", fontsize=11, fontweight="bold", loc="left")
    ax_d.grid(True, alpha=0.3)
    ax_d.set_xlim(0, MACULA_RADIUS_MM)

    colors = plt.cm.Reds(np.array(ga_fracs) / max(max(ga_fracs), 1.0))
    ax_e.bar(annulus_ids, ga_fracs, color=colors, width=0.85, edgecolor="none")
    ax_e.set_xlabel("Annulus index (1–60)")
    ax_e.set_ylabel("GA occupancy (%)")
    ax_e.set_title(
        f"E  Per-annulus GA occupancy  ·  GAWAIN = {gaw['gawain']:.1f}  "
        f"(total GA area = {area_mm2:.2f} mm²)",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    ax_e.set_xlim(0.5, NUM_ANNULI + 0.5)
    ax_e.grid(True, axis="y", alpha=0.3)

    suptitle = title or "GAWAIN — Geographic Atrophy Weighted-by-Acuity Index"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.01,
        f"Fovea ({fovea_xy[0]:.0f}, {fovea_xy[1]:.0f}) px  ·  "
        f"scale {scale_um_px:.2f} μm/px  ·  "
        f"{NUM_ANNULI} annuli × 0.05 mm within {MACULA_RADIUS_MM * 2:.0f} mm diameter",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    png_bytes = buf.getvalue()

    meta = {
        "gawain": gaw["gawain"],
        "area_mm2": round(area_mm2, 3),
        "fovea_xy": fovea_xy,
        "scale_um_px": scale_um_px,
        "annuli": annuli,
    }
    return png_bytes, meta


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
    """Figure 1 型（A–E）の GAWAIN 説明 PNG を出力する。"""
    output_path = Path(output_path)
    png_bytes, meta = render_gawain_figure_bytes(
        img_gray, mask, fovea_xy, scale_um_px, title=title, dpi=dpi,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png_bytes)
    meta["output_path"] = str(output_path)
    return meta
