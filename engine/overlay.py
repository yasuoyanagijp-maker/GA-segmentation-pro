"""overlay.py – GAマスク・fovea・乳頭計測点のオーバーレイ画像生成。"""

import cv2
import numpy as np

# MNV_analysis_pro ROI 表示と同系（緑・40% 半透明）
MASK_BLEND_COLOR_RGB = (0, 255, 0)
MASK_BLEND_ALPHA = 0.4


def blend_mask_overlay(
    img_gray: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = MASK_BLEND_ALPHA,
    color_rgb: tuple[int, int, int] = MASK_BLEND_COLOR_RGB,
) -> np.ndarray:
    """GA マスクを MNV 同様の半透明オーバーレイで合成する。"""
    base = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    if mask is None or not np.any(mask > 0):
        return base
    overlay = base.copy()
    overlay[mask > 0] = color_rgb
    return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)


def make_overlay(img_gray: np.ndarray, mask: np.ndarray,
                 fovea_xy: tuple | None = None,
                 nearest_xy: tuple | None = None,
                 disc_pts: list | None = None,
                 disc_diameter_um: float = 1800.0,
                 crop_box: tuple[int, int, int, int] | None = None,
                 crop_pts: list | None = None,
                 show_macula: bool = False,
                 scale_um_px: float | None = None,
                 mask_style: str = "solid") -> np.ndarray:
    """RGBオーバーレイ画像（numpy配列）を返す。

    mask_style: "solid" = 赤塗り, "blend" = MNV 同系の半透明緑
    """
    if mask_style == "blend":
        rgb = blend_mask_overlay(img_gray, mask)
    else:
        rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB).copy()
        rgb[mask > 0] = [255, 60, 60]

    # Line fovea → nearest GA
    if fovea_xy and nearest_xy:
        cv2.line(rgb,
                 (int(fovea_xy[0]), int(fovea_xy[1])),
                 nearest_xy,
                 (134, 239, 172), 1)

    # Fovea crosshair
    if fovea_xy:
        fx, fy = int(fovea_xy[0]), int(fovea_xy[1])
        r = 8
        cv2.circle(rgb, (fx, fy), r, (34, 211, 238), 1)
        cv2.line(rgb, (fx - 18, fy), (fx - r, fy), (34, 211, 238), 1)
        cv2.line(rgb, (fx + r, fy), (fx + 18, fy), (34, 211, 238), 1)
        cv2.line(rgb, (fx, fy - 18), (fx, fy - r), (34, 211, 238), 1)
        cv2.line(rgb, (fx, fy + r), (fx, fy + 18), (34, 211, 238), 1)
        cv2.circle(rgb, (fx, fy), 2, (34, 211, 238), -1)

    # Disc measurement points and line (no on-image text; UI shows metrics)
    amber = (251, 191, 36)
    if disc_pts and len(disc_pts) >= 1:
        p1 = (int(disc_pts[0][0]), int(disc_pts[0][1]))
        cv2.circle(rgb, p1, 7, amber, -1)
        cv2.circle(rgb, p1, 9, amber, 1)
    if disc_pts and len(disc_pts) == 2:
        p2 = (int(disc_pts[1][0]), int(disc_pts[1][1]))
        cv2.circle(rgb, p2, 7, amber, -1)
        cv2.circle(rgb, p2, 9, amber, 1)
        cv2.line(rgb, p1, p2, amber, 2)

    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        cv2.rectangle(rgb, (x0, y0), (x1 - 1, y1 - 1), (96, 165, 250), 1)

    sky = (96, 165, 250)
    if crop_pts and len(crop_pts) >= 1:
        p1 = (int(crop_pts[0][0]), int(crop_pts[0][1]))
        cv2.circle(rgb, p1, 6, sky, -1)
    if crop_pts and len(crop_pts) == 2:
        p2 = (int(crop_pts[1][0]), int(crop_pts[1][1]))
        cv2.circle(rgb, p2, 6, sky, -1)
        x0, x1 = sorted((p1[0], p2[0]))
        y0, y1 = sorted((p1[1], p2[1]))
        cv2.rectangle(rgb, (x0, y0), (x1, y1), sky, 2)
    elif crop_pts and len(crop_pts) > 2:
        pts = np.array([(int(x), int(y)) for x, y in crop_pts], np.int32)
        cv2.polylines(rgb, [pts], False, sky, 2)
        fill_layer = rgb.copy()
        cv2.fillPoly(fill_layer, [pts], sky)
        rgb = cv2.addWeighted(fill_layer, 0.22, rgb, 0.78, 0)
        cv2.polylines(rgb, [pts], False, sky, 2)

    if show_macula and fovea_xy and scale_um_px and scale_um_px > 0:
        from engine.gawain import macula_radius_px
        radius_px = int(round(macula_radius_px(scale_um_px)))
        cv2.circle(rgb, (int(fovea_xy[0]), int(fovea_xy[1])), radius_px,
                   (167, 139, 250), 1)

    return rgb
