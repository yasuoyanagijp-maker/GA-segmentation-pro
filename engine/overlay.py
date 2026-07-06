"""overlay.py – GAマスク・fovea・乳頭計測点のオーバーレイ画像生成。"""

import cv2
import numpy as np


def make_overlay(img_gray: np.ndarray, mask: np.ndarray,
                 fovea_xy: tuple | None = None,
                 nearest_xy: tuple | None = None,
                 disc_pts: list | None = None) -> np.ndarray:
    """RGBオーバーレイ画像（numpy配列）を返す。"""
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB).copy()

    # GA mask in red
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

    # Disc measurement points and line
    amber = (251, 191, 36)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if disc_pts and len(disc_pts) >= 1:
        p1 = (int(disc_pts[0][0]), int(disc_pts[0][1]))
        cv2.circle(rgb, p1, 7, amber, -1)
        cv2.circle(rgb, p1, 9, amber, 1)
        cv2.putText(rgb, "1", (p1[0]+10, p1[1]-8), font, 0.55, amber, 1, cv2.LINE_AA)
    if disc_pts and len(disc_pts) == 2:
        p2 = (int(disc_pts[1][0]), int(disc_pts[1][1]))
        cv2.circle(rgb, p2, 7, amber, -1)
        cv2.circle(rgb, p2, 9, amber, 1)
        cv2.putText(rgb, "2", (p2[0]+10, p2[1]-8), font, 0.55, amber, 1, cv2.LINE_AA)
        cv2.line(rgb, p1, p2, amber, 2)
        # midpoint label: px distance
        d_px = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        mx = (p1[0]+p2[0])//2
        my = (p1[1]+p2[1])//2 - 10
        label = f"{d_px:.0f}px = 1800um"
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        cv2.rectangle(rgb, (mx-3, my-th-3), (mx+tw+3, my+3), (0, 0, 0), -1)
        cv2.putText(rgb, label, (mx, my), font, 0.45, amber, 1, cv2.LINE_AA)

    return rgb
