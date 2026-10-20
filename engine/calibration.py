"""calibration.py – スケール較正（um/px）。

乳頭径固定・disc–fovea距離 (DFD)・画像幅指定の3系統。
"""

from __future__ import annotations

import numpy as np

DEFAULT_DISC_DIAMETER_UM = 1800.0
# Beijing Eye Study (Jonas et al.) 文献平均 disc–fovea distance
DEFAULT_DFD_UM = 4760.0

SOURCE_DISC_PREFIX = "disc_"
SOURCE_DFD_PREFIX = "dfd_"
SOURCE_WIDTH = "width"


def source_id_disc(disc_diameter_um: float = DEFAULT_DISC_DIAMETER_UM) -> str:
    """例: disc_1800um"""
    return f"disc_{int(round(disc_diameter_um))}um"


def source_id_dfd(dfd_um: float = DEFAULT_DFD_UM) -> str:
    """例: dfd_4.76mm"""
    return f"dfd_{dfd_um / 1000.0:.2f}mm"


def calibration_source_label(source: str | None) -> str:
    """UI / PDF 用の短い表示名。"""
    if not source:
        return "-"
    if source == "disc" or source.startswith(SOURCE_DISC_PREFIX):
        if source.startswith(SOURCE_DISC_PREFIX) and source.endswith("um"):
            um = source[len(SOURCE_DISC_PREFIX):-2]
            return f"乳頭 {um}μm"
        return "乳頭"
    if source.startswith(SOURCE_DFD_PREFIX):
        # dfd_4.76mm → DFD 4.76mm
        return f"DFD {source[len(SOURCE_DFD_PREFIX):]}"
    if source == SOURCE_WIDTH:
        return "幅指定"
    return source


def compute_scale_um_per_px(p1: tuple[float, float],
                             p2: tuple[float, float],
                             disc_diameter_um: float = DEFAULT_DISC_DIAMETER_UM
                             ) -> float:
    """乳頭の端2点(px)から um/px スケールを算出する。

    Args:
        p1, p2: 乳頭の端2点の画像座標 (x, y)。
        disc_diameter_um: 較正基準とする乳頭径（デフォルト1800um、UIから変更可）。

    Returns:
        1pxあたりのミクロン数。
    """
    d_px = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    if d_px <= 0:
        raise ValueError("2点が同一座標のためスケールを算出できません")
    return disc_diameter_um / d_px


def compute_scale_um_per_px_from_dfd(
    fovea_xy: tuple[float, float],
    disc_xy: tuple[float, float],
    dfd_um: float = DEFAULT_DFD_UM,
) -> float:
    """中心窩–乳頭中心距離(px)と文献DFD(um)から um/px を算出する。"""
    dx = float(disc_xy[0]) - float(fovea_xy[0])
    dy = float(disc_xy[1]) - float(fovea_xy[1])
    d_px = float(np.sqrt(dx * dx + dy * dy))
    if d_px <= 0:
        raise ValueError("中心窩と乳頭中心が同一座標のためDFDスケールを算出できません")
    if dfd_um <= 0:
        raise ValueError("DFD (um) は正の値を指定してください")
    return dfd_um / d_px


def compute_scale_um_per_px_from_field_width(image_width_px: float,
                                              field_width_mm: float) -> float:
    """画像幅(px)と実長(mm)から um/px スケールを算出する（HRA 等、乳頭非表示時）。"""
    if image_width_px <= 0:
        raise ValueError("画像幅(px)が不正です")
    if field_width_mm <= 0:
        raise ValueError("画像幅(mm)は正の値を指定してください")
    return field_width_mm * 1000.0 / image_width_px
