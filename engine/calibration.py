"""calibration.py – 視神経乳頭径クリックからのスケール較正（um/px）。"""

import numpy as np

DEFAULT_DISC_DIAMETER_UM = 1800.0


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


def compute_scale_um_per_px_from_field_width(image_width_px: float,
                                              field_width_mm: float) -> float:
    """画像幅(px)と実長(mm)から um/px スケールを算出する（HRA 等、乳頭非表示時）。"""
    if image_width_px <= 0:
        raise ValueError("画像幅(px)が不正です")
    if field_width_mm <= 0:
        raise ValueError("画像幅(mm)は正の値を指定してください")
    return field_width_mm * 1000.0 / image_width_px
