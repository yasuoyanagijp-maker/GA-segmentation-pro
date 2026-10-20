"""GA 提出に使う数値列。ARIAKE OCTA の指標名は使わない。"""

from __future__ import annotations

# 個別時点の計測値。時系列の傾きは時点値から計算するので、ここには入れない。
GA_PRIMARY_METRICS = (
    "area_mm2",
    "area_mm_sqrt",
    "lesion_count",
    "gawain",
    "dist_um_nearest",
    "gap_um",
    "intact_1mm_mm2",
)

LESION_PRESENT = "lesion present"
QUALITY = "Quality of analysis"
FILE_COL = "File"

# 数値の採用対象にしない列。座標やスケールは読影者のクリックで必ずずれる。
META_COLUMNS = (
    FILE_COL,
    "ID",
    "Analyst",
    "case_id",
    "date",
    "filename",
    "confirmed",
    LESION_PRESENT,
    QUALITY,
    "scale_um_per_px",
    "disc_x",
    "disc_y",
    "fovea_x",
    "fovea_y",
    "outside_6mm_px",
    "needs_review",
)

RPD_THRESHOLD_PCT = 20.0
ABSENT_QUALITY = "N/A"
PRESENT_QUALITY = "OK"
