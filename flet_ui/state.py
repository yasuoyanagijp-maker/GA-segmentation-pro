"""アプリケーション状態（Streamlit session_state の代替）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class GAAppState:
    input_dir: str = ""
    output_dir: str = ""
    files: list[Path] = field(default_factory=list)
    file_idx: int = 0

    mode: str | None = None  # disc_line | disc_confirm | width_cal | crop_drag | crop_confirm | fovea | foveal_confirm | seg_review | seg_erase | seg_draw | seg_clear | seg_clear_confirm
    image_modality: str = "optos"  # optos | hra
    calibration_source: str | None = None  # disc | width
    field_width_mm: float | None = None
    disc_pts: list[tuple[float, float]] = field(default_factory=list)
    disc_draft_pts: list[tuple[float, float]] = field(default_factory=list)
    fovea_xy: tuple[float, float] | None = None
    img_gray_raw: np.ndarray | None = None
    foveal_correction_applied: bool = False
    foveal_correction_skipped: bool = False
    foveal_correction_strength: float = 0.0
    use_clahe: bool = True
    clahe_clip: float = 2.0
    scale_um_px: float | None = None

    img_full: np.ndarray | None = None
    img_gray: np.ndarray | None = None
    crop_box: tuple[int, int, int, int] | None = None
    crop_pts: list[tuple[float, float]] = field(default_factory=list)
    clear_pts: list[tuple[float, float]] = field(default_factory=list)

    mask: np.ndarray | None = None
    stats: dict | None = None

    disc_diameter_um: float = 1800.0
    age: int = 75
    sex: str = "Male"
    prog: float = 73.0

    per_file_results: dict[str, dict] = field(default_factory=dict)
    batch_mode: bool = False
    series_dates: dict[str, str] = field(default_factory=dict)
    series_dates_manual: dict[str, bool] = field(default_factory=dict)  # 手動入力した撮影日時
    series_selected: dict[str, bool] = field(default_factory=dict)  # filename -> 時系列に含める
    series_methods: dict[str, str] = field(default_factory=dict)    # filename -> 位置合わせ手法
    series_registration_enabled: bool = False  # 位置合わせトグル（既定 OFF）
    series_registered: bool = False              # 現在の時系列に位置合わせ済みか
    series_results: list[dict] | None = None
    series_overlays: list[np.ndarray] | None = None
    series_gif: bytes | None = None
    series_time_idx: int = 0
    series_review_active: bool = False  # registration 確認モード中
    series_review_idx: int = 0
    graph_metric: str = "area_mm2"  # area_mm2 | area_mm_sqrt | gawain | dist_um
    estimated_prog: float | None = None
    estimated_growth_mm2_per_year: float | None = None
    estimated_growth_mm_sqrt_per_year: float | None = None
    view_mode: str = "work"  # work | gif | graph

    def reset_per_image(self) -> None:
        """画像切替時: スケール較正は維持し、中心窩・乳頭・解析結果のみリセット。

        乳頭計測点は画像ごとに異なるため、次画像へ進むたびに消す。
        （幅指定スケールは scale_um_px / calibration_source で保持）
        """
        self.mode = None
        self.fovea_xy = None
        self.disc_pts = []
        self.disc_draft_pts = []
        self.img_gray_raw = None
        self.foveal_correction_applied = False
        self.foveal_correction_skipped = False
        self.clear_pts = []
        self.mask = None
        self.stats = None

    def reset_analysis(self, *, keep_series: bool = False) -> None:
        """フルリセット（クロップ後・新規解析開始）。"""
        self.reset_per_image()
        self.disc_pts = []
        self.foveal_correction_strength = 0.0
        self.scale_um_px = None
        self.calibration_source = None
        if not keep_series:
            self.series_results = None
            self.series_overlays = None
            self.series_gif = None
            self.series_review_active = False
            self.series_review_idx = 0
            self.estimated_prog = None
            self.estimated_growth_mm2_per_year = None
            self.estimated_growth_mm_sqrt_per_year = None
            self.per_file_results = {}
            self.series_selected = {}
            self.series_methods = {}
            self.series_dates_manual = {}
            self.series_registration_enabled = False
            self.series_registered = False
            self.batch_mode = False

    def cancel_mode(self) -> None:
        if self.mode in ("disc_line", "disc_confirm"):
            self.disc_pts = []
            self.disc_draft_pts = []
        elif self.mode in ("crop_drag", "crop_confirm"):
            self.crop_pts = []
        elif self.mode == "width_cal":
            pass
        elif self.mode == "foveal_confirm":
            pass
        elif self.mode in ("seg_clear", "seg_clear_confirm"):
            self.clear_pts = []
        elif self.mode in ("seg_review", "seg_erase", "seg_draw"):
            self.mask = None
            self.stats = None
        self.mode = None

    def restart_disc(self) -> None:
        self.disc_pts = []
        self.disc_draft_pts = []
        self.scale_um_px = None
        self.calibration_source = None
        self.mode = "disc_line"
