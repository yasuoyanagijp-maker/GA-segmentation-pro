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

    mode: str | None = None  # disc_line | disc_confirm | crop_drag | crop_confirm | fovea | seg_review | seg_erase
    disc_pts: list[tuple[float, float]] = field(default_factory=list)
    disc_draft_pts: list[tuple[float, float]] = field(default_factory=list)
    fovea_xy: tuple[float, float] | None = None
    scale_um_px: float | None = None

    img_full: np.ndarray | None = None
    img_gray: np.ndarray | None = None
    crop_box: tuple[int, int, int, int] | None = None
    crop_pts: list[tuple[float, float]] = field(default_factory=list)

    mask: np.ndarray | None = None
    stats: dict | None = None

    disc_diameter_um: float = 1800.0
    age: int = 75
    sex: str = "Male"
    prog: float = 73.0

    series_dates: dict[str, str] = field(default_factory=dict)
    series_results: list[dict] | None = None
    series_overlays: list[np.ndarray] | None = None
    series_gif: bytes | None = None
    series_time_idx: int = 0
    estimated_prog: float | None = None
    view_mode: str = "work"  # work | gif | graph

    def reset_analysis(self, *, keep_series: bool = False) -> None:
        self.mode = None
        self.disc_pts = []
        self.fovea_xy = None
        self.scale_um_px = None
        self.mask = None
        self.stats = None
        if not keep_series:
            self.series_results = None
            self.series_overlays = None
            self.series_gif = None
            self.estimated_prog = None

    def cancel_mode(self) -> None:
        if self.mode in ("disc_line", "disc_confirm"):
            self.disc_pts = []
            self.disc_draft_pts = []
        elif self.mode in ("crop_drag", "crop_confirm"):
            self.crop_pts = []
        elif self.mode in ("seg_review", "seg_erase"):
            self.mask = None
            self.stats = None
        self.mode = None

    def restart_disc(self) -> None:
        self.disc_pts = []
        self.disc_draft_pts = []
        self.scale_um_px = None
        self.mode = "disc_line"
