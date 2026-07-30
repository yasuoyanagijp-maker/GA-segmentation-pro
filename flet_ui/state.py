"""Flet UI 用のアプリケーション状態（page.data に保持）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from engine.cases import UNCLASSIFIED_CASE_ID


@dataclass
class GAAppState:
    input_dir: str = ""
    output_dir: str = ""
    files: list[Path] = field(default_factory=list)
    file_idx: int = 0

    # 症例グループ（フォルダ読込時に構築）
    cases: dict[str, list[Path]] = field(default_factory=dict)
    case_id: str = UNCLASSIFIED_CASE_ID
    # case_id → その症例の確定スナップショット（切替時に stash/restore）
    case_results: dict[str, dict[str, dict]] = field(default_factory=dict)
    # 症例へ一括移動する画像（filename → checked）
    case_move_selected: dict[str, bool] = field(default_factory=dict)

    # False=独立解析（既定・registration なし） / True=時系列（症例内のみ）
    series_mode: bool = False

    mode: str | None = None  # disc_line | disc_confirm | width_cal | crop_drag | crop_confirm | fovea | foveal_confirm | seg_param | seg_review | seg_erase | seg_draw | seg_clear | seg_clear_confirm
    image_modality: str = "optos"  # optos | hra
    # disc_1800um | dfd_4.76mm | width（旧スナップショットは disc もあり）
    calibration_source: str | None = None
    # Optos 既定: DFD（文献平均 4.76mm）。disc = 乳頭径固定。
    scale_method: str = "dfd"  # dfd | disc
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
    # Auto-seg パラメータ調整（seg_param パネル）
    macula_radius_mm: float = 6.0
    seg_min_size: int = 500
    # 自動セグ後の制約付き Chan-Vese 精緻化（既定 ON）
    seg_auto_cv: bool = True
    scale_um_px: float | None = None

    img_full: np.ndarray | None = None
    img_gray: np.ndarray | None = None
    crop_box: tuple[int, int, int, int] | None = None
    crop_pts: list[tuple[float, float]] = field(default_factory=list)
    clear_pts: list[tuple[float, float]] = field(default_factory=list)

    mask: np.ndarray | None = None
    stats: dict | None = None
    # Interactive ROI grow: standard | lsf | cv_refine（Phase A 安全レールは常時）
    grow_mode: str = "standard"

    disc_diameter_um: float = 1800.0
    age: int = 75
    sex: str = "Male"
    prog: float = 73.0

    per_file_results: dict[str, dict] = field(default_factory=dict)
    batch_mode: bool = False
    series_dates: dict[str, str] = field(default_factory=dict)
    series_dates_manual: dict[str, bool] = field(default_factory=dict)  # 手動入力した撮影日時
    # fname → exif|filename|manual|mtime|missing
    series_date_source: dict[str, str] = field(default_factory=dict)
    series_selected: dict[str, bool] = field(default_factory=dict)  # filename -> 時系列に含める
    series_methods: dict[str, str] = field(default_factory=dict)    # filename -> 位置合わせ手法
    # 時系列モード時のみ意味を持つ（独立解析では常に False）
    series_registration_enabled: bool = False
    series_registered: bool = False              # 現在の時系列に位置合わせ済みか
    # 整列QC後に seg（本線）。OFF 時のみ先seg→warp フォールバック
    series_resegment_enabled: bool = True
    series_roi_enabled: bool = True   # マスクQC補正トグル（既定 ON、位置合わせ確認後に適用）
    series_roi_applied: bool = False  # 現在の時系列にQC補正適用済みか
    series_roi_qc: list[dict] | None = None  # QC補正ログ（SOP 11章相当、CSV出力用）
    # 再解析後に位置合わせを再利用するためのメモリキャッシュ
    # {selected: tuple[str], ref: str, confirmed: bool, by_file: {fname: {matrix, method}}}
    series_reg_cache: dict | None = None
    series_results: list[dict] | None = None
    series_overlays: list[np.ndarray] | None = None
    series_gif: bytes | None = None
    series_time_idx: int = 0
    series_review_active: bool = False  # registration 確認モード中
    series_roi_review_active: bool = False  # ROI時系列補正の確認モード中
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
            self.series_roi_review_active = False
            self.series_review_idx = 0
            self.estimated_prog = None
            self.estimated_growth_mm2_per_year = None
            self.estimated_growth_mm_sqrt_per_year = None
            self.per_file_results = {}
            self.series_selected = {}
            self.case_move_selected = {}
            self.series_methods = {}
            self.series_dates_manual = {}
            self.series_date_source = {}
            self.series_mode = False
            self.series_registration_enabled = False
            self.series_registered = False
            self.series_resegment_enabled = True
            self.series_roi_enabled = True
            self.series_roi_applied = False
            self.series_roi_qc = None
            self.series_reg_cache = None
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
        elif self.mode in ("seg_param", "seg_review", "seg_erase", "seg_draw"):
            self.mask = None
            self.stats = None
        self.mode = None

    def restart_disc(self) -> None:
        self.disc_pts = []
        self.disc_draft_pts = []
        self.scale_um_px = None
        self.calibration_source = None
        self.mode = "disc_line"

    def stash_current_case_results(self) -> None:
        """現在の per_file_results を case_results[case_id] へ退避。"""
        if self.case_id:
            self.case_results[self.case_id] = dict(self.per_file_results)

    def restore_case_results(self, case_id: str) -> None:
        """指定症例の確定結果を per_file_results へ復元。"""
        self.per_file_results = dict(self.case_results.get(case_id) or {})
