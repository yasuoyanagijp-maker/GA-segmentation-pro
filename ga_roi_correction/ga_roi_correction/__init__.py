"""
ga_roi_correction
==================
GA(地図状萎縮)FAF連続画像のROI時系列自動補正パイプライン。

サブモジュール:
    preprocessing  - CLAHE / ノイズ低減など前処理
    registration   - フレーム間画像登録(Rigid/Affine, ECC)
    refine         - セグメンテーション境界改善(非AI手法 x5)
    qc             - 異常検出・品質管理・ログ出力
    pipeline       - 上記を統合した全体オーケストレーション
"""

from .pipeline import correct_ga_series, PipelineConfig
from .aligned_correct import (
    correct_aligned_series,
    AlignedCorrectConfig,
    choose_refine_method,
)
from .refine import (
    interpolate_prior_mask,
    position_guided_monotonic_refine,
    trusted_prior_mask,
)

__all__ = [
    "correct_ga_series",
    "PipelineConfig",
    "correct_aligned_series",
    "AlignedCorrectConfig",
    "choose_refine_method",
    "interpolate_prior_mask",
    "position_guided_monotonic_refine",
    "trusted_prior_mask",
]
__version__ = "0.1.0"
