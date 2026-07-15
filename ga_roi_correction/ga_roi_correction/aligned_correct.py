"""
Pre-aligned series ROI correction (no registration).

Use when images/masks are already in a common coordinate space
(e.g. after engine.series_registration). Does not call registration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import preprocessing as pp
from . import refine as rf
from . import qc
from .pipeline import PipelineConfig, _apply_refine, _cap_to_neighbor


@dataclass
class AlignedCorrectConfig:
    thresholds: qc.AnomalyThresholds = field(default_factory=qc.AnomalyThresholds)
    # Confirmed ROI: default to boundary-band refine; escalate only for severe area ratios.
    refine_method: str = "band_adaptive_threshold"
    cap_margin_px: int = 8
    reconstruction_margin_px: int = 6
    clahe_clip_limit: float = 2.5
    denoise_h: float = 6.0
    # Detect grossly mislocated ROIs (area may look normal) before the
    # area-based checks, and fix them with neighbor-guided refine +
    # monotonicity constraint (prev ⊆ result ⊆ dilate(next, margin)).
    enable_position_guided_refine: bool = True
    position_next_margin_px: int = 3


def _area_severity_tier(
    prev_area: Optional[int],
    curr_area: int,
    next_area: Optional[int],
    th: qc.AnomalyThresholds,
) -> int:
    """0=mild, 1=severe→chan_vese, 2=extreme→grabcut.

    Uses the same neighbor comparisons as detect_area_anomaly, scaled by
    AnomalyThresholds (2× / 3× the mild trigger).
    """
    if prev_area is None and next_area is not None and next_area > 0:
        ratio = curr_area / next_area
        if ratio >= th.first_ratio * 3 / 2:  # e.g. 1.5 → 2.25
            return 2
        if ratio >= th.first_ratio * 4 / 3:  # e.g. 1.5 → 2.0
            return 1
        return 0
    if next_area is None and prev_area is not None and prev_area > 0:
        ratio = curr_area / prev_area
        if ratio >= th.last_ratio * 3 / 2:
            return 2
        if ratio >= th.last_ratio * 4 / 3:
            return 1
        return 0
    if prev_area is not None and prev_area > 0 and curr_area < prev_area * (1 - th.rel_drop):
        drop = 1.0 - (curr_area / prev_area)
        if drop >= th.rel_drop * 10:  # e.g. 0.05 → 0.50
            return 2
        if drop >= th.rel_drop * 5:   # e.g. 0.05 → 0.25
            return 1
        return 0
    if next_area is not None and next_area > 0 and curr_area > next_area * (1 + th.rel_jump):
        jump = (curr_area / next_area) - 1.0
        if jump >= th.rel_jump * 3:  # e.g. 0.20 → 0.60
            return 2
        if jump >= th.rel_jump * 2:  # e.g. 0.20 → 0.40
            return 1
        return 0
    return 0


def choose_refine_method(
    prev_area: Optional[int],
    curr_area: int,
    next_area: Optional[int],
    th: qc.AnomalyThresholds,
    default_method: str = "band_adaptive_threshold",
) -> str:
    """Mild→band_adaptive_threshold; severe→chan_vese; extreme→grabcut."""
    tier = _area_severity_tier(prev_area, curr_area, next_area, th)
    if tier >= 2:
        return "grabcut"
    if tier >= 1:
        return "chan_vese"
    return default_method


def correct_aligned_series(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    dates: Optional[list[str]] = None,
    config: Optional[AlignedCorrectConfig] = None,
    reg_modes: Optional[list[Optional[str]]] = None,
    reg_ccs: Optional[list[Optional[float]]] = None,
) -> dict:
    """
    Parameters
    ----------
    images, masks : already-aligned frames (same HxW each)
    dates : optional labels for QC records
    reg_modes / reg_ccs : optional passthrough into QC (no re-registration)

    Returns
    -------
    dict with corrected_masks, raw_masks, qc_records, qc_dataframe, refine_methods
    """
    assert len(images) == len(masks)
    n = len(images)
    if n == 0:
        raise ValueError("images が空です")

    cfg = config or AlignedCorrectConfig()
    # reuse PipelineConfig field names for refine helpers
    pcfg = PipelineConfig(
        thresholds=cfg.thresholds,
        refine_method=cfg.refine_method,
        cap_margin_px=cfg.cap_margin_px,
        reconstruction_margin_px=cfg.reconstruction_margin_px,
        clahe_clip_limit=cfg.clahe_clip_limit,
        denoise_h=cfg.denoise_h,
        save_overlays=False,
    )

    pre = [
        pp.preprocess(img, clip_limit=pcfg.clahe_clip_limit, denoise_h=pcfg.denoise_h)
        for img in images
    ]
    analysis = [p["analysis_image"] for p in pre]
    aligned_masks = [(m > 0).astype(np.uint8) for m in masks]

    raw_areas = [qc.mask_area(m) for m in aligned_masks]
    dice_prev_raw = [None] + [
        qc.dice(aligned_masks[i], aligned_masks[i - 1]) for i in range(1, n)
    ]

    corrected: list[np.ndarray] = [None] * n  # type: ignore
    corrected[0] = aligned_masks[0]
    refine_methods: list[Optional[str]] = [None] * n
    dice_neighbor_prior: list[Optional[float]] = [None] * n

    for i in range(n):
        prev_m = corrected[i - 1] if i > 0 else None
        next_m = aligned_masks[i + 1] if i < n - 1 else None

        # Position check runs before the area checks: a frame whose area is
        # in range can still be grossly shifted, which area QC cannot see.
        # Edge frames are skipped: with a single neighbor we cannot tell
        # which of the two frames is mislocated, and a shifted neighbor
        # would drag a correct edge frame to the wrong place.
        if cfg.enable_position_guided_refine and prev_m is not None and next_m is not None:
            prior = rf.trusted_prior_mask(prev_m, next_m, t=0.5)
            if prior is not None:
                dice_neighbor_prior[i] = qc.dice(aligned_masks[i], prior)
            if qc.detect_position_anomaly(dice_neighbor_prior[i], pcfg.thresholds):
                refine_methods[i] = "position_guided_monotonic"
                corrected[i] = rf.position_guided_monotonic_refine(
                    analysis[i], aligned_masks[i],
                    prev_mask=prev_m, next_mask=next_m,
                    next_margin_px=cfg.position_next_margin_px,
                )
                continue

        prev_a = qc.mask_area(corrected[i - 1]) if i > 0 else None
        next_a = raw_areas[i + 1] if i < n - 1 else None
        is_anom = qc.detect_area_anomaly(prev_a, raw_areas[i], next_a, pcfg.thresholds)

        if not is_anom:
            corrected[i] = aligned_masks[i]
            continue

        curr = aligned_masks[i]
        if i > 0 and prev_a is not None and raw_areas[i] < prev_a * (1 - pcfg.thresholds.rel_drop):
            curr = rf.geodesic_prior_reconstruction(
                curr, corrected[i - 1], pcfg.reconstruction_margin_px,
            )
        else:
            neighbor = corrected[i - 1] if i > 0 else aligned_masks[min(i + 1, n - 1)]
            if i == n - 1 and prev_a is not None:
                neighbor = corrected[i - 1]
            elif i == 0 and next_a is not None:
                neighbor = aligned_masks[1]
            curr = _cap_to_neighbor(curr, neighbor, pcfg.cap_margin_px)

        method = choose_refine_method(
            prev_a, raw_areas[i], next_a, pcfg.thresholds, cfg.refine_method,
        )
        refine_methods[i] = method
        prev_for_refine = corrected[i - 1] if i > 0 else None
        curr = _apply_refine(method, analysis[i], curr, prev_for_refine)
        corrected[i] = curr

    corrected_areas = [qc.mask_area(m) for m in corrected]
    dice_prev_corr = [None] + [
        qc.dice(corrected[i], corrected[i - 1]) for i in range(1, n)
    ]
    modes = list(reg_modes) if reg_modes is not None else [None] * n
    ccs = list(reg_ccs) if reg_ccs is not None else [None] * n
    if len(modes) != n:
        modes = [None] * n
    if len(ccs) != n:
        ccs = [None] * n

    records = qc.build_qc_records(
        dates, raw_areas, corrected_areas, dice_prev_raw, dice_prev_corr,
        modes, ccs, pcfg.thresholds,
        dice_neighbor_prior=dice_neighbor_prior,
    )
    return {
        "corrected_masks": corrected,
        "raw_masks": aligned_masks,
        "qc_records": records,
        "qc_dataframe": qc.records_to_dataframe(records),
        "refine_methods": refine_methods,
    }
