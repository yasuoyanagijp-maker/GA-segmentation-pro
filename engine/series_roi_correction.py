"""Apply ga_roi_correction QC/refine on already-aligned series rows.

Does not run image registration. Expects series rows from
build_series_from_saved(..., register=True) (or equivalent).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from engine.timeseries import _compute_row_stats

_PKG_ROOT = Path(__file__).resolve().parents[1] / "ga_roi_correction"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ga_roi_correction.aligned_correct import (  # noqa: E402
    AlignedCorrectConfig,
    correct_aligned_series,
)

DEFAULT_REFINE_METHOD = "band_adaptive_threshold"


def _as_mask255(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(bool)).astype(np.uint8) * 255


def apply_roi_correction_to_series(
    results: list[dict],
    *,
    refine_method: str = DEFAULT_REFINE_METHOD,
) -> dict:
    """
    Returns
    -------
    {
      "results": updated series rows (mask / stats / roi_* fields),
      "qc_records": list[FrameQCRecord],
      "needs_review_indices": list[int],
      "qc_log": SOP 11章相当のQCログ（CSV/表示向け plain dict のリスト）,
    }
    """
    if not results:
        raise ValueError("時系列がありません")

    images = [r["image"] for r in results]
    masks = [r["mask"] for r in results]
    dates = [r.get("date") for r in results]
    reg_modes = [r.get("reg_method") for r in results]
    # app registration does not store ECC cc; leave None
    reg_ccs = [None] * len(results)

    out = correct_aligned_series(
        images,
        masks,
        dates=dates,
        config=AlignedCorrectConfig(refine_method=refine_method),
        reg_modes=reg_modes,
        reg_ccs=reg_ccs,
    )

    updated: list[dict] = []
    needs_review_indices: list[int] = []
    for i, row in enumerate(results):
        rec = out["qc_records"][i]
        new_row = dict(row)
        raw255 = _as_mask255(out["raw_masks"][i])
        corr255 = _as_mask255(out["corrected_masks"][i])
        new_row["mask_pre_roi"] = raw255
        new_row["mask"] = corr255
        new_row["roi_needs_review"] = bool(rec.needs_review)
        new_row["roi_reason"] = rec.reason
        new_row["roi_ok"] = not bool(rec.needs_review)
        new_row["roi_raw_area"] = int(rec.raw_area)
        new_row["roi_corrected_area"] = int(rec.corrected_area)
        new_row["roi_refine_method"] = out["refine_methods"][i]
        scale = new_row.get("scale_um_px")
        fovea = new_row.get("fovea_xy")
        fallback = row.get("stats") or {}
        new_row["stats"] = _compute_row_stats(corr255, scale, fovea, fallback)
        if rec.needs_review:
            needs_review_indices.append(i)
        updated.append(new_row)

    qc_log = []
    for i, rec in enumerate(out["qc_records"]):
        entry = dict(rec.__dict__)
        entry["filename"] = results[i].get("filename")
        entry["refine_method"] = out["refine_methods"][i]
        qc_log.append(entry)

    return {
        "results": updated,
        "qc_records": out["qc_records"],
        "needs_review_indices": needs_review_indices,
        "qc_dataframe": out["qc_dataframe"],
        "refine_methods": out["refine_methods"],
        "qc_log": qc_log,
    }
