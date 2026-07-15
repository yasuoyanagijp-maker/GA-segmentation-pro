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
from ga_roi_correction.qc import (  # noqa: E402
    check_endpoint_containment,
    reconcile_endpoint_masks,
)

DEFAULT_REFINE_METHOD = "band_adaptive_threshold"
DEFAULT_ENDPOINT_MARGIN_PX = 3


def _as_mask255(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(bool)).astype(np.uint8) * 255


def apply_roi_correction_to_series(
    results: list[dict],
    *,
    refine_method: str = DEFAULT_REFINE_METHOD,
    endpoint_margin_px: int = DEFAULT_ENDPOINT_MARGIN_PX,
) -> dict:
    """
    Returns
    -------
    {
      "results": updated series rows (mask / stats / roi_* fields),
      "qc_records": list[FrameQCRecord],
      "needs_review_indices": list[int],
      "qc_log": SOP 11章相当のQCログ（CSV/表示向け plain dict のリスト）,
      "gate": containment info (before auto-reconcile),
      "endpoint_reconcile": None | dict (auto first/last adjust summary),
    }

    If first ⊈ dilate(last, margin), endpoints are auto-reconciled:
    remove leak from first and add the same pixels to last, then refine.
    """
    if not results:
        raise ValueError("時系列がありません")
    if len(results) < 2:
        raise ValueError("ROI補正には2時点以上が必要です")

    working = [dict(r) for r in results]
    first_mask = working[0]["mask"]
    last_mask = working[-1]["mask"]
    gate = check_endpoint_containment(
        first_mask, last_mask, margin_px=endpoint_margin_px,
    )
    gate = {
        **gate,
        "first_filename": working[0].get("filename"),
        "last_filename": working[-1].get("filename"),
        "first_date": working[0].get("date"),
        "last_date": working[-1].get("date"),
    }

    endpoint_reconcile = None
    if not gate["ok"]:
        rec = reconcile_endpoint_masks(
            first_mask, last_mask, margin_px=endpoint_margin_px,
        )
        working[0]["mask"] = _as_mask255(rec["first_mask"])
        working[-1]["mask"] = _as_mask255(rec["last_mask"])
        endpoint_reconcile = {
            "removed_from_first_px": rec["removed_from_first_px"],
            "added_to_last_px": rec["added_to_last_px"],
            "margin_px": rec["margin_px"],
            "first_filename": working[0].get("filename"),
            "last_filename": working[-1].get("filename"),
        }
        # Re-check after reconcile (should always pass).
        gate_after = check_endpoint_containment(
            working[0]["mask"], working[-1]["mask"], margin_px=endpoint_margin_px,
        )
        if not gate_after["ok"]:
            raise RuntimeError(
                "endpoint reconcile failed to restore containment "
                f"(leak={gate_after['leak_pixels']})"
            )

    images = [r["image"] for r in working]
    masks = [r["mask"] for r in working]
    dates = [r.get("date") for r in working]
    reg_modes = [r.get("reg_method") for r in working]
    # app registration does not store ECC cc; leave None
    reg_ccs = [None] * len(working)

    out = correct_aligned_series(
        images,
        masks,
        dates=dates,
        config=AlignedCorrectConfig(
            refine_method=refine_method,
            lock_endpoints=True,
            endpoint_bound_margin_px=endpoint_margin_px,
        ),
        reg_modes=reg_modes,
        reg_ccs=reg_ccs,
    )

    updated: list[dict] = []
    needs_review_indices: list[int] = []
    for i, row in enumerate(working):
        rec = out["qc_records"][i]
        new_row = dict(row)
        # mask_pre_roi = pre-refine (after endpoint reconcile if any)
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
        if endpoint_reconcile and i in (0, len(working) - 1):
            tag = "endpoint_reconcile"
            if new_row["roi_reason"] in (None, "", "ok"):
                new_row["roi_reason"] = tag
            elif tag not in str(new_row["roi_reason"]):
                new_row["roi_reason"] = f"{new_row['roi_reason']};{tag}"
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
        entry["filename"] = working[i].get("filename")
        entry["refine_method"] = out["refine_methods"][i]
        if endpoint_reconcile and i in (0, len(working) - 1):
            entry["endpoint_reconcile"] = True
        qc_log.append(entry)

    return {
        "results": updated,
        "qc_records": out["qc_records"],
        "needs_review_indices": needs_review_indices,
        "qc_dataframe": out["qc_dataframe"],
        "refine_methods": out["refine_methods"],
        "qc_log": qc_log,
        "gate_failed": False,
        "gate": gate,
        "endpoint_reconcile": endpoint_reconcile,
    }
