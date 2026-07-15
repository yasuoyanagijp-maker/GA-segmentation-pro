"""
pipeline.py
-----------
SOP 5〜11章を統合する全体オーケストレーション。

処理の流れ:
  1. 各フレームを前処理 (preprocessing)
  2. 基準フレーム(既定: 先頭)へ全フレームを画像登録 (registration)
     -> 以降はすべて「基準フレーム座標系(aligned space)」で処理する
  3. 面積異常より先に「位置ずれ」を検出 (qc.detect_position_anomaly):
     前後フレームROIのshape-based interpolationで作ったpriorとのDiceが
     低いフレームは、面積が正常でも位置がずれているとみなし、
     position_guided_monotonic_refine(前後フレーム誘導+単調性制約)で補正
  3'. SOPの初回/中間/最終ロジックで面積異常を検出 (qc.detect_area_anomaly)
  4. 異常フレームのみ補正:
       - 縮小異常(中間) -> geodesic_prior_reconstruction で前フレームの
         形状を尊重しつつ回復
       - 過大異常(初回/最終/中間の急増) -> 隣接フレームに基づく上限で
         キャップ (dilateした隣接マスクとの積集合)
       - キャップ/再構成のあと、指定した境界改善手法(既定: chan_vese)で
         境界を滑らかに仕上げる
  5. QCレコード生成・CSV保存・(任意)オーバーレイ画像保存
  6. 補正後マスクを各フレームの元座標系へ逆変換して返す(可視化用)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

from . import preprocessing as pp
from . import registration as reg
from . import refine as rf
from . import qc


@dataclass
class PipelineConfig:
    reference_index: int = 0
    thresholds: qc.AnomalyThresholds = field(default_factory=qc.AnomalyThresholds)
    refine_method: str = "chan_vese"          # rf.METHODS のいずれか
    cap_margin_px: int = 8                    # 過大異常キャップ時の許容マージン
    reconstruction_margin_px: int = 6         # 縮小異常回復時の許容マージン
    clahe_clip_limit: float = 2.5
    denoise_h: float = 6.0
    save_overlays: bool = True
    # 位置ずれフレームの検出+前後フレーム誘導補正(面積チェックより先に実施)
    enable_position_guided_refine: bool = True
    position_next_margin_px: int = 3          # 単調性上限(next側)の許容マージン


def _cap_to_neighbor(curr_mask: np.ndarray, neighbor_mask: np.ndarray, margin_px: int) -> np.ndarray:
    """過大異常マスクを、隣接フレームマスクの膨張領域内に制限する。"""
    allowed = cv2.dilate(
        (neighbor_mask.astype(np.uint8) * 255),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)),
    ) > 0
    return (curr_mask.astype(bool) & allowed).astype(np.uint8)


def _apply_refine(method_name: str, analysis_img: np.ndarray, mask: np.ndarray,
                   prev_mask: Optional[np.ndarray] = None) -> np.ndarray:
    fn = rf.METHODS[method_name]
    if method_name == "geodesic_prior_reconstruction":
        if prev_mask is None:
            return mask
        return fn(mask, prev_mask)
    return fn(analysis_img, mask)


def correct_ga_series(images: list[np.ndarray], masks: list[np.ndarray],
                       dates: Optional[list[str]] = None,
                       config: Optional[PipelineConfig] = None,
                       output_dir: Optional[str] = None) -> dict:
    """
    Parameters
    ----------
    images : 時系列順のFAF画像リスト(各 HxW または HxWx3)
    masks  : 対応する初期ROIマスクリスト(各 HxW, 0/1 or bool)
    dates  : 任意。QCログ用の撮影日文字列リスト
    config : PipelineConfig
    output_dir : 指定するとQC CSV・オーバーレイPNGを保存する

    Returns
    -------
    dict with keys:
      corrected_masks_aligned   : 基準フレーム座標系での補正後マスク一覧
      corrected_masks_original  : 各フレーム元座標系へ逆変換した補正後マスク一覧
      qc_dataframe              : pandas.DataFrame (SOP 11章の記録)
      registration_results      : 各フレームの RegistrationResult
    """
    assert len(images) == len(masks), "images と masks の枚数が一致しません"
    n = len(images)
    if n == 0:
        raise ValueError("images が空です")

    config = config or PipelineConfig()
    ref_idx = config.reference_index

    # 1. 前処理
    pre = [pp.preprocess(img, clip_limit=config.clahe_clip_limit, denoise_h=config.denoise_h) for img in images]
    shape_hw = pre[ref_idx]["registration_image"].shape[:2]

    # 2. 基準フレームへ登録
    reg_results: list[reg.RegistrationResult] = []
    aligned_analysis: list[np.ndarray] = []
    aligned_masks: list[np.ndarray] = []
    for i in range(n):
        if i == ref_idx:
            rr = reg.RegistrationResult(True, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), "reference", 1.0)
        else:
            rr = reg.register_frame(pre[ref_idx]["registration_image"], pre[i]["registration_image"])
        reg_results.append(rr)
        aligned_analysis.append(reg.warp_image(pre[i]["analysis_image"], rr.warp_matrix, shape_hw))
        aligned_masks.append(reg.warp_mask(masks[i], rr.warp_matrix, shape_hw))

    raw_areas = [qc.mask_area(m) for m in aligned_masks]
    dice_prev_raw = [None] + [qc.dice(aligned_masks[i], aligned_masks[i - 1]) for i in range(1, n)]

    # 3-4. 異常検出 + 補正(時系列順に前から処理し、直前の"補正後"マスクを参照)
    corrected: list[np.ndarray] = [None] * n  # type: ignore
    corrected[0] = aligned_masks[0]
    dice_neighbor_prior: list[Optional[float]] = [None] * n

    for i in range(n):
        prev_m = corrected[i - 1] if i > 0 else None
        next_m = aligned_masks[i + 1] if i < n - 1 else None

        # 位置ずれチェック(面積チェックより先): 面積が正常でも
        # 前後フレーム補間priorとの重なりが低ければ場所がずれている。
        # 片側しか隣が無い端フレームでは「どちらがずれているか」を
        # 判定できない(ずれた隣に引きずられて正常フレームを壊す)ため、
        # 両隣が存在し互いに整合するフレームのみを対象とする。
        if config.enable_position_guided_refine and prev_m is not None and next_m is not None:
            prior = rf.trusted_prior_mask(prev_m, next_m, t=0.5)
            if prior is not None:
                dice_neighbor_prior[i] = qc.dice(aligned_masks[i], prior)
            if qc.detect_position_anomaly(dice_neighbor_prior[i], config.thresholds):
                corrected[i] = rf.position_guided_monotonic_refine(
                    aligned_analysis[i], aligned_masks[i],
                    prev_mask=prev_m, next_mask=next_m,
                    next_margin_px=config.position_next_margin_px,
                )
                continue

        prev_a = qc.mask_area(corrected[i - 1]) if i > 0 else None
        next_a = raw_areas[i + 1] if i < n - 1 else None
        is_anom = qc.detect_area_anomaly(prev_a, raw_areas[i], next_a, config.thresholds)

        if not is_anom:
            corrected[i] = aligned_masks[i]
            continue

        curr = aligned_masks[i]
        if i > 0 and prev_a is not None and raw_areas[i] < prev_a * (1 - config.thresholds.rel_drop):
            # 縮小異常 -> 前フレーム形状を尊重して回復
            curr = rf.geodesic_prior_reconstruction(curr, corrected[i - 1], config.reconstruction_margin_px)
        else:
            # 過大異常(初回/最終/急増) -> 隣接フレームでキャップ
            neighbor = corrected[i - 1] if i > 0 else aligned_masks[min(i + 1, n - 1)]
            if i == n - 1 and prev_a is not None:
                neighbor = corrected[i - 1]
            elif i == 0 and next_a is not None:
                neighbor = aligned_masks[1]
            curr = _cap_to_neighbor(curr, neighbor, config.cap_margin_px)

        # 境界を仕上げる
        prev_for_refine = corrected[i - 1] if i > 0 else None
        curr = _apply_refine(config.refine_method, aligned_analysis[i], curr, prev_for_refine)
        corrected[i] = curr

    corrected_areas = [qc.mask_area(m) for m in corrected]
    dice_prev_corr = [None] + [qc.dice(corrected[i], corrected[i - 1]) for i in range(1, n)]
    reg_modes = [r.mode for r in reg_results]
    reg_ccs = [r.cc for r in reg_results]

    records = qc.build_qc_records(dates, raw_areas, corrected_areas, dice_prev_raw, dice_prev_corr,
                                   reg_modes, reg_ccs, config.thresholds,
                                   dice_neighbor_prior=dice_neighbor_prior)

    # 5. 元座標系へ逆変換(可視化・保存用)
    corrected_original: list[np.ndarray] = []
    for i in range(n):
        inv = cv2.invertAffineTransform(reg_results[i].warp_matrix)
        corrected_original.append(reg.warp_mask(corrected[i], inv, shape_hw))

    out = {
        "corrected_masks_aligned": corrected,
        "corrected_masks_original": corrected_original,
        "qc_dataframe": qc.records_to_dataframe(records),
        "registration_results": reg_results,
    }

    if output_dir:
        outp = Path(output_dir)
        outp.mkdir(parents=True, exist_ok=True)
        out["qc_dataframe"].to_csv(outp / "roi_qc_log.csv", index=False)
        if config.save_overlays:
            for i in range(n):
                out["qc_dataframe"]
                overlay = _make_overlay(images[i], masks[i], corrected_original[i])
                cv2.imwrite(str(outp / f"frame_{i:03d}_overlay.png"), overlay)

    return out


def _make_overlay(image: np.ndarray, orig_mask: np.ndarray, corrected_mask: np.ndarray) -> np.ndarray:
    gray = pp.to_uint8_gray(image)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    orig_b = orig_mask.astype(bool)
    corr_b = corrected_mask.astype(bool)
    overlay = bgr.copy()
    overlay[orig_b & ~corr_b] = (0, 0, 255)     # 補正で削除された領域 = 赤
    overlay[corr_b & ~orig_b] = (0, 255, 0)     # 補正で追加された領域 = 緑
    overlay[orig_b & corr_b] = (0, 200, 255)    # 両方一致 = 黄
    return cv2.addWeighted(overlay, 0.5, bgr, 0.5, 0)
