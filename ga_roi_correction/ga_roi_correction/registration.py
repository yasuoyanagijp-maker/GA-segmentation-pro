"""
registration.py
----------------
フレーム間画像登録(SOP 7章)。

方式: OpenCV の ECC(Enhanced Correlation Coefficient)アルゴリズムによる
Euclidean(剛体) -> 失敗時 Affine への段階的フォールバック。
ECCは強度ベースの登録で、FAFのような単一モダリティ・小変位(同一眼の
経時撮影)に対して頑健かつ追加学習不要という利点がある。特徴点ベース
(ORB/SIFT+RANSAC)よりも低コントラスト領域で安定しやすい。

対外的な代替: 症例数が多い/back-to-backでない撮影がある場合は
SimpleITK (elastix) の rigid+bspline registration に置き換え可能。
本モジュールは依存を OpenCV のみに絞った軽量版。
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class RegistrationResult:
    success: bool
    warp_matrix: np.ndarray          # 2x3 (Euclidean/Affine)
    mode: str                        # "euclidean" or "affine" or "identity"
    cc: float                        # 相関係数(信頼度の目安)


def _run_ecc(fixed: np.ndarray, moving: np.ndarray, motion_type: int,
             n_iters: int = 200, eps: float = 1e-6) -> tuple[bool, np.ndarray, float]:
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iters, eps)
    try:
        cc, warp_matrix = cv2.findTransformECC(
            fixed.astype(np.float32), moving.astype(np.float32),
            warp_matrix, motion_type, criteria, None, 5,
        )
        return True, warp_matrix, float(cc)
    except cv2.error:
        return False, warp_matrix, -1.0


def register_frame(fixed: np.ndarray, moving: np.ndarray,
                    min_cc_euclidean: float = 0.6) -> RegistrationResult:
    """
    moving を fixed(基準フレーム、通常は前フレームまたは初回フレーム)へ登録する。
    まず剛体(Euclidean: 回転+並進)、相関が低ければアフィンへ拡張。
    双方失敗した場合は identity(登録なし)とし、低信頼フラグを立てる前提。
    """
    ok, warp, cc = _run_ecc(fixed, moving, cv2.MOTION_EUCLIDEAN)
    if ok and cc >= min_cc_euclidean:
        return RegistrationResult(True, warp, "euclidean", cc)

    ok2, warp2, cc2 = _run_ecc(fixed, moving, cv2.MOTION_AFFINE)
    if ok2 and cc2 >= min_cc_euclidean:
        return RegistrationResult(True, warp2, "affine", cc2)

    # どちらも信頼できない場合は identity を返し、呼び出し側で低信頼フラグ扱いにする
    best_cc = max(cc, cc2)
    return RegistrationResult(False, np.eye(2, 3, dtype=np.float32), "identity", best_cc)


def warp_image(image: np.ndarray, warp_matrix: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return cv2.warpAffine(
        image, warp_matrix, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def warp_mask(mask: np.ndarray, warp_matrix: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    warped = cv2.warpAffine(
        (mask.astype(np.uint8) * 255), warp_matrix, (w, h),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return (warped > 127).astype(np.uint8)
