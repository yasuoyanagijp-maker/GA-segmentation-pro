"""series_registration.py – 時系列用の画像位置合わせ。

各画像を基準画像（先頭時点）へ合わせる 2x3 アフィン行列を求める。

固定点（landmark）方式:
- 各画像に保存された中心窩 (fovea) と、乳頭計測ライン2端点の **中点** (disc) を
  対応点として使う。
- 中心窩＋乳頭中点の2点が両画像にある場合は、その2点を厳密に一致させる相似変換
  （回転＋等方拡大＋平行移動）を求める（"landmark" および affine/rigid/translation
  でも乳頭ラインありなら同じ2点固定を優先）。
- 中心窩しかない場合は、ECC で回転・拡大・せん断を推定した上で、変換後の
  中心窩が基準中心窩に完全一致するよう平行移動で補正する。

画像だけでなくオーバーレイ用マスクや中心窩・乳頭座標も同じ行列で移動する。
位置合わせ手法は landmark / affine / rigid / translation / fovea から選べる。

行列の向き（規約）:
    p_ref = M @ [x_moving, y_moving, 1]
すなわち M は moving 画像座標 → 基準画像座標 への「順方向」写像。
cv2.warpAffine(moving, M, size)（WARP_INVERSE_MAP なし）で moving を基準
フレームへ整列でき、点も p_ref = M @ [x, y, 1] で移動できる。
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# UI 表示用（キー: 内部名, 値: 日本語ラベル）
REGISTRATION_METHODS: dict[str, str] = {
    "landmark": "固定点合わせ（中心窩＋乳頭）",
    "affine": "アフィン＋中心窩固定（回転/拡大/せん断）",
    "rigid": "剛体＋中心窩固定（回転＋平行移動）",
    "translation": "平行移動＋中心窩固定",
    "fovea": "中心窩合わせのみ（画像照合なし）",
}

_MOTION_TYPES = {
    "affine": cv2.MOTION_AFFINE,
    "rigid": cv2.MOTION_EUCLIDEAN,
    "translation": cv2.MOTION_TRANSLATION,
}


def build_gaussian_pyramid(img: np.ndarray, min_size: int = 64) -> list[np.ndarray]:
    pyramid = [img]
    while True:
        h, w = pyramid[-1].shape[:2]
        if h <= min_size * 2 or w <= min_size * 2:
            break
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def _ecc_inverse_map(ref: np.ndarray, moving: np.ndarray, motion_type: int) -> np.ndarray:
    """多重解像度 ECC で ref(template)→moving(input) の 2x3 行列を返す。

    返り値は WARP_INVERSE_MAP と組み合わせて moving を ref フレームへ整列する
    向き（＝ ref 座標 → moving 座標）。
    """
    ref_f = ref.astype(np.float32)
    mov_f = moving.astype(np.float32)
    ref_pyr = build_gaussian_pyramid(ref_f, min_size=64)
    mov_pyr = build_gaussian_pyramid(mov_f, min_size=64)
    levels = min(len(ref_pyr), len(mov_pyr))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)

    s = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    s_inv = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    warp = np.eye(2, 3, dtype=np.float32)
    for level_idx, level_res in enumerate(reversed(range(levels))):
        curr_target = ref_pyr[level_res]
        curr_source = mov_pyr[level_res]
        try:
            if level_idx == 0:
                res = cv2.matchTemplate(curr_source, curr_target, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > 0.4:
                    warp[0, 2] = float(max_loc[0])
                    warp[1, 2] = float(max_loc[1])
            _, warp = cv2.findTransformECC(
                curr_target, curr_source, warp, motion_type, criteria,
            )
        except cv2.error:
            logger.warning("ECC failed at level %s; keeping previous estimate.", level_res)

        if level_res > 0:
            m_3x3 = np.vstack([warp.astype(np.float64), [0.0, 0.0, 1.0]])
            warp = (s @ m_3x3 @ s_inv)[:2, :].astype(np.float32)

    return warp


def _invert_affine(mat_2x3: np.ndarray) -> np.ndarray:
    m_3x3 = np.vstack([mat_2x3.astype(np.float64), [0.0, 0.0, 1.0]])
    return np.linalg.inv(m_3x3)[:2, :].astype(np.float32)


def _apply_point(mat_2x3: np.ndarray, xy: tuple[float, float]) -> tuple[float, float]:
    p = mat_2x3.astype(np.float64) @ np.array([xy[0], xy[1], 1.0])
    return float(p[0]), float(p[1])


def _translation_matrix(dx: float, dy: float) -> np.ndarray:
    m = np.eye(2, 3, dtype=np.float32)
    m[0, 2] = float(dx)
    m[1, 2] = float(dy)
    return m


def _similarity_from_two_points(
    moving_pts: tuple[tuple[float, float], tuple[float, float]],
    ref_pts: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    """2 組の対応点から相似変換（回転＋等方拡大＋平行移動）の順方向行列を返す。

    2 点を厳密に moving→ref へ写す（固定点として一致させる）。
    """
    m0 = np.array(moving_pts[0], dtype=np.float64)
    m1 = np.array(moving_pts[1], dtype=np.float64)
    r0 = np.array(ref_pts[0], dtype=np.float64)
    r1 = np.array(ref_pts[1], dtype=np.float64)

    vm = m1 - m0
    vr = r1 - r0
    nm = float(np.hypot(vm[0], vm[1]))
    if nm < 1e-6:
        # 2 点が縮退 → 中心窩だけ平行移動
        return _translation_matrix(r0[0] - m0[0], r0[1] - m0[1])

    scale = float(np.hypot(vr[0], vr[1])) / nm
    ang = math.atan2(vr[1], vr[0]) - math.atan2(vm[1], vm[0])
    c = math.cos(ang) * scale
    s = math.sin(ang) * scale
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    t = r0 - rot @ m0
    return np.array(
        [[c, -s, t[0]], [s, c, t[1]]], dtype=np.float32,
    )


def _landmark_pair_available(
    moving_fovea: tuple[float, float] | None,
    moving_disc: tuple[float, float] | None,
    ref_fovea: tuple[float, float] | None,
    ref_disc: tuple[float, float] | None,
) -> bool:
    """中心窩＋乳頭ライン中点の2組が揃っているか。"""
    return (
        moving_fovea is not None and ref_fovea is not None
        and moving_disc is not None and ref_disc is not None
    )


def compute_forward_matrix(
    moving_img: np.ndarray,
    ref_img: np.ndarray,
    *,
    method: str = "affine",
    moving_fovea: tuple[float, float] | None = None,
    ref_fovea: tuple[float, float] | None = None,
    moving_disc: tuple[float, float] | None = None,
    ref_disc: tuple[float, float] | None = None,
    anchor_fovea: bool = True,
) -> np.ndarray:
    """moving → ref の順方向 2x3 アフィン行列を返す。

    moving_disc / ref_disc は乳頭計測ライン2端点の **中点**（registration 固定点）。

    - 中心窩＋乳頭中点が両画像にあれば、method が "fovea" 以外は **2点対応の
      相似変換**で両固定点を厳密一致（landmark / affine / rigid / translation 共通）。
    - "fovea": 中心窩のみ平行移動。
    - 乳頭なし: ECC + 中心窩アンカー（affine 等）または中心窩平行移動（landmark）。
    """
    have_pair = _landmark_pair_available(
        moving_fovea, moving_disc, ref_fovea, ref_disc,
    )

    if method == "fovea":
        if moving_fovea is not None and ref_fovea is not None:
            return _translation_matrix(
                ref_fovea[0] - moving_fovea[0], ref_fovea[1] - moving_fovea[1],
            )
        return np.eye(2, 3, dtype=np.float32)

    # 乳頭ラインを引いた画像同士: 中心窩＋乳頭中点の2 corresponding points
    if have_pair:
        return _similarity_from_two_points(
            (moving_fovea, moving_disc), (ref_fovea, ref_disc),
        )

    if method == "landmark":
        if moving_fovea is not None and ref_fovea is not None:
            return _translation_matrix(
                ref_fovea[0] - moving_fovea[0], ref_fovea[1] - moving_fovea[1],
            )
        return np.eye(2, 3, dtype=np.float32)

    motion_type = _MOTION_TYPES.get(method, cv2.MOTION_AFFINE)
    inv_map = _ecc_inverse_map(ref_img, moving_img, motion_type)
    forward = _invert_affine(inv_map)

    if anchor_fovea and moving_fovea is not None and ref_fovea is not None:
        mapped = _apply_point(forward, moving_fovea)
        forward = forward.copy()
        forward[0, 2] += float(ref_fovea[0] - mapped[0])
        forward[1, 2] += float(ref_fovea[1] - mapped[1])

    return forward


def warp_image(img: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """順方向行列 matrix で img を基準フレームへ整列（バイリニア）。"""
    w, h = size
    return cv2.warpAffine(
        img, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def warp_mask(mask: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """順方向行列 matrix でマスクを整列（最近傍で二値を保持）。"""
    w, h = size
    return cv2.warpAffine(
        mask, matrix, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def warp_point(
    xy: tuple[float, float] | None, matrix: np.ndarray,
) -> tuple[float, float] | None:
    if xy is None:
        return None
    return _apply_point(matrix, xy)


def default_method_for(
    moving_fovea: tuple[float, float] | None,
    moving_disc: tuple[float, float] | None,
    ref_fovea: tuple[float, float] | None,
    ref_disc: tuple[float, float] | None,
) -> str:
    """固定点の有無から既定手法を決める。

    中心窩＋乳頭ライン中点が両画像にあれば "landmark"（2点固定）、そうでなければ
    "affine"（画像照合＋中心窩固定）。
    """
    if _landmark_pair_available(moving_fovea, moving_disc, ref_fovea, ref_disc):
        return "landmark"
    return "affine"


def register_moving_to_reference(
    moving_img: np.ndarray,
    moving_mask: np.ndarray,
    ref_img: np.ndarray,
    *,
    method: str = "affine",
    moving_fovea: tuple[float, float] | None = None,
    ref_fovea: tuple[float, float] | None = None,
    moving_disc: tuple[float, float] | None = None,
    ref_disc: tuple[float, float] | None = None,
    anchor_fovea: bool = True,
) -> dict:
    """1 枚を基準へ位置合わせし、整列後の画像・マスク・中心窩・乳頭・行列を返す。

    Returns dict: image, mask, fovea_xy, disc_xy, matrix, method
    """
    h, w = ref_img.shape[:2]
    matrix = compute_forward_matrix(
        moving_img, ref_img,
        method=method,
        moving_fovea=moving_fovea,
        ref_fovea=ref_fovea,
        moving_disc=moving_disc,
        ref_disc=ref_disc,
        anchor_fovea=anchor_fovea,
    )
    return {
        "image": warp_image(moving_img, matrix, (w, h)),
        "mask": warp_mask(moving_mask, matrix, (w, h)),
        "fovea_xy": warp_point(moving_fovea, matrix),
        "disc_xy": warp_point(moving_disc, matrix),
        "matrix": matrix,
        "method": method,
    }
