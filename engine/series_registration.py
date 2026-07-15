"""series_registration.py – 時系列用の画像位置合わせ。

各画像を基準画像（先頭時点）へ合わせる 2x3 アフィン行列を求める。

固定点（landmark）方式:
- 各画像に保存された中心窩 (fovea) と、乳頭計測ライン2端点の **中点** (disc) を
  対応点として使う。
- 中心窩＋乳頭中点の2点が両画像にある場合は、2点相似変換を **初期値** とし、
  左半分クロップ（2眼キャプチャ時）・CLAHE・血管強調・UI 除外マスクの上で
  SIFT + RANSAC でリファインする。ズーム差はスケールスイープ＋位相相関で推定。
  失敗時は ECC → 2 点固定の順でフォールバック。
- 中心窩しかない場合は、SIFT → ECC の順で推定し、必要なら中心窩を平行移動で
  補正する。

画像だけでなくオーバーレイ用マスクや中心窩・乳頭座標も同じ行列で移動する。
位置合わせ手法は landmark / sift / affine / rigid / translation / fovea から選べる。

行列の向き（規約）:
    p_ref = M @ [x_moving, y_moving, 1]
すなわち M は moving 画像座標 → 基準画像座標 への「順方向」写像。
cv2.warpAffine(moving, M, size)（WARP_INVERSE_MAP なし）で moving を基準
フレームへ整列でき、点も p_ref = M @ [x, y, 1] で移動できる。
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import contextmanager

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ステージ別の実行時間ログ（ボトルネック調査用）。
# flet_ui の "ga.batch" と同様に、アプリ側で logging 未設定でも
# コンソールに出るよう専用ハンドラを付ける。
perf_logger = logging.getLogger("ga.reg.timing")
if not perf_logger.handlers:
    _perf_handler = logging.StreamHandler()
    _perf_handler.setFormatter(logging.Formatter("[reg-timing] %(message)s"))
    perf_logger.addHandler(_perf_handler)
    perf_logger.setLevel(logging.INFO)
    perf_logger.propagate = False


@contextmanager
def _perf(stage: str, detail: str = ""):
    """ステージ実行時間を INFO で記録するコンテキストマネージャ。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - t0) * 1000.0
        if detail:
            perf_logger.info("%s (%s): %.0f ms", stage, detail, ms)
        else:
            perf_logger.info("%s: %.0f ms", stage, ms)

# UI 表示用（キー: 内部名, 値: 日本語ラベル）
REGISTRATION_METHODS: dict[str, str] = {
    "landmark": "固定点＋SIFT（中心窩＋乳頭）",
    "sift": "特徴点（SIFT）＋RANSAC",
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

_CLAHE_CLIP = 2.0
_CLAHE_TILE = (8, 8)
_SIFT_RATIO = 0.72
_SIFT_MIN_MATCHES = 10
_SIFT_MIN_INLIERS = 10
_RANSAC_REPROJ_THRESH = 4.0
_SIDE_BY_SIDE_RATIO = 1.45
_SIFT_NFEATURES = 6000
_ZOOM_SCALE_MIN = 0.82
_ZOOM_SCALE_MAX = 1.18
_ZOOM_ANGLE_FINE = 10.0
_ZOOM_ANGLE_WIDE = 25.0
_ZOOM_MAE_HIGH = 28.0
_SCALE_PENALTY_WEIGHT = 18.0
# 候補の MAE が zoom より明らかに良いときはスケール罰則を免除（SIFT/ECC が勝てるように）
_SCALE_PENALTY_MAE_WAIVE = 0.0
_AFFINE_DEVIATION_WEIGHT = 35.0
# 二段 ECC の縮小段の長辺（原寸段の初期値づくり）
_ECC_COARSE_MAX_SIDE = 768


def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _crop_for_registration(img: np.ndarray) -> np.ndarray:
    """OPTOS 等の左右2眼キャプチャでは左半分（第1眼）だけを使う。"""
    gray = _to_gray_uint8(img)
    h, w = gray.shape[:2]
    if w / max(h, 1) >= _SIDE_BY_SIDE_RATIO:
        return gray[:, : w // 2]
    return gray


def _feature_detection_mask(gray: np.ndarray) -> np.ndarray:
    """黒枠・UI 文字帯を除いた SIFT / ECC 用マスク。"""
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    y0, y1 = int(h * 0.05), int(h * 0.93)
    x0, x1 = int(w * 0.03), int(w * 0.97)
    mask[y0:y1, x0:x1] = 255
    content = (gray > 30).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, content)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _prepare_registration_pair(
    ref_img: np.ndarray,
    moving_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Registration 用グレースケールペア（2眼クロップ後、共通サイズに切り揃え）。"""
    ref_g = _crop_for_registration(ref_img)
    mov_g = _crop_for_registration(moving_img)
    h = min(ref_g.shape[0], mov_g.shape[0])
    w = min(ref_g.shape[1], mov_g.shape[1])
    return ref_g[:h, :w], mov_g[:h, :w]


class _PairContext:
    """1ペアの登録処理内で共有する前処理・マスク・縮小版のキャッシュ。

    同じ ref/mov クロップに対する CLAHE＋血管強調・特徴マスク・縮小版・
    SIFT 特徴量（ref 側）は候補生成の各段で完全に同一なので、1回だけ
    計算して使い回す。計算内容は従来のその場計算と同一（結果は不変）。
    """

    def __init__(self, ref_crop: np.ndarray, mov_crop: np.ndarray):
        self.ref_crop = ref_crop
        self.mov_crop = mov_crop
        self.ref_pp = _preprocess_for_registration(ref_crop)
        self.mov_pp = _preprocess_for_registration(mov_crop)
        self.ref_pp_f32 = self.ref_pp.astype(np.float32)
        self.mov_pp_f32 = self.mov_pp.astype(np.float32)
        # グレークロップ由来（MAE / ECC / 位相相関 / zoom sweep 用）
        self.mask_crop = _feature_detection_mask(ref_crop)
        # 前処理画像由来（SIFT 用 — 従来どおり ref_pp から計算）
        self.mask_pp = _feature_detection_mask(self.ref_pp)
        # zoom sweep / 縮小 MAE 共通の縮小ペア（max_side=320）
        self.ref_s, self.mov_s, self.mask_s, self.factor = _downscale_pair(
            self.ref_pp_f32, self.mov_pp_f32, self.mask_crop,
        )
        self._sift = None
        self._ref_features: tuple | None = None
        self._matched_points: dict[bytes, tuple[np.ndarray, np.ndarray] | None] = {}

    @property
    def sift(self) -> cv2.SIFT:
        if self._sift is None:
            self._sift = cv2.SIFT_create(
                nfeatures=_SIFT_NFEATURES,
                contrastThreshold=0.04,
                edgeThreshold=15,
            )
        return self._sift

    def ref_features(self) -> tuple:
        """ref 側 SIFT keypoints/descriptors（初期値によらず不変なので1回だけ）。"""
        if self._ref_features is None:
            self._ref_features = self.sift.detectAndCompute(
                self.ref_pp, self.mask_pp,
            )
        return self._ref_features

    def matched_points(
        self, initial: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """初期値ごとの SIFT 対応点 (pts_mov, pts_ref)。

        対応点はモデル（similarity / affine）に依存しないため、同じ初期値で
        両モデルを推定するときの検出・マッチング重複を除く。
        不足時（特徴点 <4 または good マッチ < _SIFT_MIN_MATCHES）は None。
        """
        key = b"none" if initial is None else (
            np.asarray(initial, np.float32).tobytes()
        )
        if key not in self._matched_points:
            self._matched_points[key] = self._compute_matched_points(initial)
        return self._matched_points[key]

    def _compute_matched_points(
        self, initial: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        kp_ref, desc_ref = self.ref_features()
        mov_pp = self.mov_pp
        if initial is not None:
            h, w = self.ref_pp.shape[:2]
            mov_pp = cv2.warpAffine(mov_pp, initial, (w, h))
        kp_mov, desc_mov = self.sift.detectAndCompute(mov_pp, self.mask_pp)
        if (
            desc_ref is None or desc_mov is None
            or len(kp_ref) < 4 or len(kp_mov) < 4
        ):
            return None
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = matcher.knnMatch(desc_mov, desc_ref, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < _SIFT_RATIO * second.distance:
                good.append(best)
        if len(good) < _SIFT_MIN_MATCHES:
            return None
        pts_mov = np.float32(
            [kp_mov[m.queryIdx].pt for m in good],
        ).reshape(-1, 1, 2)
        pts_ref = np.float32(
            [kp_ref[m.trainIdx].pt for m in good],
        ).reshape(-1, 1, 2)
        return pts_mov, pts_ref


def _similarity_about_center(
    scale: float,
    angle_deg: float,
    center: tuple[float, float],
    dx: float = 0.0,
    dy: float = 0.0,
) -> np.ndarray:
    """中心基準の相似変換（moving→ref）2x3 行列。"""
    mat = cv2.getRotationMatrix2D(center, angle_deg, scale)
    mat = mat.astype(np.float32)
    mat[0, 2] += float(dx)
    mat[1, 2] += float(dy)
    return mat


def _matrix_similarity_scale(mat: np.ndarray) -> float:
    a, b = float(mat[0, 0]), float(mat[1, 0])
    return float(math.hypot(a, b))


def _matrix_affine_deviation(mat: np.ndarray) -> float:
    """相似変換からの逸脱（せん断・異方スケール）。0 なら等方回転＋拡大。"""
    a, b = float(mat[0, 0]), float(mat[1, 0])
    c, d = float(mat[0, 1]), float(mat[1, 1])
    s0 = math.hypot(a, b)
    s1 = math.hypot(c, d)
    if s0 < 1e-6 and s1 < 1e-6:
        return 0.0
    shear = abs(a * c + b * d)
    aniso = abs(s0 - s1)
    return shear + aniso


def _alignment_mae_downscaled(
    ref_crop: np.ndarray,
    mov_crop: np.ndarray,
    matrix: np.ndarray,
    *,
    ctx: "_PairContext | None" = None,
) -> float:
    """スイープ用: 縮小画像で MAE を計算（zoom 候補の選別向け、max_side=320）。"""
    if ctx is None:
        ctx = _PairContext(ref_crop, mov_crop)
    ref_s, mov_s, mask_s, factor = ctx.ref_s, ctx.mov_s, ctx.mask_s, ctx.factor
    if factor < 0.999:
        mat = matrix.copy().astype(np.float64)
        mat[0, 2] *= factor
        mat[1, 2] *= factor
    else:
        mat = matrix.astype(np.float64)
    aligned = cv2.warpAffine(mov_s, mat.astype(np.float32), (ref_s.shape[1], ref_s.shape[0]))
    valid = mask_s > 0
    diff = np.abs(ref_s - aligned)
    if valid.any():
        return float(diff[valid].mean())
    return float(diff.mean())


def _downscale_pair(
    ref: np.ndarray,
    mov: np.ndarray,
    mask: np.ndarray,
    *,
    max_side: int = 320,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """スイープ用に縮小。返り値の scale_factor は 縮小/原寸。"""
    h, w = ref.shape[:2]
    factor = min(1.0, float(max_side) / float(max(h, w)))
    if factor >= 0.999:
        return ref, mov, mask, 1.0
    size = (max(1, int(round(w * factor))), max(1, int(round(h * factor))))
    return (
        cv2.resize(ref, size, interpolation=cv2.INTER_AREA),
        cv2.resize(mov, size, interpolation=cv2.INTER_AREA),
        cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST),
        factor,
    )


def _zoom_scale_sweep_initial(
    ref_crop: np.ndarray,
    mov_crop: np.ndarray,
    *,
    ctx: "_PairContext | None" = None,
) -> np.ndarray | None:
    """ズーム差向け: スケール×回転スイープ＋位相相関で相似変換の初期推定。

    OPTOS 等で撮影倍率が枚ごとに異なる場合（例: 4.0x vs 4.6x）に有効。
    位相相関の response だけでは誤ったスケールを選ぶことがあるため、
    候補ごとに縮小 MAE でも評価し、大ズレ時は回転探索幅を広げる。
    """
    if ctx is None:
        ctx = _PairContext(ref_crop, mov_crop)
    ref_pp = ctx.ref_pp_f32
    mov_pp = ctx.mov_pp_f32
    mask_full = ctx.mask_crop
    window_full = mask_full.astype(np.float64) / 255.0
    if float(window_full.sum()) < 100.0:
        return None

    ref_s, mov_s, mask_s, factor = ctx.ref_s, ctx.mov_s, ctx.mask_s, ctx.factor
    window = mask_s.astype(np.float64) / 255.0
    hs, ws = ref_s.shape[:2]
    center_s = (ws * 0.5, hs * 0.5)
    h, w = ref_pp.shape[:2]
    center = (w * 0.5, h * 0.5)
    inv_f = 1.0 / factor if factor > 1e-9 else 1.0

    def _score_similarity(
        scale: float,
        angle: float,
        dx_s: float,
        dy_s: float,
        resp: float,
    ) -> tuple[float, np.ndarray]:
        cand = _similarity_about_center(
            scale, angle, center,
            dx_s * inv_f, dy_s * inv_f,
        )
        mae = _alignment_mae_downscaled(ref_crop, mov_crop, cand, ctx=ctx)
        # response は補助: MAE が同等なら位相相関が高い方を優先
        score = mae - 2.0 * min(resp, 0.35)
        return score, cand

    best_score = float("inf")
    best_scale = 1.0
    best_angle = 0.0
    best_dx_s = 0.0
    best_dy_s = 0.0
    best_resp = -1.0
    best_mae = float("inf")

    coarse_scales = np.linspace(_ZOOM_SCALE_MIN, _ZOOM_SCALE_MAX, 37)
    for scale in coarse_scales:
        mat = _similarity_about_center(float(scale), 0.0, center_s)
        warped = cv2.warpAffine(mov_s, mat, (ws, hs))
        try:
            (dx, dy), resp = cv2.phaseCorrelate(
                ref_s.astype(np.float64),
                warped.astype(np.float64),
                window,
            )
        except cv2.error:
            continue
        if resp < 0.05:
            continue
        score, _ = _score_similarity(float(scale), 0.0, float(dx), float(dy), float(resp))
        mae_only = score + 2.0 * min(float(resp), 0.35)
        if score < best_score:
            best_score = score
            best_scale = float(scale)
            best_angle = 0.0
            best_dx_s = float(dx)
            best_dy_s = float(dy)
            best_resp = float(resp)
            best_mae = mae_only

    if best_resp < 0.05:
        return None

    fine_scales = np.linspace(
        max(_ZOOM_SCALE_MIN, best_scale - 0.05),
        min(_ZOOM_SCALE_MAX, best_scale + 0.05),
        21,
    )
    angle_span = _ZOOM_ANGLE_WIDE if best_mae >= _ZOOM_MAE_HIGH else _ZOOM_ANGLE_FINE
    angle_steps = 25 if angle_span >= _ZOOM_ANGLE_WIDE else 11
    for scale in fine_scales:
        for angle in np.linspace(-angle_span, angle_span, angle_steps):
            mat = _similarity_about_center(float(scale), float(angle), center_s)
            warped = cv2.warpAffine(mov_s, mat, (ws, hs))
            try:
                (dx, dy), resp = cv2.phaseCorrelate(
                    ref_s.astype(np.float64),
                    warped.astype(np.float64),
                    window,
                )
            except cv2.error:
                continue
            if resp < 0.05:
                continue
            score, _ = _score_similarity(
                float(scale), float(angle), float(dx), float(dy), float(resp),
            )
            mae_only = score + 2.0 * min(float(resp), 0.35)
            if score < best_score:
                best_score = score
                best_scale = float(scale)
                best_angle = float(angle)
                best_dx_s = float(dx)
                best_dy_s = float(dy)
                best_resp = float(resp)
                best_mae = mae_only

    cand = _similarity_about_center(
        best_scale, best_angle, center,
        best_dx_s * inv_f, best_dy_s * inv_f,
    )
    # 原寸で位相相関を1回だけ再推定して平行移動を精緻化
    warped_full = cv2.warpAffine(mov_pp, cand, (w, h))
    try:
        (dx, dy), resp = cv2.phaseCorrelate(
            ref_pp.astype(np.float64),
            warped_full.astype(np.float64),
            window_full,
        )
        if resp >= 0.05:
            cand = _compose_forward(_translation_matrix(float(dx), float(dy)), cand)
    except cv2.error:
        pass

    logger.debug(
        "Zoom sweep: scale=%.3f angle=%.1f resp=%.3f mae=%.2f",
        best_scale, best_angle, best_resp, best_mae,
    )
    return cand


def _preprocess_for_registration(gray: np.ndarray) -> np.ndarray:
    """Registration 用 CLAHE + 血管強調（特徴点抽出・ECC 共通）。"""
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_TILE)
    eq = clahe.apply(_to_gray_uint8(gray))
    blur_f = cv2.GaussianBlur(eq, (0, 0), 1.2)
    blur_c = cv2.GaussianBlur(eq, (0, 0), 4.0)
    vessels = cv2.subtract(blur_f, blur_c)
    return cv2.addWeighted(eq, 0.65, vessels, 0.35, 0)


def _compose_forward(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """順方向行列の合成: p_ref = outer @ (inner @ p_mov)。"""
    mo = np.vstack([outer.astype(np.float64), [0.0, 0.0, 1.0]])
    mi = np.vstack([inner.astype(np.float64), [0.0, 0.0, 1.0]])
    return (mo @ mi)[:2, :].astype(np.float32)


def _sift_ransac_forward(
    ref_img: np.ndarray,
    moving_img: np.ndarray,
    *,
    model: str = "similarity",
    initial: np.ndarray | None = None,
    ctx: "_PairContext | None" = None,
) -> np.ndarray | None:
    """SIFT + Lowe ratio test + RANSAC で moving→ref の順方向 2x3 行列を返す。

    model: "similarity"（estimateAffinePartial2D）または "affine"。
    initial があれば moving を事前整列してから SIFT し、結果と合成する。
    ctx があれば ref 側特徴量と初期値ごとの対応点を使い回す。
    失敗時は None。
    """
    if ctx is None:
        ctx = _PairContext(ref_img, moving_img)
    matched = ctx.matched_points(initial)
    if matched is None:
        return None
    pts_mov, pts_ref = matched

    kwargs = dict(
        method=cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_REPROJ_THRESH,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )

    if model == "affine":
        mat, inliers = cv2.estimateAffine2D(pts_mov, pts_ref, **kwargs)
    else:
        mat, inliers = cv2.estimateAffinePartial2D(pts_mov, pts_ref, **kwargs)

    if mat is None or inliers is None:
        return None
    if int(np.sum(inliers)) < _SIFT_MIN_INLIERS:
        return None
    mat = mat.astype(np.float32)
    if initial is not None:
        return _compose_forward(mat, initial)
    return mat


def _landmark_initial_matrix(
    moving_fovea: tuple[float, float] | None,
    moving_disc: tuple[float, float] | None,
    ref_fovea: tuple[float, float] | None,
    ref_disc: tuple[float, float] | None,
) -> np.ndarray | None:
    if not _landmark_pair_available(moving_fovea, moving_disc, ref_fovea, ref_disc):
        return None
    return _similarity_from_two_points(
        (moving_fovea, moving_disc), (ref_fovea, ref_disc),
    )


def _rescale_ecc_warp(warp: np.ndarray, factor: float) -> np.ndarray:
    """縮小画像座標系の ECC warp を原寸座標系へ変換（平行移動を 1/factor 倍）。"""
    s = np.array(
        [[factor, 0.0, 0.0], [0.0, factor, 0.0], [0.0, 0.0, 1.0]], np.float64,
    )
    s_inv = np.array(
        [[1.0 / factor, 0.0, 0.0], [0.0, 1.0 / factor, 0.0], [0.0, 0.0, 1.0]],
        np.float64,
    )
    m = np.vstack([warp.astype(np.float64), [0.0, 0.0, 1.0]])
    return (s_inv @ m @ s)[:2, :].astype(np.float32)


def _ecc_refine_forward(
    ref_img: np.ndarray,
    moving_img: np.ndarray,
    motion_type: int,
    *,
    initial: np.ndarray | None = None,
    ctx: "_PairContext | None" = None,
) -> np.ndarray | None:
    """SIFT 失敗時: マスク付き ECC で moving→ref の順方向行列を返す。

    coarse-to-fine の二段構成:
    1. 縮小画像（長辺 _ECC_COARSE_MAX_SIDE）で収束させ、
    2. その結果を初期値に原寸で仕上げる（eps 収束なので数反復で停止）。
    原寸のみの一発 ECC と同じ criteria（反復上限・eps）を使い、
    縮小段や初期値付き原寸段が失敗した場合は従来どおり
    原寸・単位行列初期値へフォールバックする。
    """
    if ctx is None:
        ref_crop, mov_crop = _prepare_registration_pair(ref_img, moving_img)
        ctx = _PairContext(ref_crop, mov_crop)
    ref_pp = ctx.ref_pp_f32
    mov_pp = ctx.mov_pp_f32
    mask = ctx.mask_crop
    if initial is not None:
        mov_pp = cv2.warpAffine(mov_pp, initial, (ref_pp.shape[1], ref_pp.shape[0]))

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)

    init_full: np.ndarray | None = None
    ref_c, mov_c, mask_c, factor = _downscale_pair(
        ref_pp, mov_pp, mask, max_side=_ECC_COARSE_MAX_SIDE,
    )
    if factor < 0.999:
        warp_c = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp_c = cv2.findTransformECC(
                ref_c, mov_c, warp_c, motion_type, criteria,
                inputMask=mask_c,
            )
            init_full = _rescale_ecc_warp(warp_c, factor)
        except cv2.error:
            init_full = None

    inv_map: np.ndarray | None = None
    if init_full is not None:
        try:
            _, inv_map = cv2.findTransformECC(
                ref_pp, mov_pp, init_full.copy(), motion_type, criteria,
                inputMask=mask,
            )
        except cv2.error:
            inv_map = None
    if inv_map is None:
        # 従来どおりの原寸・単位行列初期値へフォールバック
        try:
            _, inv_map = cv2.findTransformECC(
                ref_pp, mov_pp, np.eye(2, 3, dtype=np.float32),
                motion_type, criteria,
                inputMask=mask,
            )
        except cv2.error:
            return initial
    forward = _invert_affine(inv_map)
    if initial is not None:
        return _compose_forward(forward, initial)
    return forward


def _anchor_fovea(
    forward: np.ndarray,
    moving_fovea: tuple[float, float] | None,
    ref_fovea: tuple[float, float] | None,
) -> np.ndarray:
    if moving_fovea is None or ref_fovea is None:
        return forward
    mapped = _apply_point(forward, moving_fovea)
    out = forward.copy()
    out[0, 2] += float(ref_fovea[0] - mapped[0])
    out[1, 2] += float(ref_fovea[1] - mapped[1])
    return out


def _phase_translation_initial(
    ref_crop: np.ndarray,
    mov_crop: np.ndarray,
    *,
    ctx: "_PairContext | None" = None,
) -> np.ndarray | None:
    """位相相関で平行移動の初期推定（小ズレ向け）。"""
    if ctx is None:
        ctx = _PairContext(ref_crop, mov_crop)
    ref_pp = ctx.ref_pp
    mov_pp = ctx.mov_pp
    window = ctx.mask_crop.astype(np.float64) / 255.0
    if float(window.sum()) < 100.0:
        return None
    try:
        (dx, dy), response = cv2.phaseCorrelate(
            ref_pp.astype(np.float64),
            mov_pp.astype(np.float64),
            window,
        )
    except cv2.error:
        return None
    if response < 0.08:
        return None
    return _translation_matrix(float(dx), float(dy))


def _alignment_mae(
    ref_crop: np.ndarray,
    mov_crop: np.ndarray,
    matrix: np.ndarray,
    *,
    ctx: "_PairContext | None" = None,
) -> float:
    """マスク領域の平均絶対差（小さいほど整列が良い）。"""
    if ctx is None:
        ctx = _PairContext(ref_crop, mov_crop)
    ref_pp = ctx.ref_pp_f32
    mov_pp = ctx.mov_pp_f32
    aligned = cv2.warpAffine(
        mov_pp, matrix, (ref_pp.shape[1], ref_pp.shape[0]),
    )
    mask = ctx.mask_crop > 0
    diff = np.abs(ref_pp - aligned)
    if mask.any():
        return float(diff[mask].mean())
    return float(diff.mean())


def _landmark_penalty(
    matrix: np.ndarray,
    *,
    moving_fovea: tuple[float, float] | None,
    ref_fovea: tuple[float, float] | None,
    moving_disc: tuple[float, float] | None,
    ref_disc: tuple[float, float] | None,
) -> float:
    err = 0.0
    if moving_fovea is not None and ref_fovea is not None:
        mapped = _apply_point(matrix, moving_fovea)
        err += math.hypot(mapped[0] - ref_fovea[0], mapped[1] - ref_fovea[1])
    if moving_disc is not None and ref_disc is not None:
        mapped = _apply_point(matrix, moving_disc)
        err += math.hypot(mapped[0] - ref_disc[0], mapped[1] - ref_disc[1])
    return err


def _alignment_score(
    ref_crop: np.ndarray,
    mov_crop: np.ndarray,
    matrix: np.ndarray,
    *,
    moving_fovea: tuple[float, float] | None = None,
    ref_fovea: tuple[float, float] | None = None,
    moving_disc: tuple[float, float] | None = None,
    ref_disc: tuple[float, float] | None = None,
    have_pair: bool = False,
    zoom_scale: float | None = None,
    zoom_mae: float | None = None,
    ctx: "_PairContext | None" = None,
) -> float:
    mae = _alignment_mae(ref_crop, mov_crop, matrix, ctx=ctx)
    score = mae
    if have_pair:
        lm_err = _landmark_penalty(
            matrix,
            moving_fovea=moving_fovea,
            ref_fovea=ref_fovea,
            moving_disc=moving_disc,
            ref_disc=ref_disc,
        )
        if lm_err > 3.0:
            score += 25.0 * lm_err
    if zoom_scale is not None and zoom_scale > 1e-6:
        cand_scale = _matrix_similarity_scale(matrix)
        scale_ratio = cand_scale / zoom_scale
        if scale_ratio > 0.0:
            waive = (
                zoom_mae is not None
                and mae < zoom_mae - _SCALE_PENALTY_MAE_WAIVE
            )
            if not waive:
                score += _SCALE_PENALTY_WEIGHT * abs(math.log(scale_ratio))
    dev = _matrix_affine_deviation(matrix)
    if dev > 0.015:
        score += _AFFINE_DEVIATION_WEIGHT * dev
    return score


def _pick_best_matrix(
    ref_crop: np.ndarray,
    mov_crop: np.ndarray,
    candidates: list[np.ndarray | None],
    *,
    moving_fovea: tuple[float, float] | None = None,
    ref_fovea: tuple[float, float] | None = None,
    moving_disc: tuple[float, float] | None = None,
    ref_disc: tuple[float, float] | None = None,
    have_pair: bool = False,
    zoom_scale: float | None = None,
    zoom_mae: float | None = None,
    ctx: "_PairContext | None" = None,
) -> np.ndarray:
    best = np.eye(2, 3, dtype=np.float32)
    best_score = float("inf")
    seen: set[tuple[float, ...]] = set()
    for mat in candidates:
        if mat is None:
            continue
        key = tuple(np.round(mat.ravel(), 2).tolist())
        if key in seen:
            continue
        seen.add(key)
        score = _alignment_score(
            ref_crop, mov_crop, mat,
            moving_fovea=moving_fovea,
            ref_fovea=ref_fovea,
            moving_disc=moving_disc,
            ref_disc=ref_disc,
            have_pair=have_pair,
            zoom_scale=zoom_scale,
            zoom_mae=zoom_mae,
            ctx=ctx,
        )
        if score < best_score:
            best_score = score
            best = mat.astype(np.float32)
    return best


def _ecc_motion_for_method(method: str, *, allow_scale: bool) -> int:
    """ズーム差がある候補ではスケールを含む MOTION_AFFINE を使う。

    OpenCV ECC に相似変換（等方スケール）専用モードはないため、
    スケール保持が必要な landmark/sift でも AFFINE を選ぶ。
    """
    if method == "translation":
        return cv2.MOTION_TRANSLATION
    if method == "rigid" and not allow_scale:
        return cv2.MOTION_EUCLIDEAN
    if method in ("landmark", "sift", "affine") or allow_scale:
        return cv2.MOTION_AFFINE
    if method in _MOTION_TYPES:
        return _MOTION_TYPES[method]
    return cv2.MOTION_AFFINE


def _image_registration_candidates(
    ref_img: np.ndarray,
    moving_img: np.ndarray,
    *,
    method: str,
    initial: np.ndarray | None,
    moving_fovea: tuple[float, float] | None = None,
    ref_fovea: tuple[float, float] | None = None,
    moving_disc: tuple[float, float] | None = None,
    ref_disc: tuple[float, float] | None = None,
    have_pair: bool = False,
) -> np.ndarray:
    """複数推定を試し、マスク領域 MAE が最小の行列を返す。"""
    ref_c, mov_c = _prepare_registration_pair(ref_img, moving_img)
    with _perf("preprocess_cache"):
        ctx = _PairContext(ref_c, mov_c)

    candidates: list[np.ndarray | None] = [np.eye(2, 3, dtype=np.float32)]

    with _perf("zoom_sweep"):
        zoom = _zoom_scale_sweep_initial(ref_c, mov_c, ctx=ctx)
    zoom_scale = _matrix_similarity_scale(zoom) if zoom is not None else None
    zoom_mae = (
        _alignment_mae(ref_c, mov_c, zoom, ctx=ctx) if zoom is not None else None
    )
    if zoom is not None:
        candidates.append(zoom)

    with _perf("phase_translation"):
        phase = _phase_translation_initial(ref_c, mov_c, ctx=ctx)
    if phase is not None:
        candidates.append(phase)
    if initial is not None:
        candidates.append(initial)
    elif (
        method == "landmark"
        and moving_fovea is not None and ref_fovea is not None
    ):
        candidates.append(_translation_matrix(
            ref_fovea[0] - moving_fovea[0], ref_fovea[1] - moving_fovea[1],
        ))

    allow_scale = (
        method in ("landmark", "sift", "affine")
        or (zoom is not None and abs(_matrix_similarity_scale(zoom) - 1.0) > 0.02)
        or (initial is not None and abs(_matrix_similarity_scale(initial) - 1.0) > 0.02)
    )
    motion = _ecc_motion_for_method(method, allow_scale=allow_scale)

    sift_inits: list[np.ndarray | None] = []
    for init in (initial, zoom, phase):
        if init is not None and not any(np.allclose(init, x) for x in sift_inits if x is not None):
            sift_inits.append(init)
    if not sift_inits:
        sift_inits = [None]

    for i, init in enumerate(sift_inits):
        with _perf("sift_similarity", f"init {i + 1}/{len(sift_inits)}"):
            sift = _sift_ransac_forward(
                ref_c, mov_c,
                model="similarity",
                initial=init,
                ctx=ctx,
            )
        if sift is not None:
            candidates.append(sift)

    # affine は landmark/sift 候補が弱いときだけ追加（wild affine を避ける）
    if method == "affine":
        for i, init in enumerate(sift_inits):
            with _perf("sift_affine", f"init {i + 1}/{len(sift_inits)}"):
                sift_a = _sift_ransac_forward(
                    ref_c, mov_c,
                    model="affine",
                    initial=init,
                    ctx=ctx,
                )
            if sift_a is not None:
                candidates.append(sift_a)

    ecc_inits: list[np.ndarray | None] = []
    for init in (initial, zoom, phase):
        if init is not None and not any(np.allclose(init, x) for x in ecc_inits if x is not None):
            ecc_inits.append(init)
    if not ecc_inits:
        ecc_inits = [None]

    for i, init in enumerate(ecc_inits):
        with _perf("ecc_refine", f"init {i + 1}/{len(ecc_inits)}"):
            ecc = _ecc_refine_forward(
                ref_img, moving_img, motion, initial=init, ctx=ctx,
            )
        if ecc is not None:
            candidates.append(ecc)

    with _perf("ecc_pyramid"):
        inv_map = _ecc_inverse_map(ref_img, moving_img, motion, ctx=ctx)
    candidates.append(_invert_affine(inv_map))

    with _perf("pick_best", f"{len(candidates)} candidates"):
        return _pick_best_matrix(
            ref_c, mov_c, candidates,
            moving_fovea=moving_fovea,
            ref_fovea=ref_fovea,
            moving_disc=moving_disc,
            ref_disc=ref_disc,
            have_pair=have_pair,
            zoom_scale=zoom_scale,
            zoom_mae=zoom_mae,
            ctx=ctx,
        )


def build_gaussian_pyramid(img: np.ndarray, min_size: int = 64) -> list[np.ndarray]:
    pyramid = [img]
    while True:
        h, w = pyramid[-1].shape[:2]
        if h <= min_size * 2 or w <= min_size * 2:
            break
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def _ecc_inverse_map(
    ref: np.ndarray,
    moving: np.ndarray,
    motion_type: int,
    *,
    ctx: "_PairContext | None" = None,
) -> np.ndarray:
    """多重解像度 ECC で ref(template)→moving(input) の 2x3 行列を返す。

    返り値は WARP_INVERSE_MAP と組み合わせて moving を ref フレームへ整列する
    向き（＝ ref 座標 → moving 座標）。
    """
    if ctx is None:
        ref_crop, mov_crop = _prepare_registration_pair(ref, moving)
        ctx = _PairContext(ref_crop, mov_crop)
    mask = ctx.mask_crop
    ref_f = ctx.ref_pp_f32
    mov_f = ctx.mov_pp_f32
    ref_pyr = build_gaussian_pyramid(ref_f, min_size=64)
    mov_pyr = build_gaussian_pyramid(mov_f, min_size=64)
    levels = min(len(ref_pyr), len(mov_pyr))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)

    s = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    s_inv = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    warp = np.eye(2, 3, dtype=np.float32)
    level_mask = mask
    for level_idx, level_res in enumerate(reversed(range(levels))):
        curr_target = ref_pyr[level_res]
        curr_source = mov_pyr[level_res]
        if level_res > 0:
            level_mask = cv2.resize(
                mask, (curr_target.shape[1], curr_target.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        try:
            if level_idx == 0:
                res = cv2.matchTemplate(curr_source, curr_target, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > 0.4:
                    warp[0, 2] = float(max_loc[0])
                    warp[1, 2] = float(max_loc[1])
            _, warp = cv2.findTransformECC(
                curr_target, curr_source, warp, motion_type, criteria,
                inputMask=level_mask,
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

    - "landmark": 2点相似を初期値に SIFT+RANSAC でリファイン（失敗時 2点固定）。
    - "sift" / affine / rigid / translation: SIFT → ECC →（必要なら）中心窩アンカー。
    - 乳頭ラインありの affine/rigid/translation も landmark と同様に 2点初期化。
    - "fovea": 中心窩のみ平行移動。
    """
    have_pair = _landmark_pair_available(
        moving_fovea, moving_disc, ref_fovea, ref_disc,
    )
    initial = _landmark_initial_matrix(
        moving_fovea, moving_disc, ref_fovea, ref_disc,
    )

    if method == "fovea":
        if moving_fovea is not None and ref_fovea is not None:
            return _translation_matrix(
                ref_fovea[0] - moving_fovea[0], ref_fovea[1] - moving_fovea[1],
            )
        return np.eye(2, 3, dtype=np.float32)

    if method in ("landmark", "sift", "affine", "rigid", "translation"):
        with _perf("compute_forward_matrix TOTAL", f"method={method}"):
            best = _image_registration_candidates(
                ref_img, moving_img,
                method=method,
                initial=initial,
                moving_fovea=moving_fovea,
                ref_fovea=ref_fovea,
                moving_disc=moving_disc,
                ref_disc=ref_disc,
                have_pair=have_pair,
            )
        if anchor_fovea and method != "sift" and not have_pair:
            best = _anchor_fovea(best, moving_fovea, ref_fovea)
        return best

    motion_type = _MOTION_TYPES.get(method, cv2.MOTION_AFFINE)
    with _perf("compute_forward_matrix TOTAL", f"method={method} (ecc only)"):
        inv_map = _ecc_inverse_map(ref_img, moving_img, motion_type)
    forward = _invert_affine(inv_map)

    if anchor_fovea and moving_fovea is not None and ref_fovea is not None:
        forward = _anchor_fovea(forward, moving_fovea, ref_fovea)

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

    中心窩＋乳頭ライン中点が両画像にあれば "landmark"（2点初期化＋SIFT）、
    そうでなければ "affine"（SIFT/ECC ＋中心窩固定）。
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
