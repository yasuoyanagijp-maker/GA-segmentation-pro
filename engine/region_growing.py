"""region_growing.py – リージョングローイング（MNV=暗部 / GA=白部・過分割）。

Interactive expand/erase は常時 Phase A 安全レール（半径・画素数、
a は UI 側でキャップ）を適用。expand は既存マスク隣接を優先するが、
非隣接シードでは局所の新規島を追加する。成長モードは standard / lsf /（確定後）cv。
"""

from __future__ import annotations

import cv2
import numpy as np

# Phase A defaults（全モード共通の爆発防止）
DEFAULT_RADIUS_PX = 64
DEFAULT_MAX_DELTA_PX = 4000

# grow_mode 定数（UI / state と共有）
GROW_MODE_STANDARD = "standard"
GROW_MODE_LSF = "lsf"
GROW_MODE_CV = "cv_refine"

# 未実装フック（UI 非公開）:
# GROW_MODE_ENTROPY = "li_entropy"   # Li et al.
# GROW_MODE_CHEN = "chen"
# GROW_MODE_SCHERZER = "scherzer"


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _local_stats(gray: np.ndarray, x: int, y: int, window_size: int) -> tuple[float, float]:
    h, w = gray.shape[:2]
    half_w = window_size // 2
    y_min, y_max = max(0, y - half_w), min(h, y + half_w + 1)
    x_min, x_max = max(0, x - half_w), min(w, x + half_w + 1)
    roi = gray[y_min:y_max, x_min:x_max]
    mu = float(np.mean(roi))
    sigma = float(np.maximum(np.std(roi), 0.5))
    return mu, sigma


def _flood_from_potential(potential_mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    x, y = seed
    h, w = potential_mask.shape[:2]
    if potential_mask[y, x] == 0:
        return np.zeros((h, w), dtype=np.uint8)

    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood_mask[1:-1, 1:-1] = np.where(potential_mask == 0, 1, 0).astype(np.uint8)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
    image_dummy = np.zeros((h, w), dtype=np.uint8)
    cv2.floodFill(image_dummy, flood_mask, seed, 0, 0, 0, flags)
    return np.where(flood_mask[1:-1, 1:-1] == 255, 255, 0).astype(np.uint8)


def _apply_radius_cap(
    potential: np.ndarray,
    seed: tuple[int, int],
    radius_px: int | None,
) -> np.ndarray:
    if radius_px is None or radius_px <= 0:
        return potential
    h, w = potential.shape[:2]
    x, y = int(seed[0]), int(seed[1])
    yy, xx = np.ogrid[:h, :w]
    out = potential.copy()
    out[(xx - x) ** 2 + (yy - y) ** 2 > int(radius_px) ** 2] = 0
    return out


def _cap_delta_by_distance(
    delta_mask: np.ndarray,
    seed: tuple[int, int],
    max_px: int | None,
) -> np.ndarray:
    """Keep at most max_px True pixels, preferring those closest to seed."""
    if max_px is None or max_px <= 0:
        return delta_mask
    ys, xs = np.where(delta_mask > 0)
    n = int(ys.size)
    if n <= max_px:
        return delta_mask
    x, y = int(seed[0]), int(seed[1])
    order = np.argsort((xs - x) ** 2 + (ys - y) ** 2)
    keep_idx = order[: int(max_px)]
    out = np.zeros_like(delta_mask)
    out[ys[keep_idx], xs[keep_idx]] = 255
    return out


def _restrict_connected_to_mask(
    candidate: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Keep candidate pixels 4-connected to existing mask through candidate∪mask."""
    if not np.any(mask > 0):
        return candidate
    if not np.any(candidate > 0):
        return candidate
    union = np.where((candidate > 0) | (mask > 0), 255, 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(union, connectivity=4)
    if num <= 1:
        return np.zeros_like(candidate)
    mask_labels = np.unique(labels[mask > 0])
    keep = np.zeros_like(candidate)
    for lab in mask_labels:
        if lab == 0:
            continue
        keep[labels == lab] = 255
    return np.where((keep > 0) & (candidate > 0), 255, 0).astype(np.uint8)


def _potential_mu_sigma(
    gray: np.ndarray,
    seed: tuple[int, int],
    a: float,
    window_size: int,
) -> np.ndarray:
    x, y = seed
    mu, sigma = _local_stats(gray, x, y, window_size)
    seed_val = float(gray[y, x])
    low = float(np.clip(mu - a * sigma, 0, 255))
    high = float(np.clip(mu + a * sigma, 0, 255))
    # 暗部シードが明部背景で μ が上がると μ±aσ から外れ、flood が空になる。
    # シード強度は常に含める（境界クリック・島クリックの無反応防止）。
    low = float(min(low, seed_val))
    high = float(max(high, seed_val))
    return cv2.inRange(gray, low, high)


def _potential_lsf(
    gray: np.ndarray,
    seed: tuple[int, int],
    a: float,
    window_size: int,
) -> np.ndarray:
    """Niu-inspired local similarity factor (practical approximation).

    Accept pixels where both intensity and local mean stay close to the seed
    neighborhood, using a locally adaptive σ (box-filter std).
    """
    x, y = int(seed[0]), int(seed[1])
    mu_s, sigma_s = _local_stats(gray, x, y, window_size)
    gray_f = gray.astype(np.float32)
    k = max(3, window_size | 1)
    local_mean = cv2.blur(gray_f, (k, k))
    local_sq = cv2.blur(gray_f * gray_f, (k, k))
    local_var = np.maximum(local_sq - local_mean * local_mean, 0.25)
    local_std = np.sqrt(local_var)
    adapt = np.maximum(local_std, sigma_s)
    thr = float(max(a, 0.15)) * adapt
    dist_i = np.abs(gray_f - mu_s)
    dist_m = np.abs(local_mean - mu_s)
    ok = (dist_i <= thr) & (dist_m <= thr * 0.85)
    return (ok.astype(np.uint8) * 255)


def _build_grow_potential(
    gray: np.ndarray,
    seed: tuple[int, int],
    a: float,
    *,
    window_size: int,
    mode: str,
) -> np.ndarray:
    if mode == GROW_MODE_LSF:
        return _potential_lsf(gray, seed, a, window_size)
    # standard / cv_refine preview: μ±aσ
    return _potential_mu_sigma(gray, seed, a, window_size)


def fast_region_growing(
    img: np.ndarray,
    seed: tuple[int, int],
    a: float,
    window_size: int = 5,
) -> np.ndarray:
    """MNV 向け: シードと同程度の**暗部**へ μ±aσ で連結拡大（双方向）。"""
    gray = _to_gray(img)
    x, y = seed
    h, w = gray.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return np.zeros((h, w), dtype=np.uint8)

    potential_mask = _potential_mu_sigma(gray, (x, y), a, window_size)
    return _flood_from_potential(potential_mask, seed)


def fast_region_growing_bright(
    img: np.ndarray,
    seed: tuple[int, int],
    a: float,
    *,
    window_size: int = 5,
    dark_margin: float = 0.25,
) -> np.ndarray:
    """明るい画素へ拡大（シードが画像中央値より暗い場合は空）。"""
    gray = _to_gray(img)
    x, y = seed
    h, w = gray.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return np.zeros((h, w), dtype=np.uint8)

    seed_val = float(gray[y, x])
    if seed_val < float(np.median(gray)):
        return np.zeros((h, w), dtype=np.uint8)

    mu, sigma = _local_stats(gray, x, y, window_size)
    low = float(np.clip(max(mu - dark_margin * a * sigma, seed_val - dark_margin * a * sigma), 0, 255))
    high = float(np.clip(mu + a * sigma, 0, 255))
    potential_mask = cv2.inRange(gray, low, high)

    grown = _flood_from_potential(potential_mask, seed)
    if not np.any(grown):
        return grown

    bright_floor = float(np.clip(seed_val - dark_margin * a * sigma, 0, 255))
    grown[gray < bright_floor] = 0
    return grown


def erase_oversegmentation_from_mask(
    img_gray: np.ndarray,
    mask: np.ndarray,
    seed: tuple[int, int],
    a: float,
    *,
    window_size: int = 11,
    radius_px: int | None = DEFAULT_RADIUS_PX,
    max_delta_px: int | None = DEFAULT_MAX_DELTA_PX,
    require_adjacency: bool = True,
    mode: str = GROW_MODE_STANDARD,
) -> tuple[np.ndarray, int]:
    """FAF 向け: マスク内の明るい過分割を除去（元画像グレーのみ使用）。

    緑の半透明表示は参照しない。暗い GA 本体上のクリックでも、
    連結するより明るいマスク画素を長押しで拡大削除する。
    Phase A: 半径・最大削除画素・既存マスク隣接を常時適用。
    戻り値: (新マスク, 削除画素数)
    """
    del mode  # erase は potential を独自構築（LSF は expand 側）
    gray = _to_gray(img_gray)
    x, y = int(seed[0]), int(seed[1])
    h, w = gray.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return mask, 0

    on_mask = mask[y, x] > 0
    seed_val = float(gray[y, x])
    mu, sigma = _local_stats(gray, x, y, window_size)
    peel_floor = float(np.clip(seed_val + 0.15 * a * sigma, seed_val + 1, 255))
    high = float(np.clip(seed_val + a * sigma * 1.5, peel_floor + 1, 255))

    potential = cv2.inRange(gray, int(seed_val), int(high))
    potential = _apply_radius_cap(potential, (x, y), radius_px)
    if on_mask:
        # マスク上クリック: 拡大は GA 領域内に限定
        potential[mask == 0] = 0
        if potential[y, x] == 0:
            return mask, 0
    elif not np.any((potential > 0) & (mask > 0)):
        # 緑外クリック: 強度連結でマスクに届かない場合は何もしない
        return mask, 0

    grown = _flood_from_potential(potential, (x, y))
    if not np.any(grown):
        return mask, 0

    mask_in_grown = (grown > 0) & (mask > 0)
    if not np.any(mask_in_grown):
        return mask, 0

    # 暗い GA 本体: より明るい縁だけ削除 / 明るい過分割: 連結領域を削除 / 均一領域: 何もしない
    if seed_val < mu - 0.25 * sigma:
        remove = mask_in_grown & (gray >= peel_floor)
    elif seed_val > mu + 0.25 * sigma:
        remove = mask_in_grown
    else:
        return mask, 0

    remove_u8 = np.where(remove, 255, 0).astype(np.uint8)
    if require_adjacency:
        # 削除候補は既存マスクに接している連結成分のみ（孤立飛び火防止）
        remove_u8 = _restrict_connected_to_mask(remove_u8, mask)
    remove_u8 = _cap_delta_by_distance(remove_u8, (x, y), max_delta_px)
    removed = int(np.sum(remove_u8 > 0))
    if removed == 0:
        return mask, 0

    out = mask.copy()
    out[remove_u8 > 0] = 0
    return out, removed


def expand_segmentation_from_mask(
    img_gray: np.ndarray,
    mask: np.ndarray,
    seed: tuple[int, int],
    a: float,
    *,
    window_size: int = 11,
    radius_px: int | None = DEFAULT_RADIUS_PX,
    max_delta_px: int | None = DEFAULT_MAX_DELTA_PX,
    require_adjacency: bool = True,
    mode: str = GROW_MODE_STANDARD,
) -> tuple[np.ndarray, int]:
    """FAF 向け: 暗い GA 領域をマスクに追加（erase の逆操作）。

    緑の内外どちらでも、暗部を長押しで連結拡大しマスクに加える。
    Phase A: 半径・最大追加画素は常時適用。
    require_adjacency=True でも、既存マスクと連結できない場合は
    シードから局所の新規島を追加する（遠方飛び火は半径キャップで抑制）。
    mode=lsf で局所類似ポテンシャル。cv_refine は preview では standard 相当。
    戻り値: (新マスク, 追加画素数)
    """
    gray = _to_gray(img_gray)
    x, y = int(seed[0]), int(seed[1])
    h, w = gray.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return mask, 0

    seed_val = float(gray[y, x])
    mu, sigma = _local_stats(gray, x, y, window_size)

    # 明らかに明るい画素は GA 暗部成長の対象外（中立帯は許容）
    if seed_val > mu + 0.4 * sigma:
        return mask, 0

    grow_mode = GROW_MODE_LSF if mode == GROW_MODE_LSF else GROW_MODE_STANDARD
    potential = _build_grow_potential(
        gray, (x, y), a, window_size=window_size, mode=grow_mode,
    )
    potential = _apply_radius_cap(potential, (x, y), radius_px)
    if potential[y, x] == 0:
        # LSF 等でシードが弾かれた場合は μ±aσ（シード強度保証付き）へフォールバック
        potential = _apply_radius_cap(
            _potential_mu_sigma(gray, (x, y), a, window_size),
            (x, y),
            radius_px,
        )
    grown = _flood_from_potential(potential, (x, y))
    if not np.any(grown):
        return mask, 0

    add = np.where((grown > 0) & (mask == 0), 255, 0).astype(np.uint8)
    if require_adjacency and np.any(mask > 0) and np.any(add):
        adjacent = _restrict_connected_to_mask(add, mask)
        if np.any(adjacent):
            # 既存マスクと連結できる部分だけ拡張
            add = adjacent
        # else: 非隣接 → 局所新規島として add を維持（半径・max_delta で抑制）
    add = _cap_delta_by_distance(add, (x, y), max_delta_px)
    added = int(np.sum(add > 0))
    if added == 0:
        return mask, 0

    out = mask.copy()
    out[add > 0] = 255
    return out, added


def refine_mask_chan_vese_light(
    img_gray: np.ndarray,
    mask: np.ndarray,
    *,
    num_iter: int = 6,
    smoothing: int = 1,
    pad: int = 24,
    max_outward_px: int = 1,
) -> np.ndarray:
    """Stroke commit 用の軽量 Morphological Chan-Vese（ROI crop のみ）。

    プレビュー（連続編集）では呼ばない。ga_roi_correction.refine があれば再利用し、
    なければ skimage.morphological_chan_vese を直接使う。

    CV は均質領域へ外向きにも広がるため、確定後モードでは
    ``refined = (cv ∩ dilate(pre, r)) ∪ pre`` で外周成長をクリップし、
    続けて ``binary_fill_holes`` で内部穴だけ埋める（外周はほぼ維持）。
    """
    from scipy.ndimage import binary_dilation, binary_fill_holes

    gray = _to_gray(img_gray)
    if mask is None or not np.any(mask > 0):
        return mask

    ys, xs = np.where(mask > 0)
    h, w = gray.shape[:2]
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    crop_img = gray[y0:y1, x0:x1]
    crop_mask = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    pre_b = crop_mask.astype(bool)

    refined_crop: np.ndarray | None = None
    try:
        import sys
        from pathlib import Path

        pkg_root = Path(__file__).resolve().parents[1] / "ga_roi_correction"
        if str(pkg_root) not in sys.path:
            sys.path.insert(0, str(pkg_root))
        from ga_roi_correction.refine import chan_vese_refine  # type: ignore

        refined_crop = chan_vese_refine(
            crop_img, crop_mask, num_iter=num_iter, smoothing=smoothing,
        )
    except Exception:
        refined_crop = None

    if refined_crop is None:
        from skimage.segmentation import morphological_chan_vese

        image_f = crop_img.astype(np.float64) / 255.0
        evolved = morphological_chan_vese(
            image_f,
            num_iter=num_iter,
            init_level_set=crop_mask.astype(np.uint8),
            smoothing=smoothing,
        )
        refined_crop = (np.asarray(evolved).astype(bool)).astype(np.uint8)

    cv_b = refined_crop.astype(bool)
    # 外周の大きな外向き成長を抑制: CV 追加は dilate(pre, r) 内のみ許可
    r = max(0, int(max_outward_px))
    allowed = binary_dilation(pre_b, iterations=r) if r > 0 else pre_b
    clipped = (cv_b & allowed) | pre_b
    # 内部穴・ギャップは必ず埋める（外周は上記クリップで維持）
    clipped = binary_fill_holes(clipped)

    out = mask.copy()
    # crop 内だけ置換（外側の既存マスクは維持）
    out[y0:y1, x0:x1] = np.where(clipped, 255, 0).astype(np.uint8)
    return out


def clear_mask_in_freehand_region(
    mask: np.ndarray,
    points: list[tuple[float, float]],
    *,
    outside: bool = False,
) -> tuple[np.ndarray, int]:
    """フリーハンドで囲んだ領域のマスクを消去する。

    outside=False: 範囲内を消去（従来どおり）
    outside=True: 範囲外を消去（囲んだ内側だけ残す）
    """
    h, w = mask.shape[:2]
    if len(points) < 3:
        return mask, 0
    pts = np.array([[int(x), int(y)] for x, y in points], dtype=np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    region = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(region, [pts], 255)
    if outside:
        remove = (mask > 0) & (region == 0)
    else:
        remove = (mask > 0) & (region > 0)
    removed = int(np.sum(remove))
    if removed == 0:
        return mask, 0
    out = mask.copy()
    out[remove] = 0
    return out, removed


# 後方互換
def erase_bright_from_mask(
    img_gray: np.ndarray,
    mask: np.ndarray,
    seed: tuple[int, int],
    a: float,
) -> np.ndarray:
    out, _ = erase_oversegmentation_from_mask(img_gray, mask, seed, a)
    return out
