import cv2
import numpy as np

from engine.region_growing import (
    DEFAULT_MAX_DELTA_PX,
    DEFAULT_RADIUS_PX,
    GROW_MODE_STANDARD,
    expand_segmentation_from_mask,
)
from engine.segmentation import (
    SEED_MAX_COMPONENTS,
    SEG_AUTO_MODE_OTSU,
    SEG_AUTO_MODE_SEEDED,
    segment_ga,
    segment_ga_seeded,
    segment_ga_with_optional_cv,
)
from skimage.measure import label


def _synthetic_faf_image(size=200, blob_radius=30, bg=200, blob=30) -> np.ndarray:
    """明るい背景に中心暗色円（GAを模した病変）を持つ合成FAF画像を作る。"""
    img = np.full((size, size), bg, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), blob_radius, blob, thickness=-1)
    return img


def test_segment_ga_detects_central_dark_lesion():
    img = _synthetic_faf_image()
    mask = segment_ga(img)

    assert mask.shape == img.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    # 中心付近に検出されたGA画素があること
    assert mask[100, 100] == 255
    assert mask.sum() > 0


def test_segment_ga_no_lesion_on_uniform_image():
    img = np.full((100, 100), 128, dtype=np.uint8)
    mask = segment_ga(img)
    assert mask.shape == img.shape


def test_segment_ga_accepts_macula_radius_and_min_size():
    """高影響パラメータが kwargs で受け付けられ、min_size が大きいと面積が減る。"""
    # 中程度の病変（面積 ~700px）— min_size=2000 で消える
    img = _synthetic_faf_image(size=200, blob_radius=15, bg=200, blob=30)
    fovea = (100.0, 100.0)
    scale = 10.0  # um/px → 6mm ROI は画像全体を覆う
    kept = segment_ga(
        img, fovea_xy=fovea, scale_um_px=scale, macula_radius_mm=6.0, min_size=100,
    )
    dropped = segment_ga(
        img, fovea_xy=fovea, scale_um_px=scale, macula_radius_mm=6.0, min_size=2000,
    )
    assert kept.shape == img.shape and dropped.shape == img.shape
    assert set(np.unique(kept)).issubset({0, 255})
    assert int(np.count_nonzero(kept)) > 0
    assert int(np.count_nonzero(dropped)) < int(np.count_nonzero(kept))
    # macula_radius_mm も実行できること
    narrow = segment_ga(
        img, fovea_xy=fovea, scale_um_px=scale, macula_radius_mm=3.0, min_size=100,
    )
    assert narrow.shape == img.shape


def test_auto_cv_off_matches_segment_ga_alone():
    """auto_cv=False は Otsu のみ（segment_ga）と一致。"""
    img = _synthetic_faf_image()
    otsu = segment_ga(img)
    off = segment_ga_with_optional_cv(img, auto_cv=False)
    assert np.array_equal(off, otsu)


def test_auto_cv_on_fills_interior_holes(monkeypatch):
    """auto_cv=ON: Otsu 後マスクの内部穴を埋める。"""
    img = np.full((64, 64), 200, dtype=np.uint8)
    img[16:48, 16:48] = 40
    base = np.zeros((64, 64), dtype=np.uint8)
    base[18:46, 18:46] = 255
    base[28:36, 28:36] = 0  # enclosed hole

    monkeypatch.setattr(
        "engine.segmentation.segment_ga",
        lambda *a, **k: base.copy(),
    )
    refined = segment_ga_with_optional_cv(img, auto_cv=True)
    assert refined[32, 32] == 255
    assert np.all(refined[base > 0] == 255)


def test_auto_cv_on_clips_peripheral_expansion(monkeypatch):
    """auto_cv=ON: 外周は max_outward_px 超えて外へ広がらない。"""
    from scipy.ndimage import binary_dilation

    img = np.full((64, 64), 40, dtype=np.uint8)  # uniform dark → CV wants expand
    base = np.zeros((64, 64), dtype=np.uint8)
    base[24:40, 24:40] = 255

    monkeypatch.setattr(
        "engine.segmentation.segment_ga",
        lambda *a, **k: base.copy(),
    )
    r = 1
    refined = segment_ga_with_optional_cv(img, auto_cv=True)
    pre_b = base > 0
    allowed = binary_dilation(pre_b, iterations=r)
    added = (refined > 0) & ~pre_b
    assert not np.any(added & ~allowed)
    assert np.all(refined[pre_b] == 255)


def test_auto_cv_skips_empty_mask(monkeypatch):
    """空マスクでは CV を呼ばずそのまま返す。"""
    empty = np.zeros((32, 32), dtype=np.uint8)
    monkeypatch.setattr(
        "engine.segmentation.segment_ga",
        lambda *a, **k: empty.copy(),
    )
    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("CV should be skipped on empty mask")

    monkeypatch.setattr(
        "engine.region_growing.refine_mask_chan_vese_light",
        _boom,
    )
    out = segment_ga_with_optional_cv(np.zeros((32, 32), dtype=np.uint8), auto_cv=True)
    assert np.array_equal(out, empty)
    assert calls["n"] == 0


def test_seeded_auto_keeps_multiple_seed_slots():
    """多巣向けに複数シード枠を確保。"""
    assert SEED_MAX_COMPONENTS >= 8


def test_standard_grow_prefers_dark_over_bright():
    """標準成長: 暗部周囲には広がり、白い背景には乗りにくい。"""
    size = 120
    img = np.full((size, size), 220, dtype=np.uint8)  # white bg
    cv2.circle(img, (60, 60), 25, 30, thickness=-1)   # dark lesion
    mask0 = np.zeros((size, size), dtype=np.uint8)
    mask0[60, 60] = 255
    grown, added = expand_segmentation_from_mask(
        img,
        mask0,
        (60, 60),
        a=1.5,
        window_size=11,
        radius_px=40,
        max_delta_px=3000,
        require_adjacency=False,
        mode=GROW_MODE_STANDARD,
    )
    assert added > 50
    assert grown[60, 60] == 255
    # 暗コア内は多く覆う
    yy, xx = np.ogrid[:size, :size]
    core = (xx - 60) ** 2 + (yy - 60) ** 2 <= 18 ** 2
    assert int(np.count_nonzero(grown[core] > 0)) > 0.6 * int(np.count_nonzero(core))
    # 白い背景遠方には乗らない
    far = (xx - 60) ** 2 + (yy - 60) ** 2 > 40 ** 2
    assert int(np.count_nonzero(grown[far] > 0)) < 0.02 * int(np.count_nonzero(far))


def test_segment_ga_seeded_keeps_multifocal_islands():
    """複数の小円形暗部を主病変1つに潰さず残す。"""
    size = 240
    img = np.full((size, size), 200, dtype=np.uint8)
    centers = [(120, 120), (80, 90), (160, 100), (100, 160)]
    for cx, cy in centers:
        cv2.circle(img, (cx, cy), 12, 25, thickness=-1)
    fovea = (120.0, 120.0)
    mask, info = segment_ga_seeded(
        img,
        fovea_xy=fovea,
        scale_um_px=10.0,
        macula_radius_mm=6.0,
        min_size=30,
        use_entropy_stop=True,
    )
    assert info.get("fallback") is False
    labeled = label(mask > 0)
    n_comp = int(labeled.max())
    assert n_comp >= 3
    for cx, cy in centers:
        assert mask[cy, cx] == 255, f"missing island at {(cx, cy)}"


def test_segment_ga_seeded_mottled_multifocal_survives_ui_min_size():
    """斑状多巣の小暗コアが UI 既定 min_size=500 でも後処理で空にならない。"""
    size = 320
    # Optos 風: やや暗め背景＋斑状の小暗巣（各コア ≪ 500px）
    rng = np.random.default_rng(0)
    img = np.full((size, size), 145, dtype=np.uint8)
    noise = rng.integers(-12, 13, size=(size, size), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    centers = [
        (160, 160), (130, 140), (190, 145), (145, 195),
        (175, 185), (115, 170), (200, 175), (155, 125),
    ]
    for cx, cy in centers:
        cv2.circle(img, (cx, cy), 7, 35, thickness=-1)  # ~150px each
        cv2.circle(img, (cx, cy), 3, 20, thickness=-1)
    fovea = (160.0, 160.0)
    mask, info = segment_ga_seeded(
        img,
        fovea_xy=fovea,
        scale_um_px=18.96,
        macula_radius_mm=6.0,
        min_size=500,  # UI seg_min_size 既定
    )
    assert info.get("fallback") is False, info
    assert "後処理で空" not in str(info.get("message", ""))
    assert info.get("mode_used") == SEG_AUTO_MODE_SEEDED
    assert info.get("n_seeds", 0) >= 3
    assert int(np.count_nonzero(mask)) > 0
    # 巨大 Otsu 状マスクにならない（画像の 15% 未満）
    assert int(np.count_nonzero(mask)) < 0.15 * size * size
    hit = sum(1 for cx, cy in centers if mask[cy, cx] == 255)
    assert hit >= 3, f"only {hit}/{len(centers)} cores kept"


def test_segment_ga_seeded_grows_from_fovea_dark_cluster():
    """Seeded: 中心窩近傍の暗部から成長し中心をマスクする。"""
    img = _synthetic_faf_image(size=200, blob_radius=35, bg=200, blob=30)
    fovea = (100.0, 100.0)
    scale = 10.0  # um/px → 6mm ROI covers image
    mask, info = segment_ga_seeded(
        img,
        fovea_xy=fovea,
        scale_um_px=scale,
        macula_radius_mm=6.0,
        min_size=100,
        use_entropy_stop=True,
    )
    assert mask.shape == img.shape
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask[100, 100] == 255
    assert int(np.count_nonzero(mask)) > 0
    assert info.get("fallback") is False
    assert info.get("n_seeds", 0) >= 1
    assert info.get("mode_used") == SEG_AUTO_MODE_SEEDED


def test_segment_ga_seeded_resists_gray_fringe_expansion():
    """自動 Seeded はコア検出のみで、灰縁まで埋めない。"""
    size = 220
    cx = cy = size // 2
    img = np.full((size, size), 210, dtype=np.uint8)
    cv2.circle(img, (cx, cy), 55, 110, thickness=-1)  # gray fringe
    cv2.circle(img, (cx, cy), 28, 25, thickness=-1)   # dark GA core
    fovea = (float(cx), float(cy))
    mask, info = segment_ga_seeded(
        img,
        fovea_xy=fovea,
        scale_um_px=10.0,
        macula_radius_mm=6.0,
        min_size=50,
    )
    assert info.get("fallback") is False
    assert mask[cy, cx] == 255
    yy, xx = np.ogrid[:size, :size]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    fringe = (r2 >= 40 ** 2) & (r2 <= 55 ** 2)
    # コア検出のみなので灰縁への乗りはごくわずか
    assert int(np.count_nonzero(mask[fringe] > 0)) < 0.15 * int(np.count_nonzero(fringe))
    core = r2 <= 20 ** 2
    assert int(np.count_nonzero(mask[core] > 0)) > 0.4 * int(np.count_nonzero(core))


def test_segment_ga_seeded_empty_seed_falls_back_to_otsu():
    """均一画像などシード不能時は Otsu にフォールバック。"""
    img = np.full((120, 120), 160, dtype=np.uint8)
    fovea = (60.0, 60.0)
    scale = 10.0
    mask, info = segment_ga_seeded(
        img,
        fovea_xy=fovea,
        scale_um_px=scale,
        macula_radius_mm=6.0,
        min_size=50,
    )
    assert mask.shape == img.shape
    assert info.get("fallback") is True
    assert info.get("mode_used") == SEG_AUTO_MODE_OTSU
    assert "フォールバック" in str(info.get("message", ""))
    assert "暗部シードなし" in str(info.get("message", "")) or "Seeded不可" in str(
        info.get("message", "")
    )


def test_segment_ga_with_optional_cv_seeded_mode_wires_info():
    """UI 共通入口: mode=seeded で info が埋まる。"""
    img = _synthetic_faf_image()
    info: dict = {}
    mask = segment_ga_with_optional_cv(
        img,
        auto_cv=False,
        mode=SEG_AUTO_MODE_SEEDED,
        fovea_xy=(100.0, 100.0),
        scale_um_px=10.0,
        macula_radius_mm=6.0,
        min_size=100,
        info=info,
    )
    assert mask[100, 100] == 255
    assert info.get("mode_used") in (SEG_AUTO_MODE_SEEDED, SEG_AUTO_MODE_OTSU)
    assert "message" in info


def test_segment_ga_with_optional_cv_default_mode_remains_otsu():
    """既存呼び出し（mode 省略）は Otsu 経路のまま。"""
    img = _synthetic_faf_image()
    info: dict = {}
    mask = segment_ga_with_optional_cv(img, auto_cv=False, info=info)
    otsu = segment_ga(img)
    assert np.array_equal(mask, otsu)
    assert info.get("mode_used") == SEG_AUTO_MODE_OTSU
