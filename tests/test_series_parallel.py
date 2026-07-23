"""build_series_from_saved の並列登録が直列相当と同一結果になることの検証。"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.series_registration import register_moving_to_reference
from engine.stats import ga_stats
from engine.timeseries import build_series_from_saved


def _synthetic_faf(size=256, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.hypot(xx - size * 0.45, yy - size * 0.5)
    img += 120.0 * np.exp(-(r / (size * 0.55)) ** 2)
    x, y = size * 0.75, size * 0.5
    for _ in range(10):
        ang = rng.uniform(0, 2 * np.pi)
        px, py = x, y
        for _ in range(100):
            ang += rng.normal(0, 0.2)
            nx, ny = px + 3 * np.cos(ang), py + 3 * np.sin(ang)
            if not (0 <= nx < size and 0 <= ny < size):
                break
            cv2.line(img, (int(px), int(py)), (int(nx), int(ny)), -60.0, 2)
            px, py = nx, ny
    img += rng.normal(0, 5, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def series_inputs():
    size = 256
    ref = _synthetic_faf(size=size, seed=0)
    per_file = {}
    files = []
    dates = {}
    fovea = (size * 0.45, size * 0.5)
    scale = 10.0
    for i, (angle, dx) in enumerate([(0.0, 0), (2.0, 6), (-3.0, -8)]):
        if i == 0:
            img = ref
        else:
            m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
            m[0, 2] += dx
            img = cv2.warpAffine(ref, m, (size, size))
        mask = np.zeros((size, size), np.uint8)
        cv2.circle(
            mask, (int(size * 0.45), int(size * 0.5)),
            int(size * (0.06 + 0.02 * i)), 255, -1,
        )
        fname = f"t{i}.png"
        date = f"202{i}-01-01"
        per_file[fname] = {
            "path": Path(f"/tmp/{fname}"),
            "filename": fname,
            "date": date,
            "image": img,
            "mask": mask,
            "stats": ga_stats(mask, scale, fovea),
            "fovea_xy": fovea,
            "scale_um_px": scale,
        }
        files.append(Path(f"/tmp/{fname}"))
        dates[fname] = date
    return per_file, files, dates


def test_parallel_registration_matches_direct_calls(series_inputs):
    """並列ビルドの各行列が、同じ入力での直接呼び出しと一致する。"""
    per_file, files, dates = series_inputs
    # 行列比較のため再segはOFF（マスクは warp 結果と一致させる）
    results = build_series_from_saved(
        per_file, files, dates, register=True, resegment_after_reg=False,
    )
    assert len(results) == 3
    assert results[0]["is_reference"]
    ref_img = results[0]["image"]
    ref_fovea = per_file["t0.png"]["fovea_xy"]
    for row in results[1:]:
        snap = per_file[row["filename"]]
        direct = register_moving_to_reference(
            snap["image"], snap["mask"], ref_img,
            method=row["reg_method"],
            moving_fovea=snap["fovea_xy"],
            ref_fovea=ref_fovea,
            moving_disc=None,
            ref_disc=None,
            anchor_fovea=True,
        )
        assert np.allclose(row["reg_matrix"], direct["matrix"], atol=1e-4)
        assert np.array_equal(row["mask"], direct["mask"])
        assert row["mask_source"] == "warp"


def test_parallel_registration_is_deterministic(series_inputs):
    """同じ入力で2回ビルドしても行列・stats が一致する（並列順序に非依存）。"""
    per_file, files, dates = series_inputs
    a = build_series_from_saved(
        per_file, files, dates, register=True, resegment_after_reg=False,
    )
    b = build_series_from_saved(
        per_file, files, dates, register=True, resegment_after_reg=False,
    )
    for ra, rb in zip(a, b):
        assert np.allclose(ra["reg_matrix"], rb["reg_matrix"], atol=1e-4)
        assert ra["stats"]["area_mm2"] == rb["stats"]["area_mm2"]
        assert ra["stats"]["gawain"] == rb["stats"]["gawain"]
