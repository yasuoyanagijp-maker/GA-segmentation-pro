"""Benchmark registration on FAF progression / cropped folders."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.io_utils import load_gray
from engine.overlay import registration_diff_overlay
from engine.series_registration import (
    _matrix_similarity_scale,
    _prepare_registration_pair,
    compute_forward_matrix,
    warp_image,
)


def run(folder: Path) -> None:
    files = sorted(
        [
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and not p.name.startswith("reg_")
            and not p.name.startswith("debug_")
            and p.name != "名称未設定.jpg"
        ],
        key=lambda p: p.name,
    )
    if len(files) < 2:
        raise SystemExit(f"Need 2+ images in {folder}")

    ref_full = load_gray(files[0])
    ref, _ = _prepare_registration_pair(ref_full, ref_full)
    print(f"ref={files[0].name} shape={ref.shape}  n={len(files)}")

    for f in files[1:]:
        mov_full = load_gray(f)
        _, mov = _prepare_registration_pair(ref_full, mov_full)
        if mov.shape != ref.shape:
            mov = cv2.resize(mov, (ref.shape[1], ref.shape[0]))
        m = compute_forward_matrix(mov, ref, method="landmark", anchor_fovea=False)
        aligned = warp_image(mov, m, (ref.shape[1], ref.shape[0]))
        diff = cv2.absdiff(ref, aligned)
        mae = float(diff.mean())
        sc = _matrix_similarity_scale(m)
        ang = math.degrees(math.atan2(float(m[1, 0]), float(m[0, 0])))
        out = folder / f"reg_v2_{f.stem}.png"
        cv2.imwrite(str(out), registration_diff_overlay(ref, aligned))
        print(
            f"  {f.name:28} mae={mae:6.2f}  scale={sc:.3f}"
            f"  rot={ang:+.1f}°  tx={m[0, 2]:+7.1f} ty={m[1, 2]:+7.1f}"
            f"  -> {out.name}"
        )


if __name__ == "__main__":
    folder = Path(
        sys.argv[1] if len(sys.argv) > 1 else "/Users/yy/Desktop/FAF_GA_progression"
    )
    run(folder)
