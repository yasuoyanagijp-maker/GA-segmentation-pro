#!/usr/bin/env python3
"""FAF 画像から GAWAIN 説明図（Figure 1 型 A–E）を生成する。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.calibration import compute_scale_um_per_px_from_field_width
from engine.gawain_figure import (
    crop_display_margins,
    estimate_fovea_from_mask,
    render_gawain_figure,
    split_optos_dual_panel,
)
from engine.io_utils import load_gray
from engine.segmentation import segment_ga


def _parse_fovea(text: str) -> tuple[float, float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("fovea は 'x y' 形式で指定してください")
    return float(parts[0]), float(parts[1])


def main() -> int:
    parser = argparse.ArgumentParser(description="GAWAIN Figure 1 型説明図を生成")
    parser.add_argument("image", type=Path, help="FAF 画像パス")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="出力 PNG（省略時は <stem>_gawain_figure.png）",
    )
    parser.add_argument(
        "--eye", choices=("R", "L", "both"), default="R",
        help="Optos 2眼並列画像の場合の対象眼（既定: R=左パネル）",
    )
    parser.add_argument(
        "--field-width-mm", type=float, default=30.0,
        help="クロップ後画像の水平実長 mm（Optos 既定 30）",
    )
    parser.add_argument(
        "--scale-um-px", type=float, default=None,
        help="μm/px を直接指定（指定時は field-width を無視）",
    )
    parser.add_argument(
        "--fovea", type=_parse_fovea, default=None,
        help="中心窩座標 'x y'（省略時は GA 重心から推定）",
    )
    parser.add_argument("--no-crop-margins", action="store_true")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    img_path = args.image.expanduser()
    if not img_path.is_file():
        print(f"Error: file not found: {img_path}", file=sys.stderr)
        return 1

    img = load_gray(img_path)
    if args.eye != "both":
        img = split_optos_dual_panel(img, args.eye)
    if not args.no_crop_margins:
        img = crop_display_margins(img)

    scale = args.scale_um_px
    if scale is None:
        h, w = img.shape[:2]
        scale = compute_scale_um_per_px_from_field_width(w, args.field_width_mm)

    fovea = args.fovea
    if fovea is None:
        probe = segment_ga(img, fovea_xy=None, scale_um_px=scale)
        fovea = estimate_fovea_from_mask(probe)
        if fovea is None:
            fovea = (img.shape[1] / 2.0, img.shape[0] / 2.0)
            print("Warning: GA not detected; using image center as fovea", file=sys.stderr)

    mask = segment_ga(img, fovea_xy=fovea, scale_um_px=scale)

    out = args.output
    if out is None:
        suffix = "" if args.eye == "both" else f"_{args.eye}"
        out = img_path.with_name(f"{img_path.stem}{suffix}_gawain_figure.png")

    title = args.title or f"GAWAIN — {img_path.name}"
    result = render_gawain_figure(
        img, mask, fovea, scale, out, title=title,
    )
    print(f"Saved: {result['output_path']}")
    print(f"GAWAIN = {result['gawain']:.1f}  ·  GA area = {result['area_mm2']:.3f} mm²")
    print(f"Fovea = ({result['fovea_xy'][0]:.0f}, {result['fovea_xy'][1]:.0f})  ·  scale = {scale:.2f} μm/px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
