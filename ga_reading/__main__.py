"""第1・第2の統合と、最終読影の確定。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ga_reading.merge import merge_graders
from ga_reading.recompute_scale import main as recompute_scale_main
from ga_reading.triad import resolve_triad


def main() -> None:
    parser = argparse.ArgumentParser(description="GA 3者読影の統合と確定")
    sub = parser.add_subparsers(dest="command", required=True)

    merge = sub.add_parser("merge", help="第1と第2の CSV を統合する")
    merge.add_argument("first_csv", type=Path)
    merge.add_argument("second_csv", type=Path)
    merge.add_argument("--out", type=Path, required=True)
    merge.add_argument("--prefix", default=None)

    triad = sub.add_parser("triad", help="RECHECK だけを最終読影で確定する")
    triad.add_argument("summary_md", type=Path)
    triad.add_argument("final_csv", type=Path)
    triad.add_argument("--out", type=Path, required=True)
    triad.add_argument("--prefix", default=None)

    recompute = sub.add_parser(
        "recompute-scale",
        help="既存の Optos grader.csv を眼ごとの固定 µm/px で面積だけやり直す",
    )
    recompute.add_argument("reading_dir", type=Path)
    recompute.add_argument("--out", type=Path, default=None)
    recompute.add_argument("--outlier-ratio", type=float, default=None)
    recompute.add_argument("--rerun-adoption", action="store_true")
    recompute.add_argument("--scale-um-px", type=float, default=None)
    recompute.add_argument("--dicom", type=Path, default=None)
    recompute.add_argument("--from-image-width", action="store_true")
    recompute.add_argument("--native-um-per-px", type=float, default=None)
    recompute.add_argument("--native-width-px", type=float, default=None)

    args = parser.parse_args()
    if args.command == "merge":
        summary = merge_graders(args.first_csv, args.second_csv, args.out, prefix=args.prefix)
        print(summary["adopted_csv"])
        print(summary["summary_md"])
    elif args.command == "triad":
        result = resolve_triad(args.summary_md, args.final_csv, args.out, prefix=args.prefix)
        print(result["triad_csv"])
    else:
        argv = [str(args.reading_dir)]
        if args.out is not None:
            argv.extend(["--out", str(args.out)])
        if args.outlier_ratio is not None:
            argv.extend(["--outlier-ratio", str(args.outlier_ratio)])
        if args.rerun_adoption:
            argv.append("--rerun-adoption")
        if args.scale_um_px is not None:
            argv.extend(["--scale-um-px", str(args.scale_um_px)])
        if args.dicom is not None:
            argv.extend(["--dicom", str(args.dicom)])
        if args.from_image_width:
            argv.append("--from-image-width")
        if args.native_um_per_px is not None:
            argv.extend(["--native-um-per-px", str(args.native_um_per_px)])
        if args.native_width_px is not None:
            argv.extend(["--native-width-px", str(args.native_width_px)])
        recompute_scale_main(argv)


if __name__ == "__main__":
    main()
