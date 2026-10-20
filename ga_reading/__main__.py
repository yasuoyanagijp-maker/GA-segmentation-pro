"""第1・第2の統合と、最終読影の確定。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ga_reading.merge import merge_graders
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

    args = parser.parse_args()
    if args.command == "merge":
        summary = merge_graders(args.first_csv, args.second_csv, args.out, prefix=args.prefix)
        print(summary["adopted_csv"])
        print(summary["summary_md"])
    else:
        result = resolve_triad(args.summary_md, args.final_csv, args.out, prefix=args.prefix)
        print(result["triad_csv"])


if __name__ == "__main__":
    main()
