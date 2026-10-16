"""最終読影者は RECHECK の画像だけを読み、3値の中央値で確定する。"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from ga_reading.merge import file_stem, read_csv, write_csv
from ga_reading.metrics import FILE_COL, GA_PRIMARY_METRICS, RPD_THRESHOLD_PCT
from ga_reading.queue import recheck_targets
from ga_reading.rpd import rpd_pct, to_float


def triad_median(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    return float(statistics.median(finite))


def resolve_triad(
    summary_md: Path,
    final_csv: Path,
    out_dir: Path,
    *,
    adopted_csv: Path | None = None,
    first_csv: Path | None = None,
    second_csv: Path | None = None,
    prefix: str | None = None,
    rpd_threshold: float = RPD_THRESHOLD_PCT,
    dry_run: bool = False,
) -> dict[str, Any]:
    """すでに平均で採用したセルは上書きしない。中央値はレビュー印が付いても残す。"""
    summary_md = Path(summary_md)
    targets = recheck_targets(summary_md)
    if adopted_csv is None:
        adopted_csv = _sibling(summary_md, "_summary.md", "_adopted_values.csv")
    source = summary_md.with_name(summary_md.name.replace("_summary.md", "_source_csvs.json"))
    g1_path, g2_path = first_csv, second_csv
    if source.is_file():
        data = json.loads(source.read_text(encoding="utf-8"))
        g1_path = g1_path or Path(data["first_csv"])
        g2_path = g2_path or Path(data["second_csv"])
    if g1_path is None or g2_path is None or adopted_csv is None:
        raise ValueError("第1・第2・採用 CSV が見つかりません。")

    _, adopted_rows = read_csv(Path(adopted_csv))
    _, g1_rows = read_csv(Path(g1_path))
    _, g2_rows = read_csv(Path(g2_path))
    _, final_rows = read_csv(Path(final_csv))
    g1 = _by_stem(g1_rows)
    g2 = _by_stem(g2_rows)
    final = _by_stem(final_rows)
    queue = {file_stem(name) for name, _metrics in targets}

    fields = list(read_csv(Path(adopted_csv))[0])
    if "needs_review" not in fields:
        fields.append("needs_review")
    resolved: list[dict[str, Any]] = []
    preview: list[str] = []
    review_cells = 0
    filled = 0
    for row in adopted_rows:
        stem = file_stem(row.get(FILE_COL, ""))
        out = dict(row)
        flags: list[str] = []
        if stem in queue:
            for column in GA_PRIMARY_METRICS:
                text = str(out.get(column, "")).strip()
                if text.upper() != "NA" and text != "":
                    continue
                fr = to_float((final.get(stem) or {}).get(column))
                a = to_float((g1.get(stem) or {}).get(column))
                b = to_float((g2.get(stem) or {}).get(column))
                if fr is None or (a is None and b is None):
                    out[column] = "NA" if text.upper() == "NA" else ""
                    continue
                median = triad_median([a, b, fr])
                if median is None:
                    out[column] = "NA"
                    continue
                out[column] = f"{median:.10g}"
                filled += 1
                score = rpd_pct(median, fr)
                marked = score is None or score > rpd_threshold
                if marked:
                    flags.append(column)
                    review_cells += 1
                preview.append(
                    f"{out.get(FILE_COL, stem)}  {column}={out[column]}"
                    + ("  要レビュー" if marked else "")
                )
        out["needs_review"] = ";".join(flags)
        resolved.append(out)

    out_dir = Path(out_dir)
    stem_name = summary_md.name[: -len("_summary.md")] if summary_md.name.endswith("_summary.md") else summary_md.stem
    prefix = prefix or stem_name
    path = out_dir / f"{prefix}_triad_adopted_values.csv"
    if dry_run:
        return {
            "triad_csv": str(path),
            "filled_cells": filled,
            "review_cells": review_cells,
            "recheck_images": len(queue),
            "preview": preview,
            "dry_run": True,
        }
    write_csv(path, fields, resolved)
    note = out_dir / f"{prefix}_triad_summary.md"
    note.write_text(
        "\n".join([
            f"# GA 3者確定 — {datetime.now().date().isoformat()}",
            "",
            f"- 入力: `{summary_md}`",
            f"- 最終読影 CSV: `{final_csv}`",
            f"- RECHECK 画像: {len(queue)}",
            f"- 中央値で埋めたセル: {filled}",
            f"- 要レビュー（中央値と最終読影者の RPD が {rpd_threshold:g}% 超）: {review_cells}",
            "- 値は採用したまま、needs_review に列名を付ける。",
            "- すでに採用済みのセルは上書きしない。",
            "- 正式な確定ファイルは *_triad_adopted_values.csv。",
            "",
        ]),
        encoding="utf-8",
    )
    return {
        "triad_csv": str(path),
        "filled_cells": filled,
        "review_cells": review_cells,
        "recheck_images": len(queue),
        "preview": preview,
        "dry_run": False,
    }


def _sibling(summary: Path, old: str, new: str) -> Path | None:
    if summary.name.endswith(old):
        candidate = summary.with_name(summary.name.replace(old, new))
        if candidate.is_file():
            return candidate
    return None


def _by_stem(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        file_stem(row.get(FILE_COL, "") or row.get("filename", "")): row
        for row in rows
        if file_stem(row.get(FILE_COL, "") or row.get("filename", ""))
    }
