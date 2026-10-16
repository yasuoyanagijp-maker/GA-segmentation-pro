"""第1と第2の CSV をファイル名 stem で突合し、採用値を書く。"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ga_reading.metrics import (
    FILE_COL,
    GA_PRIMARY_METRICS,
    LESION_PRESENT,
    QUALITY,
    RPD_THRESHOLD_PCT,
)
from ga_reading.rpd import adopt_pair, to_float

RECHECK_FIELDS = (
    "File",
    "Metric",
    "Value_grader1",
    "Value_reader2",
    "RPD_pct",
    "Adopted",
    "Rule",
)


def file_stem(name: str) -> str:
    return Path(str(name or "").strip()).stem.lower()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [{k: (v if v is not None else "") for k, v in row.items()} for row in reader]
    if not fields:
        raise ValueError(f"CSV にヘッダがありません: {path}")
    return fields, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _index(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], list[str]]:
    index: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        key = file_stem(row.get(FILE_COL, "")) or file_stem(row.get("filename", ""))
        if not key:
            continue
        if key in index:
            duplicates.append(key)
        index[key] = row
    return index, duplicates


def _same_text(left: str, right: str) -> str:
    if left == right:
        return left
    if left and right:
        return "NA"
    return left or right


def merge_graders(
    first_csv: Path,
    second_csv: Path,
    out_dir: Path,
    *,
    prefix: str | None = None,
    rpd_threshold: float = RPD_THRESHOLD_PCT,
) -> dict[str, Any]:
    """正式な採用は *_adopted_values.csv。平均で埋めた CSV は参考だけ。"""
    first_csv = Path(first_csv)
    second_csv = Path(second_csv)
    fields_a, rows_a = read_csv(first_csv)
    fields_b, rows_b = read_csv(second_csv)
    fieldnames = list(fields_a)
    for column in list(fields_b) + [LESION_PRESENT, QUALITY, *GA_PRIMARY_METRICS]:
        if column not in fieldnames:
            fieldnames.append(column)

    index_a, dup_a = _index(rows_a)
    index_b, dup_b = _index(rows_b)
    warnings: list[str] = []
    if dup_a:
        warnings.append("第1に同じファイル名が重複しています（後の行を使います）")
    if dup_b:
        warnings.append("第2に同じファイル名が重複しています（後の行を使います）")

    order = []
    seen: set[str] = set()
    for row in rows_a:
        key = file_stem(row.get(FILE_COL, "")) or file_stem(row.get("filename", ""))
        if key and key in index_b and key not in seen:
            seen.add(key)
            order.append(key)
    if not order:
        raise ValueError("第1と第2でファイル名 stem が一致する行がありません。")

    numeric = list(GA_PRIMARY_METRICS)
    adopted_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    recheck_rows: list[dict[str, str]] = []

    for key in order:
        left, right = index_a[key], index_b[key]
        out = {column: "" for column in fieldnames}
        out[FILE_COL] = left.get(FILE_COL) or right.get(FILE_COL) or left.get("filename") or key
        for column in fieldnames:
            if column in numeric or column in {FILE_COL, LESION_PRESENT, QUALITY}:
                continue
            out[column] = left.get(column, "")
        out[LESION_PRESENT] = _same_text(
            str(left.get(LESION_PRESENT, "")).strip(),
            str(right.get(LESION_PRESENT, "")).strip(),
        )
        out[QUALITY] = _same_text(
            str(left.get(QUALITY, "")).strip(),
            str(right.get(QUALITY, "")).strip(),
        )
        filled: dict[str, str] = {}
        for column in numeric:
            a = to_float(left.get(column))
            b = to_float(right.get(column))
            value, status, score = adopt_pair(a, b, rpd_threshold)
            out[column] = value
            if value == "NA" and a is not None and b is not None:
                filled[column] = f"{(a + b) / 2.0:.10g}"
            if status in {"RECHECK", "MISSING"}:
                recheck_rows.append({
                    "File": out[FILE_COL],
                    "Metric": column,
                    "Value_grader1": "" if a is None else f"{a:.10g}",
                    "Value_reader2": "" if b is None else f"{b:.10g}",
                    "RPD_pct": score,
                    "Adopted": "NA",
                    "Rule": "MISSING" if status == "MISSING" else f"RPD>{rpd_threshold:g}%",
                })
        adopted_rows.append(out)
        fallback = dict(out)
        fallback.update(filled)
        fallback["avg_filled_columns"] = ";".join(filled)
        fallback_rows.append(fallback)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = prefix or f"GA_integrated_{stamp}"
    adopted_path = out_dir / f"{prefix}_adopted_values.csv"
    fallback_path = out_dir / f"{prefix}_avg_fallback.csv"
    readme_path = out_dir / f"{prefix}_avg_fallback_README.txt"
    recheck_path = out_dir / f"{prefix}_recheck_list.csv"
    summary_path = out_dir / f"{prefix}_summary.md"
    source_path = out_dir / f"{prefix}_source_csvs.json"

    write_csv(adopted_path, fieldnames, adopted_rows)
    write_csv(fallback_path, fieldnames + ["avg_filled_columns"], fallback_rows)
    write_csv(recheck_path, list(RECHECK_FIELDS), recheck_rows)
    readme_path.write_text(
        "\n".join([
            f"{fallback_path.name} は参考だけです。提出には使いません。",
            "RPD が 20% を超えたセルを、第1と第2の単純平均で埋めています。",
            "提出には *_adopted_values.csv を使い、NA の確定は *_triad_adopted_values.csv を使います。",
            "",
        ]),
        encoding="utf-8",
    )
    source_path.write_text(
        json.dumps(
            {"first_csv": str(first_csv.resolve()), "second_csv": str(second_csv.resolve())},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    by_file: dict[str, list[str]] = {}
    for row in recheck_rows:
        by_file.setdefault(row["File"], []).append(row["Metric"])
    only_first = sorted(set(index_a) - set(index_b))
    only_second = sorted(set(index_b) - set(index_a))
    summary = {
        "first_csv": str(first_csv),
        "second_csv": str(second_csv),
        "threshold_pct": float(rpd_threshold),
        "n_matched": len(order),
        "n_first_only": len(only_first),
        "n_second_only": len(only_second),
        "recheck_cells": len(recheck_rows),
        "recheck_files": len(by_file),
        "recheck_by_file": by_file,
        "warnings": warnings,
        "adopted_csv": str(adopted_path),
        "avg_fallback_csv": str(fallback_path),
    }
    summary_path.write_text(_summary_md(summary), encoding="utf-8")
    summary["summary_md"] = str(summary_path)
    summary["recheck_csv"] = str(recheck_path)
    return summary


def _summary_md(summary: dict[str, Any]) -> str:
    lines = [
        f"# GA 統合（第1・第2） — {datetime.now().date().isoformat()}",
        "",
        f"- 第1 CSV: `{summary['first_csv']}`",
        f"- 第2 CSV: `{summary['second_csv']}`",
        f"- RPD 閾値: **{summary['threshold_pct']:g}%**",
        f"- 突合成功: **{summary['n_matched']}** 行"
        f"（第1のみ {summary['n_first_only']} / 第2のみ {summary['n_second_only']}）",
        "",
        "## ルール",
        "",
        "1. CSV の File のファイル名 stem で突合する。",
        "2. 両方有限で RPD が 20% 以下なら算術平均。両方 0 は 0。",
        "3. 20% 超、またはどちらかが空欄なら NA。空欄を 0 とはみなさない。",
        "4. 主要指標の NA はその画像を RECHECK にする。",
        "5. *_avg_fallback.csv は参考だけで、提出には使わない。",
        "",
        "## RECHECK",
        "",
        f"- 主要指標セル: {summary['recheck_cells']} 件（対象症例 {summary['recheck_files']} 件）",
    ]
    by_file = summary["recheck_by_file"]
    if by_file:
        lines.append("- 症例別（NA となった主要指標）:")
        for name in sorted(by_file):
            lines.append(f"  - {name}: {', '.join(by_file[name])}")
    else:
        lines.append("- 症例別（NA となった主要指標）:")
        lines.append("  - (なし)")
    if summary["warnings"]:
        lines += ["", "## 警告", ""]
        lines.extend(f"- {item}" for item in summary["warnings"])
    lines.append("")
    return "\n".join(lines)
