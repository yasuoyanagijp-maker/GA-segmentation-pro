"""既存の grader.csv を、眼ごとに固定した µm/px で面積だけやり直す。

第1読影は export/masks の画素数から、当時の µm/px を逆算して中央値を取る。
第2・最終はマスクが無いので、同じ画像の (s_fixed / s_original)² を面積に掛ける。
採用ルール（20% 以内は平均、それ以外は3者の中央値）は、直した CSV を
``python -m ga_reading merge`` / ``triad`` に渡せば再実行できる。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from engine.eye_scale import (
    SCALE_OUTLIER_RATIO,
    apply_area_correction,
    area_correction_factor,
    area_mm2_from_mask_px,
    area_mm_sqrt,
    fixed_eye_scale,
    is_scale_outlier,
    recover_scale_um_px,
)
from ga_reading.bundle import grader_csv_fields
from ga_reading.merge import file_stem, merge_graders, read_csv, write_csv
from ga_reading.metrics import FILE_COL, SCALE_EXPORT_COLUMNS
from ga_reading.queue import find_first_grader_csv, first_reader_dir, latest_workflow_csv
from ga_reading.rpd import sync_area_sqrt, to_float
from ga_reading.triad import resolve_triad


def mask_pixel_count(path: Path) -> int:
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"マスクを読めません: {path}")
    return int(np.count_nonzero(np.asarray(image) > 0))


def find_mask(masks_dir: Path, row: dict[str, str]) -> Path | None:
    """export/masks から、行の File / filename に対応する PNG を探す。"""
    masks_dir = Path(masks_dir)
    if not masks_dir.is_dir():
        return None
    names = [row.get(FILE_COL, ""), row.get("filename", "")]
    tried: list[Path] = []
    for name in names:
        if not str(name).strip():
            continue
        stem = Path(str(name)).stem
        tried.append(masks_dir / f"{stem}.png")
        if "__" in stem:
            tried.append(masks_dir / f"{stem.split('__', 1)[1]}.png")
    for path in tried:
        if path.is_file():
            return path
    key = file_stem(row.get(FILE_COL, "") or row.get("filename", ""))
    if not key:
        return None
    for path in sorted(masks_dir.iterdir(), key=lambda item: item.name.lower()):
        if path.suffix.lower() == ".png" and file_stem(path.name) == key:
            return path
    return None


def _row_key(row: dict[str, str]) -> str:
    return file_stem(row.get(FILE_COL, "") or row.get("filename", ""))


def _append_review_flag(row: dict[str, Any], flag: str) -> None:
    current = str(row.get("needs_review") or "")
    parts = [part for part in current.split(";") if part]
    if flag not in parts:
        parts.append(flag)
    row["needs_review"] = ";".join(parts)


def _format_scale(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def recover_row_scale(row: dict[str, str], mask_px: int | None) -> float | None:
    """その行で面積に使われた µm/px。マスクと面積から逆算する。"""
    area = to_float(row.get("area_mm2"))
    if mask_px and mask_px > 0 and area is not None and area > 0:
        return recover_scale_um_px(area, mask_px)
    for column in ("scale_estimate_um_per_px", "scale_um_per_px"):
        value = to_float(row.get(column))
        if value is not None and value > 0:
            return value
    return None


def collect_first_reader_factors(
    rows: list[dict[str, str]],
    masks_dir: Path,
    *,
    outlier_ratio: float = SCALE_OUTLIER_RATIO,
) -> tuple[float | None, dict[str, dict[str, Any]]]:
    """第1の各行について、推定スケールとマスク画素数を集める。"""
    collected: dict[str, dict[str, Any]] = {}
    estimates: list[float] = []
    for row in rows:
        key = _row_key(row)
        if not key:
            continue
        mask_path = find_mask(masks_dir, row)
        mask_px = mask_pixel_count(mask_path) if mask_path is not None else None
        estimate = recover_row_scale(row, mask_px)
        if estimate is not None:
            estimates.append(estimate)
        collected[key] = {
            "mask_px": mask_px,
            "estimate": estimate,
            "area_original": to_float(row.get("area_mm2")),
        }
    fixed = fixed_eye_scale(estimates)
    for item in collected.values():
        estimate = item["estimate"]
        item["fixed"] = fixed
        item["outlier"] = is_scale_outlier(estimate, fixed, ratio=outlier_ratio)
        if estimate is not None and fixed is not None:
            item["factor"] = area_correction_factor(estimate, fixed)
        else:
            item["factor"] = None
    return fixed, collected


def apply_fixed_scale_to_row(
    row: dict[str, str],
    info: dict[str, Any],
) -> dict[str, str]:
    """1行の面積を固定スケールで書き直す。他の主要指標は触らない。"""
    out = dict(row)
    estimate = info.get("estimate")
    fixed = info.get("fixed")
    factor = info.get("factor")
    mask_px = info.get("mask_px")
    area_original = info.get("area_original")
    area = None
    if mask_px and mask_px > 0 and fixed is not None:
        area = area_mm2_from_mask_px(int(mask_px), float(fixed))
    elif area_original is not None and factor is not None:
        area = apply_area_correction(area_original, factor)
    if area is not None:
        out["area_mm2"] = f"{round(area, 3):.10g}"
        out["area_mm_sqrt"] = f"{round(area_mm_sqrt(area), 2):.10g}"
    elif str(out.get("area_mm2", "")).strip().upper() == "NA":
        sync_area_sqrt(out)
    if info.get("outlier"):
        _append_review_flag(out, "scale_outlier")
    out["scale_um_per_px"] = _format_scale(fixed)
    out["scale_estimate_um_per_px"] = _format_scale(estimate)
    out["mask_px"] = "" if mask_px is None else str(int(mask_px))
    out["scale_outlier"] = "1" if info.get("outlier") else "0"
    return out


def apply_factor_to_reader_row(
    row: dict[str, str],
    info: dict[str, Any] | None,
) -> dict[str, str]:
    """マスクの無い第2・最終行。第1と同じ日付の補正係数を面積に掛ける。"""
    out = dict(row)
    if info is None:
        return out
    area = to_float(row.get("area_mm2"))
    factor = info.get("factor")
    if area is not None and factor is not None:
        corrected = apply_area_correction(area, factor)
        out["area_mm2"] = f"{round(corrected, 3):.10g}"
        out["area_mm_sqrt"] = f"{round(area_mm_sqrt(corrected), 2):.10g}"
    if info.get("outlier"):
        _append_review_flag(out, "scale_outlier")
    out["scale_um_per_px"] = _format_scale(info.get("fixed"))
    out["scale_estimate_um_per_px"] = _format_scale(info.get("estimate"))
    out["mask_px"] = ""
    out["scale_outlier"] = "1" if info.get("outlier") else "0"
    return out


def _output_fields(original: list[str]) -> list[str]:
    fields = list(original)
    required = grader_csv_fields()
    for column in required:
        if column not in fields:
            fields.append(column)
    return fields


def _factor_rows(first_rows: list[dict[str, str]], collected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in first_rows:
        key = _row_key(row)
        info = collected.get(key) or {}
        rows.append({
            "File": row.get(FILE_COL, ""),
            "date": row.get("date", ""),
            "mask_px": "" if info.get("mask_px") is None else info.get("mask_px"),
            "scale_estimate_um_per_px": _format_scale(info.get("estimate")),
            "scale_um_per_px": _format_scale(info.get("fixed")),
            "correction_factor": (
                "" if info.get("factor") is None else f"{float(info['factor']):.8g}"
            ),
            "scale_outlier": "1" if info.get("outlier") else "0",
            "area_mm2_original": row.get("area_mm2", ""),
        })
    return rows


def recompute_reading_dir(
    reading_dir: Path,
    *,
    out_dir: Path | None = None,
    outlier_ratio: float = SCALE_OUTLIER_RATIO,
    rerun_adoption: bool = False,
) -> dict[str, Any]:
    """1眼の読影フォルダ（例: /path/case2/OD_reading）を直す。"""
    reading_dir = Path(reading_dir).expanduser().resolve()
    root = first_reader_dir(reading_dir)
    first_csv = find_first_grader_csv(root)
    if first_csv is None:
        raise ValueError(f"grader.csv が見つかりません: {reading_dir}")
    masks_dir = root / "export" / "masks"
    if not masks_dir.is_dir():
        raise ValueError(f"export/masks が見つかりません: {root}")

    first_fields, first_rows = read_csv(first_csv)
    fixed, collected = collect_first_reader_factors(
        first_rows, masks_dir, outlier_ratio=outlier_ratio,
    )
    if fixed is None:
        raise ValueError("マスクと面積から µm/px を復元できる行がありません")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = Path(out_dir) if out_dir is not None else root / f"scale_fixed_output_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)

    corrected_first = [
        apply_fixed_scale_to_row(row, collected.get(_row_key(row)) or {"fixed": fixed})
        for row in first_rows
    ]
    first_out = dest / "grader.csv"
    write_csv(first_out, _output_fields(first_fields), corrected_first)
    write_csv(
        dest / "scale_factors.csv",
        [
            "File",
            "date",
            "mask_px",
            "scale_estimate_um_per_px",
            "scale_um_per_px",
            "correction_factor",
            "scale_outlier",
            "area_mm2_original",
        ],
        _factor_rows(first_rows, collected),
    )

    second_csv = latest_workflow_csv(root, "second_reader_output")
    second_out = None
    if second_csv is not None:
        second_fields, second_rows = read_csv(second_csv)
        corrected_second = [
            apply_factor_to_reader_row(row, collected.get(_row_key(row)))
            for row in second_rows
        ]
        second_out = dest / "second_grader.csv"
        write_csv(second_out, _output_fields(second_fields), corrected_second)

    final_csv = latest_workflow_csv(root, "final_reader_output")
    final_out = None
    if final_csv is not None:
        final_fields, final_rows = read_csv(final_csv)
        corrected_final = [
            apply_factor_to_reader_row(row, collected.get(_row_key(row)))
            for row in final_rows
        ]
        final_out = dest / "final_grader.csv"
        write_csv(final_out, _output_fields(final_fields), corrected_final)

    adoption: dict[str, Any] | None = None
    if rerun_adoption and second_out is not None:
        merge_dir = dest / "integrated_output"
        summary = merge_graders(first_out, second_out, merge_dir, prefix="GA")
        adoption = {"merge": summary}
        if final_out is not None:
            adoption["triad"] = resolve_triad(
                Path(summary["summary_md"]),
                final_out,
                merge_dir,
                prefix="GA",
            )

    readme = dest / "README.txt"
    readme.write_text(_readme_text(root, first_csv, second_csv, final_csv, dest, fixed), encoding="utf-8")
    n_outlier = sum(1 for item in collected.values() if item.get("outlier"))
    return {
        "reading_dir": str(root),
        "out_dir": str(dest),
        "first_csv": str(first_out),
        "second_csv": None if second_out is None else str(second_out),
        "final_csv": None if final_out is None else str(final_out),
        "scale_factors_csv": str(dest / "scale_factors.csv"),
        "eye_scale_um_per_px": fixed,
        "n_visits": len(collected),
        "n_outliers": n_outlier,
        "outlier_ratio": outlier_ratio,
        "adoption": adoption,
    }


def _readme_text(
    root: Path,
    first_csv: Path,
    second_csv: Path | None,
    final_csv: Path | None,
    dest: Path,
    fixed: float,
) -> str:
    lines = [
        "Optos 面積を、この眼の固定 µm/px（時点推定の中央値）でやり直した結果です。",
        f"固定スケール: {fixed:.4f} µm/px",
        f"第1入力: {first_csv}",
        f"第2入力: {second_csv or 'なし'}",
        f"最終入力: {final_csv or 'なし'}",
        "",
        "第1は export/masks の画素数 × (固定スケール)² で area_mm2 を出し直しています。",
        "第2・最終はマスクが無いので、同じ画像の (s_fixed / s_original)² を面積に掛けています。",
        "元の grader.csv は上書きしていません。",
        "",
        "採用ルールをやり直すとき:",
        f"  python -m ga_reading merge {dest / 'grader.csv'} {dest / 'second_grader.csv'} --out {dest / 'integrated_output'}",
        "最終読影があるとき、そのあと:",
        f"  python -m ga_reading triad <summary.md> {dest / 'final_grader.csv'} --out {dest / 'integrated_output'}",
        "または最初から --rerun-adoption を付けます。",
        "",
        f"読影フォルダ: {root}",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="既存の Optos grader.csv を、眼ごとの固定 µm/px で面積だけやり直す",
    )
    parser.add_argument(
        "reading_dir",
        type=Path,
        help="第1読影フォルダ（例: /path/case2/OD_reading）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力フォルダ。省略時は reading_dir/scale_fixed_output_<時刻>",
    )
    parser.add_argument(
        "--outlier-ratio",
        type=float,
        default=SCALE_OUTLIER_RATIO,
        help="中央値からこの比率を超えた時点推定に scale_outlier を付ける（既定 0.03）",
    )
    parser.add_argument(
        "--rerun-adoption",
        action="store_true",
        help="直した第1・第2（と最終）で merge / triad を再実行する",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    result = recompute_reading_dir(
        args.reading_dir,
        out_dir=args.out,
        outlier_ratio=args.outlier_ratio,
        rerun_adoption=args.rerun_adoption,
    )
    print(f"eye_scale_um_per_px={result['eye_scale_um_per_px']:.4f}")
    print(f"first_csv={result['first_csv']}")
    if result["second_csv"]:
        print(f"second_csv={result['second_csv']}")
    if result["final_csv"]:
        print(f"final_csv={result['final_csv']}")
    print(f"scale_factors_csv={result['scale_factors_csv']}")
    print(f"outliers={result['n_outliers']}/{result['n_visits']}")
    if result["adoption"] and result["adoption"].get("merge"):
        print(f"adopted_csv={result['adoption']['merge']['adopted_csv']}")
        triad = result["adoption"].get("triad")
        if triad:
            print(f"triad_csv={triad['triad_csv']}")
    return result


if __name__ == "__main__":
    main()
