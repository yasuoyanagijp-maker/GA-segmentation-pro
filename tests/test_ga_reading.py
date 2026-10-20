"""GA 指標での第1・第2の採用と、最終読影の中央値。"""

from pathlib import Path

import numpy as np

from ga_reading.absent import absent_row, write_empty_mask
from ga_reading.bundle import write_grader_bundle
from ga_reading.merge import file_stem, merge_graders, read_csv
from ga_reading.metrics import GA_PRIMARY_METRICS
from ga_reading.queue import find_first_grader_csv, recheck_images, recheck_stems, second_reader_images
from ga_reading.triad import resolve_triad


def _row(name: str, area: str, sqrt: str, present: str = "1", quality: str = "OK") -> dict[str, str]:
    row = {column: "" for column in GA_PRIMARY_METRICS}
    row.update({
        "File": name,
        "lesion present": present,
        "Quality of analysis": quality,
        "area_mm2": area,
        "area_mm_sqrt": sqrt,
        "lesion_count": "1" if present == "1" else "",
        "gawain": "50" if present == "1" else "",
        "dist_um_nearest": "100" if present == "1" else "",
        "gap_um": "80" if present == "1" else "",
        "intact_1mm_mm2": "1.2" if present == "1" else "",
    })
    return row


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["File", "lesion present", "Quality of analysis", *GA_PRIMARY_METRICS]
    from ga_reading.merge import write_csv
    write_csv(path, fields, rows)


def test_role_prefix_does_not_change_the_match_key():
    assert file_stem("第2__2022-02-25.png") == "2022-02-25"
    assert file_stem("最終__2022-02-25.png") == "2022-02-25"
    assert file_stem("2022-02-25.png") == "2022-02-25"


def test_prefixed_reader_files_still_merge_and_resolve(tmp_path: Path):
    first = tmp_path / "g1.csv"
    second = tmp_path / "g2.csv"
    left = _row("2022-02-25.png", "6.0", "2.4")
    right = _row("第2__2022-02-25.png", "6.2", "2.5")
    right["dist_um_nearest"] = "200"
    _write(first, [left])
    _write(second, [right])
    summary = merge_graders(first, second, tmp_path / "out", prefix="GA")
    assert summary["n_matched"] == 1
    assert "2022-02-25.png" in summary["recheck_by_file"]
    final = tmp_path / "final.csv"
    done = _row("最終__2022-02-25.png", "6.1", "2.45")
    done["dist_um_nearest"] = "140"
    _write(final, [done])
    result = resolve_triad(Path(summary["summary_md"]), final, tmp_path / "out")
    _fields, rows = read_csv(Path(result["triad_csv"]))
    assert file_stem(rows[0]["File"]) == "2022-02-25"
    assert float(rows[0]["dist_um_nearest"]) == 140
    assert float(rows[0]["area_mm2"]) == 6.1


def test_primary_metrics_are_the_ga_outputs():
    assert GA_PRIMARY_METRICS == (
        "area_mm2",
        "area_mm_sqrt",
        "lesion_count",
        "gawain",
        "dist_um_nearest",
        "gap_um",
        "intact_1mm_mm2",
    )


def test_blank_is_not_zero_and_both_zeros_average_to_zero(tmp_path: Path):
    first = tmp_path / "g1.csv"
    second = tmp_path / "g2.csv"
    _write(first, [
        _row("a.png", "1.0", "1.0"),
        _row("b.png", "0", "0"),
        _row("c.png", "", "", present="0", quality="N/A"),
    ])
    _write(second, [
        _row("a.png", "1.1", "1.05"),
        _row("b.png", "0", "0"),
        _row("c.png", "2", "1.4"),
    ])
    summary = merge_graders(first, second, tmp_path / "out", prefix="GA")
    _fields, rows = read_csv(Path(summary["adopted_csv"]))
    by_file = {file_stem(row["File"]): row for row in rows}
    assert by_file["a"]["area_mm2"] == "1.05"
    assert by_file["b"]["area_mm2"] == "0"
    assert by_file["c"]["area_mm2"] == "NA"
    assert "c.png" in summary["recheck_by_file"]
    assert "b.png" not in summary["recheck_by_file"]
    text = Path(summary["summary_md"]).read_text(encoding="utf-8")
    assert "c.png" in text
    assert "b.png" not in text.split("## RECHECK", 1)[1]


def test_rpd_over_20_is_na_in_adopted_and_mean_only_in_fallback(tmp_path: Path):
    first = tmp_path / "g1.csv"
    second = tmp_path / "g2.csv"
    _write(first, [_row("wide.png", "1.0", "1.0")])
    _write(second, [_row("wide.png", "2.0", "1.1")])
    summary = merge_graders(first, second, tmp_path / "out", prefix="GA")
    _fields, adopted = read_csv(Path(summary["adopted_csv"]))
    _fields, fallback = read_csv(Path(summary["avg_fallback_csv"]))
    assert adopted[0]["area_mm2"] == "NA"
    assert fallback[0]["area_mm2"] == "1.5"
    assert "提出には使いません" in Path(summary["avg_fallback_csv"]).with_name(
        "GA_avg_fallback_README.txt"
    ).read_text(encoding="utf-8")


def test_triad_keeps_adopted_cells_and_flags_review(tmp_path: Path):
    first = tmp_path / "g1.csv"
    second = tmp_path / "g2.csv"
    close = _row("ok.png", "10", "3")
    close_b = _row("ok.png", "10.5", "3.1")
    far = _row("redo.png", "1", "1")
    far_b = _row("redo.png", "2", "1.1")
    _write(first, [close, far])
    _write(second, [close_b, far_b])
    summary = merge_graders(first, second, tmp_path / "out", prefix="GA")
    final = tmp_path / "final.csv"
    _write(final, [
        _row("ok.png", "99", "9"),
        _row("redo.png", "10", "1.05"),
    ])
    result = resolve_triad(Path(summary["summary_md"]), final, tmp_path / "out")
    _fields, rows = read_csv(Path(result["triad_csv"]))
    by_file = {file_stem(row["File"]): row for row in rows}
    assert float(by_file["ok"]["area_mm2"]) == 10.25
    assert by_file["ok"]["needs_review"] == ""
    assert float(by_file["redo"]["area_mm2"]) == 2
    assert "area_mm2" in by_file["redo"]["needs_review"]
    assert recheck_stems(Path(summary["summary_md"])) == ["redo"]


def test_second_reader_takes_images_not_masks(tmp_path: Path):
    image = tmp_path / "src.png"
    image.write_bytes(b"png")
    mask = np.zeros((4, 4), dtype=np.uint8)
    write_grader_bundle(tmp_path / "grader1", [{
        "path": str(image),
        "filename": "visit.png",
        "shape": (4, 4),
        "mask": mask,
        "lesion_present": "1",
        "case_id": "OD",
        "date": "2026-01-01",
        "area_mm2": 1.0,
        "area_mm_sqrt": 1.0,
        "lesion_count": 1,
        "gawain": 40,
        "dist_um_nearest": 200,
        "gap_um": 180,
        "intact_1mm_mm2": 2.0,
    }])
    images = second_reader_images(tmp_path / "grader1")
    assert [path.name for path in images] == ["visit.png"]
    assert all("masks" not in path.parts for path in images)


def test_second_reader_csv_is_not_the_first_and_recheck_queue_is_narrow(tmp_path: Path):
    import cv2

    root = tmp_path / "grader1"
    images = root / "export" / "images"
    masks = root / "export" / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    cv2.imwrite(str(images / "keep.png"), np.zeros((4, 4), dtype=np.uint8))
    cv2.imwrite(str(images / "skip.png"), np.zeros((4, 4), dtype=np.uint8))
    cv2.imwrite(str(masks / "keep.png"), np.full((4, 4), 255, dtype=np.uint8))
    (root / "grader.csv").write_text("File\nkeep.png\n", encoding="utf-8")
    later = root / "second_reader_output_20260101" / "grader.csv"
    later.parent.mkdir()
    later.write_text("File\n", encoding="utf-8")
    assert find_first_grader_csv(root) == root / "grader.csv"
    opened = second_reader_images(root)
    assert [path.name for path in opened] == ["keep.png", "skip.png"]
    summary = root / "integrated_output_x" / "GA_summary.md"
    summary.parent.mkdir()
    summary.write_text(
        "\n".join([
            "## RECHECK",
            "",
            "- 主要指標セル: 1 件（対象症例 1 件）",
            "- 症例別（NA となった主要指標）:",
            "  - keep.png: area_mm2",
            "",
        ]),
        encoding="utf-8",
    )
    assert [path.name for path in recheck_images(summary)] == ["keep.png"]


def test_triad_uses_the_final_reader_when_both_graders_are_blank(tmp_path: Path):
    first = tmp_path / "g1.csv"
    second = tmp_path / "g2.csv"
    left = _row("gap.png", "4.0", "2.0")
    right = _row("gap.png", "4.1", "2.02")
    left["dist_um_nearest"] = ""
    right["dist_um_nearest"] = ""
    _write(first, [left])
    _write(second, [right])
    summary = merge_graders(first, second, tmp_path / "out", prefix="GA")
    final = tmp_path / "final.csv"
    done = _row("gap.png", "4.05", "2.01")
    done["dist_um_nearest"] = "140"
    _write(final, [done])
    result = resolve_triad(Path(summary["summary_md"]), final, tmp_path / "out")
    _fields, rows = read_csv(Path(result["triad_csv"]))
    assert float(rows[0]["dist_um_nearest"]) == 140
    assert float(rows[0]["area_mm2"]) == 4.05


def test_triad_preview_does_not_write(tmp_path: Path):
    first = tmp_path / "g1.csv"
    second = tmp_path / "g2.csv"
    _write(first, [_row("wide.png", "1.0", "1.0")])
    _write(second, [_row("wide.png", "2.0", "1.1")])
    summary = merge_graders(first, second, tmp_path / "out", prefix="GA")
    final = tmp_path / "final.csv"
    _write(final, [_row("wide.png", "10", "3")])
    out = tmp_path / "final_out"
    preview = resolve_triad(Path(summary["summary_md"]), final, out, dry_run=True)
    assert preview["dry_run"] is True
    assert preview["filled_cells"] >= 1
    assert not (out / "GA_triad_adopted_values.csv").exists()


def test_load_paths_does_not_open_masks(tmp_path: Path):
    import cv2

    from optos_ui.session import OptosSession

    images = tmp_path / "export" / "images"
    masks = tmp_path / "export" / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    cv2.imwrite(str(images / "visit.png"), np.zeros((8, 8), dtype=np.uint8))
    cv2.imwrite(str(masks / "visit.png"), np.full((8, 8), 255, dtype=np.uint8))
    session = OptosSession()
    session.load_paths(second_reader_images(tmp_path), label="第2")
    assert [path.name for path in session.files] == ["visit.png"]
    assert session.files[0].parent.name == "images"


def test_absent_row_leaves_metrics_blank_and_writes_empty_mask(tmp_path: Path):
    row = absent_row("gone.png")
    assert row["lesion present"] == "0"
    assert row["Quality of analysis"] == "N/A"
    assert all(row[column] == "" for column in GA_PRIMARY_METRICS)
    path = write_empty_mask(tmp_path / "gone.png", (3, 5))
    import cv2
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (3, 5)
    assert int(mask.sum()) == 0
