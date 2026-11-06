"""眼ごとの固定 µm/px。合成データのみ。"""

from pathlib import Path

import cv2
import numpy as np
import pytest

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
from ga_reading.bundle import grader_csv_fields, write_grader_bundle
from ga_reading.merge import read_csv
from ga_reading.metrics import GA_PRIMARY_METRICS, SCALE_EXPORT_COLUMNS
from ga_reading.recompute_scale import recompute_reading_dir
from optos_ui.session import OptosSession, VisitSnap


def test_median_is_the_eye_scale():
    assert fixed_eye_scale([52.8, 55.5, 55.7, 55.8, 55.6, 55.0, 56.5, 55.2, 55.8, 57.9]) == pytest.approx(55.65)
    assert fixed_eye_scale([55.2]) == 55.2
    assert fixed_eye_scale([]) is None
    assert fixed_eye_scale([0.0, -1.0]) is None


def test_outlier_is_three_percent_from_the_median():
    assert is_scale_outlier(55.7, 55.7) is False
    assert is_scale_outlier(55.7 * (1 + SCALE_OUTLIER_RATIO - 0.001), 55.7) is False
    assert is_scale_outlier(57.9, 55.7) is True
    assert is_scale_outlier(52.2, 55.7) is True
    assert is_scale_outlier(None, 55.7) is False


def test_area_and_recovered_scale_round_trip():
    scale = recover_scale_um_px(9.575, 3077)
    assert scale == pytest.approx(55.78, abs=0.02)
    assert area_mm2_from_mask_px(3077, scale) == pytest.approx(9.575)
    factor = area_correction_factor(57.9, 55.7)
    assert apply_area_correction(10.471, factor) == pytest.approx(
        10.471 * (55.7 / 57.9) ** 2,
    )
    assert area_mm_sqrt(9.0) == 3.0


def test_same_mask_pixels_keep_the_same_area_when_estimates_drift():
    """画素がほぼ同じでも、時点推定がぶれると面積だけ跳ねる、を止める。"""
    session = OptosSession()
    session.case_id = "OD"
    early = Path("2022-02-25.png")
    mid = Path("2023-08-25.png")
    late = Path("2026-10-05.png")
    session.files = [early, mid, late]
    session.index = 2
    session.date_iso = "2026-10-05"
    session.gray = np.full((80, 80), 180, dtype=np.uint8)
    session.disc_xy = (60.0, 40.0)
    session.fovea_xy = (30.0, 40.0)
    session.fovea_user_placed = True
    session.scale_um_px = 57.9
    session.mask = np.zeros((80, 80), dtype=np.uint8)
    session.mask[20:38, 20:38] = 255
    early_mask = np.zeros((80, 80), dtype=np.uint8)
    early_mask[20:37, 20:38] = 255
    mid_mask = early_mask.copy()
    session.snaps["OD/2022-02-25.png"] = VisitSnap(
        filename=early.name,
        path=str(early),
        date_iso="2022-02-25",
        date_source="filename",
        gray=session.gray,
        disc_xy=(60.0, 40.0),
        fovea_xy=(30.0, 40.0),
        scale_um_px=55.8,
        mask=early_mask,
        confirmed=True,
        fovea_user_placed=True,
    )
    session.snaps["OD/2023-08-25.png"] = VisitSnap(
        filename=mid.name,
        path=str(mid),
        date_iso="2023-08-25",
        date_source="filename",
        gray=session.gray,
        disc_xy=(60.0, 40.0),
        fovea_xy=(30.0, 40.0),
        scale_um_px=55.6,
        mask=mid_mask,
        confirmed=True,
        fovea_user_placed=True,
    )
    session.confirmed = True
    rows = session.visit_rows()
    by_date = {row["date"]: row for row in rows}
    used = session.eye_scale_um_px()
    assert used == pytest.approx(55.8)
    assert by_date["2022-02-25"]["scale_um_per_px"] == pytest.approx(used)
    assert by_date["2026-10-05"]["scale_um_per_px"] == pytest.approx(used)
    assert by_date["2026-10-05"]["scale_estimate_um_per_px"] == pytest.approx(57.9)
    assert by_date["2026-10-05"]["scale_outlier"] == 1
    assert by_date["2026-10-05"]["scale_source"] == "eye_median"
    late_px = int(np.count_nonzero(session.mask))
    early_px = int(np.count_nonzero(early_mask))
    assert by_date["2026-10-05"]["mask_px"] == late_px
    pixel_ratio = late_px / early_px
    area_ratio = by_date["2026-10-05"]["area_mm2"] / by_date["2022-02-25"]["area_mm2"]
    assert area_ratio == pytest.approx(pixel_ratio, rel=0.02)
    drifted = late_px * (57.9 ** 2) / 1e6
    assert by_date["2026-10-05"]["area_mm2"] < drifted * 0.97


def test_grader_csv_keeps_old_columns_and_appends_scale(tmp_path: Path):
    image = tmp_path / "visit.png"
    cv2.imwrite(str(image), np.zeros((8, 8), dtype=np.uint8))
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:4, 1:4] = 255
    write_grader_bundle(tmp_path / "grader1", [{
        "path": str(image),
        "filename": "visit.png",
        "shape": (8, 8),
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
        "scale_um_per_px": 55.7,
        "mask_px": 9,
        "scale_estimate_um_per_px": 57.9,
        "scale_outlier": 1,
        "needs_review": "scale_outlier",
    }])
    fields, rows = read_csv(tmp_path / "grader1" / "grader.csv")
    expected_head = [
        "File",
        "lesion present",
        "Quality of analysis",
        "case_id",
        "date",
        "filename",
        *GA_PRIMARY_METRICS,
        "needs_review",
    ]
    assert fields[: len(expected_head)] == expected_head
    assert fields[-len(SCALE_EXPORT_COLUMNS):] == list(SCALE_EXPORT_COLUMNS)
    assert grader_csv_fields() == fields
    assert rows[0]["scale_um_per_px"] == "55.7000"
    assert rows[0]["mask_px"] == "9"
    assert rows[0]["scale_estimate_um_per_px"] == "57.9000"
    assert rows[0]["scale_outlier"] == "1"


def _write_mask(path: Path, n: int) -> None:
    image = np.zeros((64, 64), dtype=np.uint8)
    image[4:4 + n, 4:4 + n] = 255
    cv2.imwrite(str(path), image)


def test_recompute_script_fixes_first_and_second_without_second_masks(tmp_path: Path):
    reading = tmp_path / "OD_reading"
    masks = reading / "export" / "masks"
    images = reading / "export" / "images"
    masks.mkdir(parents=True)
    images.mkdir(parents=True)
    visits = [
        ("2022-02-25.png", 55.8, 20),
        ("2023-04-28.png", 55.7, 21),
        ("2023-08-25.png", 55.6, 21),
        ("2026-10-05.png", 57.9, 22),
    ]
    from ga_reading.merge import write_csv
    from ga_reading.metrics import FILE_COL

    first_rows = []
    for name, scale, side in visits:
        _write_mask(masks / name, side)
        cv2.imwrite(str(images / name), np.zeros((64, 64), dtype=np.uint8))
        px = side * side
        area = area_mm2_from_mask_px(px, scale)
        first_rows.append({
            FILE_COL: name,
            "lesion present": "1",
            "Quality of analysis": "OK",
            "case_id": "OD",
            "date": name[:10],
            "filename": name,
            "area_mm2": f"{area:.10g}",
            "area_mm_sqrt": f"{area_mm_sqrt(area):.10g}",
            "lesion_count": "1",
            "gawain": "40",
            "dist_um_nearest": "200",
            "gap_um": "180",
            "intact_1mm_mm2": "1.2",
            "needs_review": "",
        })
    first_fields = [
        FILE_COL, "lesion present", "Quality of analysis", "case_id", "date", "filename",
        *GA_PRIMARY_METRICS, "needs_review",
    ]
    write_csv(reading / "grader.csv", first_fields, first_rows)

    second_dir = reading / "second_reader_output_20260101_000000"
    second_dir.mkdir()
    second_rows = []
    for name, scale, side in visits:
        px = side * side
        # 第2はマスクを持たない。同じ誤った時点スケールで少し違う面積。
        area = area_mm2_from_mask_px(px, scale) * 1.02
        second_rows.append({
            FILE_COL: name,
            "lesion present": "1",
            "Quality of analysis": "OK",
            "area_mm2": f"{area:.10g}",
            "area_mm_sqrt": f"{area_mm_sqrt(area):.10g}",
            "lesion_count": "1",
            "gawain": "41",
            "dist_um_nearest": "210",
            "gap_um": "190",
            "intact_1mm_mm2": "1.1",
            "needs_review": "",
        })
    write_csv(second_dir / "grader.csv", [FILE_COL, "lesion present", "Quality of analysis", *GA_PRIMARY_METRICS, "needs_review"], second_rows)

    out = tmp_path / "fixed"
    result = recompute_reading_dir(reading, out_dir=out, rerun_adoption=True)
    assert result["n_visits"] == 4
    assert result["n_outliers"] == 1
    assert result["eye_scale_um_per_px"] == pytest.approx(55.75)
    fields, rows = read_csv(Path(result["first_csv"]))
    assert fields[: len(first_fields)] == first_fields
    by_file = {row["File"]: row for row in rows}
    late = by_file["2026-10-05.png"]
    assert late["scale_outlier"] == "1"
    assert "scale_outlier" in late["needs_review"]
    assert float(late["scale_um_per_px"]) == pytest.approx(55.75)
    assert int(late["mask_px"]) == 484
    expected_late = area_mm2_from_mask_px(484, 55.75)
    assert float(late["area_mm2"]) == pytest.approx(round(expected_late, 3))
    # 画素 +10% 前後でも、誤った 57.9 では面積が跳ねない。
    drifted = area_mm2_from_mask_px(484, 57.9)
    assert float(late["area_mm2"]) < drifted * 0.96

    _fields, second = read_csv(Path(result["second_csv"]))
    second_late = {row["File"]: row for row in second}["2026-10-05.png"]
    factor = (55.75 / 57.9) ** 2
    original_second = area_mm2_from_mask_px(484, 57.9) * 1.02
    assert float(second_late["area_mm2"]) == pytest.approx(round(original_second * factor, 3))
    assert second_late["mask_px"] == ""

    assert result["adoption"] is not None
    adopted = Path(result["adoption"]["merge"]["adopted_csv"])
    assert adopted.is_file()
    _fields, adopted_rows = read_csv(adopted)
    adopted_late = {row["File"]: row for row in adopted_rows}["2026-10-05.png"]
    # 2者は 20% 以内なので平均。
    first_area = float(late["area_mm2"])
    second_area = float(second_late["area_mm2"])
    assert float(adopted_late["area_mm2"]) == pytest.approx((first_area + second_area) / 2.0)


def test_recompute_then_triad_uses_the_corrected_areas(tmp_path: Path):
    reading = tmp_path / "OS_reading"
    masks = reading / "export" / "masks"
    images = reading / "export" / "images"
    masks.mkdir(parents=True)
    images.mkdir(parents=True)
    name = "2026-09-07.png"
    _write_mask(masks / name, 18)
    cv2.imwrite(str(images / name), np.zeros((64, 64), dtype=np.uint8))
    px = 18 * 18
    from ga_reading.merge import write_csv
    from ga_reading.metrics import FILE_COL

    first_area = area_mm2_from_mask_px(px, 52.2)
    second_area = area_mm2_from_mask_px(px, 52.2) * 1.8
    fields = [FILE_COL, "lesion present", "Quality of analysis", *GA_PRIMARY_METRICS]
    write_csv(reading / "grader.csv", fields, [{
        FILE_COL: name,
        "lesion present": "1",
        "Quality of analysis": "OK",
        "area_mm2": f"{first_area:.10g}",
        "area_mm_sqrt": f"{area_mm_sqrt(first_area):.10g}",
        "lesion_count": "1",
        "gawain": "40",
        "dist_um_nearest": "100",
        "gap_um": "80",
        "intact_1mm_mm2": "1.0",
    }])
    second_dir = reading / "second_reader_output_20260101_000000"
    second_dir.mkdir()
    write_csv(second_dir / "grader.csv", fields, [{
        FILE_COL: name,
        "lesion present": "1",
        "Quality of analysis": "OK",
        "area_mm2": f"{second_area:.10g}",
        "area_mm_sqrt": f"{area_mm_sqrt(second_area):.10g}",
        "lesion_count": "1",
        "gawain": "40",
        "dist_um_nearest": "100",
        "gap_um": "80",
        "intact_1mm_mm2": "1.0",
    }])
    # 中央値を取るための別の時点（推定はクラスタ側）。
    other = "2026-10-05.png"
    _write_mask(masks / other, 18)
    cv2.imwrite(str(images / other), np.zeros((64, 64), dtype=np.uint8))
    other_area = area_mm2_from_mask_px(px, 55.7)
    first_fields, first_rows = read_csv(reading / "grader.csv")
    first_rows.append({
        FILE_COL: other,
        "lesion present": "1",
        "Quality of analysis": "OK",
        "area_mm2": f"{other_area:.10g}",
        "area_mm_sqrt": f"{area_mm_sqrt(other_area):.10g}",
        "lesion_count": "1",
        "gawain": "40",
        "dist_um_nearest": "100",
        "gap_um": "80",
        "intact_1mm_mm2": "1.0",
    })
    write_csv(reading / "grader.csv", first_fields, first_rows)
    second_fields, second_rows = read_csv(second_dir / "grader.csv")
    second_rows.append({
        FILE_COL: other,
        "lesion present": "1",
        "Quality of analysis": "OK",
        "area_mm2": f"{other_area * 1.01:.10g}",
        "area_mm_sqrt": f"{area_mm_sqrt(other_area * 1.01):.10g}",
        "lesion_count": "1",
        "gawain": "40",
        "dist_um_nearest": "100",
        "gap_um": "80",
        "intact_1mm_mm2": "1.0",
    })
    write_csv(second_dir / "grader.csv", second_fields, second_rows)

    final_dir = reading / "final_reader_output_20260102_000000"
    final_dir.mkdir()
    write_csv(final_dir / "grader.csv", fields, [{
        FILE_COL: name,
        "lesion present": "1",
        "Quality of analysis": "OK",
        "area_mm2": f"{first_area * 1.1:.10g}",
        "area_mm_sqrt": f"{area_mm_sqrt(first_area * 1.1):.10g}",
        "lesion_count": "1",
        "gawain": "40",
        "dist_um_nearest": "100",
        "gap_um": "80",
        "intact_1mm_mm2": "1.0",
    }])

    out = tmp_path / "fixed"
    result = recompute_reading_dir(reading, out_dir=out, rerun_adoption=True)
    triad = result["adoption"]["triad"]
    _fields, triad_rows = read_csv(Path(triad["triad_csv"]))
    by_file = {row["File"]: row for row in triad_rows}
    # 第1・第2は補正後も 20% 超なので NA → 3者中央値。
    assert "area_mm2" in Path(result["adoption"]["merge"]["summary_md"]).read_text(encoding="utf-8")
    first_fixed = area_mm2_from_mask_px(px, result["eye_scale_um_per_px"])
    second_fixed = second_area * (result["eye_scale_um_per_px"] / 52.2) ** 2
    final_fixed = (first_area * 1.1) * (result["eye_scale_um_per_px"] / 52.2) ** 2
    expected = float(np.median([first_fixed, second_fixed, final_fixed]))
    assert float(by_file[name]["area_mm2"]) == pytest.approx(round(expected, 3), abs=0.02)


def test_native_frame_formula_and_priority():
    from engine.eye_scale import (
        SOURCE_DICOM,
        SOURCE_EYE_MEDIAN,
        SOURCE_FIXED,
        resolve_measurement_scale,
        scale_from_native_frame,
    )

    assert scale_from_native_frame(4000) == pytest.approx(10.4)
    assert scale_from_native_frame(2000) == pytest.approx(20.8)
    assert scale_from_native_frame(4000, native_um_per_px=10.4, native_width_px=4000) == 10.4
    explicit = resolve_measurement_scale(
        explicit_um_px=10.4,
        dicom_um_per_px=12.0,
        use_native_frame=True,
        image_width_px=2000,
        eye_estimates=[55.0, 56.0],
    )
    assert explicit is not None
    assert explicit.source == SOURCE_FIXED
    assert explicit.um_per_px == pytest.approx(10.4)
    dicom = resolve_measurement_scale(
        dicom_um_per_px=10.4,
        dicom_columns_px=4000,
        image_width_px=2000,
        eye_estimates=[55.0],
    )
    assert dicom is not None
    assert dicom.source == SOURCE_DICOM
    assert dicom.um_per_px == pytest.approx(20.8)
    median = resolve_measurement_scale(eye_estimates=[55.0, 56.0, 57.0])
    assert median is not None
    assert median.source == SOURCE_EYE_MEDIAN
    assert median.um_per_px == pytest.approx(56.0)


def test_synthetic_dicom_pixel_spacing(tmp_path: Path):
    from engine.dicom_scale import read_dicom_pixel_spacing, write_synthetic_dicom

    path = write_synthetic_dicom(tmp_path / "optos.dcm", spacing_mm=0.0104, columns=4000)
    um_px, columns = read_dicom_pixel_spacing(path)
    assert um_px == pytest.approx(10.4)
    assert columns == 4000
    other = write_synthetic_dicom(tmp_path / "wide.dcm", spacing_mm=0.0125, columns=3200)
    um_px, columns = read_dicom_pixel_spacing(other)
    assert um_px == pytest.approx(12.5)
    assert columns == 3200


def test_session_fixed_scale_beats_the_eye_median():
    session = OptosSession()
    session.case_id = "OD"
    session.files = [Path("a.png"), Path("b.png")]
    session.gray = np.full((80, 80), 180, dtype=np.uint8)
    session.fovea_user_placed = True
    session.fovea_xy = (30.0, 40.0)
    session.disc_xy = (60.0, 40.0)
    session.scale_um_px = 57.9
    session.mask = np.zeros((80, 80), dtype=np.uint8)
    session.mask[20:30, 20:30] = 255
    session.snaps["OD/a.png"] = VisitSnap(
        filename="a.png",
        path="a.png",
        date_iso="2022-01-01",
        date_source="filename",
        gray=session.gray,
        disc_xy=(60.0, 40.0),
        fovea_xy=(30.0, 40.0),
        scale_um_px=55.6,
        mask=session.mask,
        confirmed=True,
        fovea_user_placed=True,
    )
    session.set_fixed_scale("10.4")
    used, source = session.resolve_used_scale(57.9)
    assert used == pytest.approx(10.4)
    assert source == "fixed"
    rows = session.visit_rows()
    assert {row["scale_source"] for row in rows} == {"fixed"}
    assert all(row["scale_um_per_px"] == pytest.approx(10.4) for row in rows)


def test_session_native_frame_uses_image_width():
    session = OptosSession()
    session.gray = np.full((100, 2000), 180, dtype=np.uint8)
    session.set_native_frame_scale(True)
    used, source = session.resolve_used_scale(55.0)
    assert source == "fixed"
    assert used == pytest.approx(20.8)


def test_session_dicom_scale_uses_columns_and_width(tmp_path: Path):
    from engine.dicom_scale import write_synthetic_dicom

    path = write_synthetic_dicom(tmp_path / "ref.dcm", spacing_mm=0.0104, columns=4000)
    session = OptosSession()
    session.gray = np.full((50, 4000), 180, dtype=np.uint8)
    note = session.load_dicom_scale(path)
    assert "10.4" in note
    used, source = session.resolve_used_scale(55.0)
    assert source == "dicom"
    assert used == pytest.approx(10.4)
    session.gray = np.full((50, 2000), 180, dtype=np.uint8)
    used, source = session.resolve_used_scale(55.0)
    assert source == "dicom"
    assert used == pytest.approx(20.8)


def test_recompute_explicit_and_dicom_override_median(tmp_path: Path):
    from engine.dicom_scale import write_synthetic_dicom
    from ga_reading.merge import write_csv
    from ga_reading.metrics import FILE_COL

    reading = tmp_path / "OD_reading"
    masks = reading / "export" / "masks"
    images = reading / "export" / "images"
    masks.mkdir(parents=True)
    images.mkdir(parents=True)
    name = "2026-01-01.png"
    _write_mask(masks / name, 10)
    cv2.imwrite(str(images / name), np.zeros((32, 2000), dtype=np.uint8))
    px = 100
    area = area_mm2_from_mask_px(px, 55.7)
    fields = [FILE_COL, "lesion present", "Quality of analysis", *GA_PRIMARY_METRICS]
    write_csv(reading / "grader.csv", fields, [{
        FILE_COL: name,
        "lesion present": "1",
        "Quality of analysis": "OK",
        "area_mm2": f"{area:.10g}",
        "area_mm_sqrt": f"{area_mm_sqrt(area):.10g}",
        "lesion_count": "1",
        "gawain": "40",
        "dist_um_nearest": "100",
        "gap_um": "80",
        "intact_1mm_mm2": "1.0",
    }])
    explicit = recompute_reading_dir(reading, out_dir=tmp_path / "explicit", scale_um_px=10.4)
    assert explicit["scale_source"] == "fixed"
    assert explicit["eye_scale_um_per_px"] == pytest.approx(10.4)
    _fields, rows = read_csv(Path(explicit["first_csv"]))
    assert rows[0]["scale_source"] == "fixed"
    assert float(rows[0]["area_mm2"]) == pytest.approx(round(area_mm2_from_mask_px(px, 10.4), 3))

    native = recompute_reading_dir(reading, out_dir=tmp_path / "native", from_image_width=True)
    assert native["scale_source"] == "fixed"
    assert native["eye_scale_um_per_px"] == pytest.approx(20.8)

    dcm = write_synthetic_dicom(tmp_path / "ref.dcm", spacing_mm=0.0104, columns=4000)
    dicom = recompute_reading_dir(reading, out_dir=tmp_path / "dicom", dicom=dcm)
    assert dicom["scale_source"] == "dicom"
    assert dicom["eye_scale_um_per_px"] == pytest.approx(20.8)
    _fields, rows = read_csv(Path(dicom["first_csv"]))
    assert rows[0]["scale_source"] == "dicom"
