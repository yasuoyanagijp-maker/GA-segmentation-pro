import numpy as np

from engine.export import (
    build_batch_csv_bytes,
    generate_pdf_report_bytes,
    suggested_csv_filename,
)


def test_build_batch_csv_bytes():
    results = [
        {
            "case_id": "patient1",
            "date": "2024-01-01",
            "exam_datetime": "2024-01-01T10:00:00",
            "filename": "a.jpg",
            "stats": {
                "area_mm2": 1.2,
                "area_mm_sqrt": 1.1,
                "gawain": 80.0,
                "dist_um": 500.0,
                "lesion_count": 2,
            },
        },
    ]
    data = build_batch_csv_bytes(results)
    text = data.decode("utf-8-sig")
    assert "case_id" in text
    assert "exam_datetime" in text
    assert "patient1" in text
    assert "2024-01-01T10:00:00" in text
    assert "area_mm2" in text
    assert "a.jpg" in text
    assert "1.2" in text


def test_build_batch_csv_bytes_with_roi_qc_log():
    results = [
        {
            "date": "2024-01-01",
            "filename": "a.jpg",
            "stats": {"area_mm2": 1.2},
        },
    ]
    roi_qc = [
        {
            "index": 0,
            "date": "2024-01-01",
            "filename": "a.jpg",
            "raw_area": 1000,
            "corrected_area": 800,
            "dice_with_prev_raw": None,
            "dice_with_prev_corrected": None,
            "refine_method": "grabcut",
            "area_anomaly": True,
            "shape_anomaly": False,
            "needs_review": True,
            "reason": "area_anomaly",
        },
    ]
    text = build_batch_csv_bytes(results, roi_qc=roi_qc).decode("utf-8-sig")
    assert "ROI QC correction log" in text
    assert "grabcut" in text
    assert "area_anomaly" in text
    # QC ログなしなら従来どおりセクションは付かない
    plain = build_batch_csv_bytes(results).decode("utf-8-sig")
    assert "ROI QC correction log" not in plain


def test_generate_pdf_report_bytes():
    overlay = np.zeros((40, 40, 3), dtype=np.uint8)
    pdf_bytes = generate_pdf_report_bytes(
        filename="test.jpg",
        date="2024-03-15",
        stats={
            "area_mm2": 1.0,
            "area_mm_sqrt": 1.0,
            "gawain": 75.0,
            "dist_um": 400.0,
            "lesion_count": 1,
        },
        overlay_rgb=overlay,
        roi_qc_note="corrected (grabcut): 1000 -> 800 px",
    )
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 100


def test_generate_pdf_report_bytes_japanese_text():
    """Helvetica では「未」等が落ちるため、Unicode フォントで描画できること。"""
    from engine.export import _resolve_unicode_font_path

    if _resolve_unicode_font_path() is None:
        import pytest

        pytest.skip("No Unicode PDF font available on this host")

    overlay = np.zeros((40, 40, 3), dtype=np.uint8)
    pdf_bytes = generate_pdf_report_bytes(
        filename="test.jpg",
        date="2024-03-15",
        stats={
            "area_mm2": 1.0,
            "area_mm_sqrt": 1.0,
            "gawain": 75.0,
            "dist_um": 400.0,
            "lesion_count": 1,
        },
        overlay_rgb=overlay,
        case_id="未分類",
        calibration_source="disc_1800um",
        scale_um_px=21.43,
    )
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 100


def test_generate_pdf_report_bytes_without_unicode_font(monkeypatch):
    """日本語フォントが無いときも、未分類などの文字で出力が落ちない。"""
    monkeypatch.setattr("engine.export._resolve_unicode_font_path", lambda: None)
    overlay = np.zeros((40, 40, 3), dtype=np.uint8)
    pdf_bytes = generate_pdf_report_bytes(
        filename="test.jpg",
        date="2024-03-15",
        stats={
            "area_mm2": 1.0,
            "area_mm_sqrt": 1.0,
            "gawain": 75.0,
            "dist_um": 400.0,
            "lesion_count": 1,
        },
        overlay_rgb=overlay,
        case_id="未分類",
        calibration_source="disc_1800um",
        scale_um_px=21.43,
    )
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 100


def test_suggested_csv_filename():
    name = suggested_csv_filename()
    assert name.startswith("GA_batch_")
    assert name.endswith(".csv")
