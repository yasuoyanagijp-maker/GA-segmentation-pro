import numpy as np
from pathlib import Path

from engine.export import build_batch_csv_bytes, generate_pdf_report, suggested_csv_filename


def test_build_batch_csv_bytes():
    results = [
        {
            "date": "2024-01-01",
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
    assert "area_mm2" in text
    assert "a.jpg" in text
    assert "1.2" in text


def test_generate_pdf_report(tmp_path: Path):
    overlay = np.zeros((40, 40, 3), dtype=np.uint8)
    out = tmp_path / "report.pdf"
    generate_pdf_report(
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
        output_path=out,
    )
    assert out.exists()
    assert out.stat().st_size > 100


def test_suggested_csv_filename():
    name = suggested_csv_filename()
    assert name.startswith("GA_batch_")
    assert name.endswith(".csv")
