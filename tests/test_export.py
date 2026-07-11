import numpy as np
from pathlib import Path

from engine.export import (
    build_batch_csv_bytes,
    gawain_figure_png_from_context,
    generate_pdf_report,
    generate_pdf_report_bytes,
    suggested_csv_filename,
)


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
            "lesions": [{"id": 1, "area_mm2": 1.0, "dist_um": 400.0}],
        },
        overlay_rgb=overlay,
        output_path=out,
        scale_um_px=10.0,
        calibration_source="width",
        field_width_mm=8.0,
        fovea_xy=(20.0, 20.0),
        foveal_correction_applied=True,
        foveal_correction_strength=0.5,
    )
    assert out.exists()
    assert out.stat().st_size > 100


def test_generate_pdf_report_with_gawain_page(tmp_path: Path):
    overlay = np.zeros((80, 80, 3), dtype=np.uint8)
    img = np.full((120, 120), 180, dtype=np.uint8)
    mask = np.zeros_like(img)
    mask[50:70, 50:70] = 255
    ctx = {
        "filename": "test.jpg",
        "image": img,
        "mask": mask,
        "fovea_xy": (60.0, 60.0),
        "scale_um_px": 10.0,
    }
    gawain_png = gawain_figure_png_from_context(ctx)
    assert gawain_png is not None
    assert len(gawain_png) > 1000

    pdf_bytes = generate_pdf_report_bytes(
        filename="test.jpg",
        date="2024-03-15",
        stats={
            "area_mm2": 0.4,
            "area_mm_sqrt": 0.63,
            "gawain": 12.0,
            "dist_um": 100.0,
            "lesion_count": 1,
            "lesions": [{"id": 1, "area_mm2": 0.4, "dist_um": 100.0}],
        },
        overlay_rgb=overlay,
        gawain_figure_png=gawain_png,
        scale_um_px=10.0,
        fovea_xy=(60.0, 60.0),
    )
    assert len(pdf_bytes) > len(gawain_png)
    out = tmp_path / "two_page.pdf"
    out.write_bytes(pdf_bytes)
    assert out.stat().st_size > 5000


def test_suggested_csv_filename():
    name = suggested_csv_filename()
    assert name.startswith("GA_batch_")
    assert name.endswith(".csv")
