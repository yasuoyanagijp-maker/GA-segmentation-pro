"""Widget ↔ image coordinate mapping for landmark taps."""

from flet_ui.main_view import compute_display_box, local_to_image_xy


def test_compute_display_box_preserves_aspect():
    box_w, box_h = compute_display_box(953, 829, 520)
    assert box_w == 520
    assert box_h == round(829 * 520 / 953)


def test_local_to_image_roundtrip_corners():
    img_w, img_h = 953, 829
    box_w, box_h = compute_display_box(img_w, img_h, 520)

    # Top-left
    x, y = local_to_image_xy(0, 0, img_w=img_w, img_h=img_h, box_w=box_w, box_h=box_h)
    assert x == 0.0 and y == 0.0

    # Bottom-right of widget maps near bottom-right of image
    x, y = local_to_image_xy(
        box_w - 1e-6, box_h - 1e-6,
        img_w=img_w, img_h=img_h, box_w=box_w, box_h=box_h,
    )
    assert abs(x - (img_w - 1)) < 1.5
    assert abs(y - (img_h - 1)) < 1.5


def test_local_to_image_uses_independent_xy_scales():
    """Y must use box_h/img_h — not the X scale (common letterbox bug)."""
    img_w, img_h = 1000, 500
    box_w, box_h = 200.0, 50.0  # anamorphic box (as after int rounding)
    # Mid widget → mid image
    x, y = local_to_image_xy(
        100, 25, img_w=img_w, img_h=img_h, box_w=box_w, box_h=box_h,
    )
    assert abs(x - 500) < 1e-6
    assert abs(y - 250) < 1e-6


def test_local_to_image_subtracts_letterbox_offset():
    img_w, img_h = 100, 100
    # Content painted at 1:1 into a padded widget: 50px left inset.
    content_w, content_h = 100.0, 100.0
    x, y = local_to_image_xy(
        50, 0,
        img_w=img_w, img_h=img_h,
        box_w=content_w, box_h=content_h,
        off_x=50.0, off_y=0.0,
    )
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6
