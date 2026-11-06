"""DICOM から PixelSpacing と Columns だけを読む。画素データは見ない。"""

from __future__ import annotations

from pathlib import Path

_COLUMNS_TAG = b"\x28\x00\x11\x00"
_SPACING_TAG = b"\x28\x00\x30\x00"
_LONG_VR = {b"OB", b"OD", b"OF", b"OL", b"OW", b"SQ", b"UC", b"UN", b"UR", b"UT"}


def pixel_spacing_mm_to_um(spacing_mm: float) -> float:
    if spacing_mm <= 0:
        raise ValueError("PixelSpacing は正の値が必要です")
    return float(spacing_mm) * 1000.0


def read_dicom_pixel_spacing(path: Path) -> tuple[float, int]:
    """(µm/px, Columns)。PixelSpacing は mm。行・列が違えば列側を使う。"""
    data = Path(path).read_bytes()
    if len(data) < 132:
        raise ValueError(f"DICOM が短すぎます: {path}")
    columns = _read_columns(data)
    spacing_mm = _read_spacing_mm(data)
    if columns is None or spacing_mm is None:
        raise ValueError(f"PixelSpacing または Columns がありません: {path}")
    return pixel_spacing_mm_to_um(spacing_mm), columns


def write_synthetic_dicom(
    path: Path,
    *,
    spacing_mm: float = 0.0104,
    columns: int = 4000,
) -> Path:
    """テスト用の最小 DICOM（Explicit VR LE、画素なし）。"""
    path = Path(path)
    ts = b"1.2.840.10008.1.2.1"
    meta_body = _explicit(0x0002, 0x0010, b"UI", ts)
    meta = _explicit(0x0002, 0x0000, b"UL", len(meta_body).to_bytes(4, "little")) + meta_body
    dataset = _explicit(0x0028, 0x0011, b"US", int(columns).to_bytes(2, "little"))
    text = f"{spacing_mm}\\{spacing_mm}".encode("ascii")
    dataset += _explicit(0x0028, 0x0030, b"DS", text)
    path.write_bytes(b"\x00" * 128 + b"DICM" + meta + dataset)
    return path


def _explicit(group: int, element: int, vr: bytes, value: bytes) -> bytes:
    tag = group.to_bytes(2, "little") + element.to_bytes(2, "little")
    if vr in _LONG_VR:
        return tag + vr + b"\x00\x00" + len(value).to_bytes(4, "little") + value
    if len(value) % 2 == 1:
        value += b" "
    return tag + vr + len(value).to_bytes(2, "little") + value


def _read_columns(data: bytes) -> int | None:
    raw = _value_after_tag(data, _COLUMNS_TAG, prefer_vr=b"US")
    if raw is None or len(raw) < 2:
        return None
    return int.from_bytes(raw[:2], "little")


def _read_spacing_mm(data: bytes) -> float | None:
    raw = _value_after_tag(data, _SPACING_TAG, prefer_vr=b"DS")
    if raw is None:
        return None
    text = raw.decode("ascii", errors="ignore").strip().strip("\x00")
    parts = [part for part in text.replace(",", "\\").split("\\") if part.strip()]
    if not parts:
        return None
    # 行\\列。面積換算は列方向（幅）を使う。
    chosen = parts[1] if len(parts) > 1 else parts[0]
    value = float(chosen)
    return value if value > 0 else None


def _value_after_tag(data: bytes, tag: bytes, *, prefer_vr: bytes) -> bytes | None:
    start = 0
    while True:
        idx = data.find(tag, start)
        if idx < 0:
            return None
        after = data[idx + 4:]
        explicit = _explicit_value(after, prefer_vr)
        if explicit is not None:
            return explicit
        if len(after) >= 4:
            length = int.from_bytes(after[:4], "little")
            if 0 < length <= 64 and 4 + length <= len(after):
                return after[4:4 + length]
        start = idx + 4


def _explicit_value(after: bytes, prefer_vr: bytes) -> bytes | None:
    if len(after) < 4:
        return None
    vr = after[:2]
    if vr != prefer_vr:
        return None
    if vr in _LONG_VR:
        if len(after) < 8:
            return None
        length = int.from_bytes(after[4:8], "little")
        return after[8:8 + length] if 0 <= length <= 64 else None
    length = int.from_bytes(after[2:4], "little")
    if not (0 <= length <= 64) or 4 + length > len(after):
        return None
    return after[4:4 + length]
