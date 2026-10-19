"""第2は export の親から画像だけを取る。最終読影の列は summary の RECHECK だけ。"""

from __future__ import annotations

import re
from pathlib import Path

from ga_reading.merge import file_stem
from ga_reading.metrics import GA_PRIMARY_METRICS

_IMAGE = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_WORKFLOW_PREFIXES = ("second_reader_output", "integrated_output", "final_reader_output")


def latest_workflow_csv(scan_root: Path, prefix: str) -> Path | None:
    """second_reader_output_* や final_reader_output_* のうち、新しい grader.csv。"""
    root = Path(scan_root)
    if not root.is_dir():
        return None
    folders = sorted(
        (
            child for child in root.iterdir()
            if child.is_dir() and child.name.startswith(f"{prefix}_")
        ),
        key=lambda path: path.name,
    )
    for folder in reversed(folders):
        candidate = folder / "grader.csv"
        if candidate.is_file():
            return candidate
    return None


def find_first_grader_csv(scan_root: Path) -> Path | None:
    """第1の grader.csv。第2・統合・最終の出力フォルダは見ない。"""
    root = Path(scan_root)
    direct = root / "grader.csv"
    if direct.is_file():
        return direct
    found: list[Path] = []
    if not root.is_dir():
        return None
    for child in root.iterdir():
        if not child.is_dir() or child.name.startswith(_WORKFLOW_PREFIXES):
            continue
        candidate = child / "grader.csv"
        if candidate.is_file():
            found.append(candidate)
    return found[0] if len(found) == 1 else None


def recheck_images(summary_md: Path) -> list[Path]:
    """*_summary.md の RECHECK 画像だけを、export/images から返す。"""
    summary_md = Path(summary_md)
    if not summary_md.is_file() or not summary_md.name.endswith("_summary.md"):
        raise ValueError("RECHECK の *_summary.md を選んでください。")
    stems = set(recheck_stems(summary_md))
    if not stems:
        raise ValueError("この summary に RECHECK の画像はありません。")
    images: list[Path] | None = None
    anchor = summary_md.parent
    for candidate in (anchor, anchor.parent, anchor.parent.parent):
        try:
            images = second_reader_images(candidate)
            break
        except ValueError:
            continue
    if images is None:
        raise ValueError("export/images が見つかりません。第1の出力フォルダの近くに summary があるか確認してください。")
    picked = [path for path in images if file_stem(path.name) in stems]
    if not picked:
        raise ValueError("RECHECK の画像が export/images にありません。")
    return picked


def second_reader_images(selected: Path) -> list[Path]:
    """第1の出力（export の親、または export 自身）から画像を返す。マスクは返さない。"""
    selected = Path(selected).expanduser().resolve()
    if selected.is_file():
        selected = selected.parent
    export = selected / "export" if (selected / "export").is_dir() else selected
    images_dir = export / "images"
    if not images_dir.is_dir():
        raise ValueError(f"export/images が見つかりません: {selected}")
    images = [
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE and "masks" not in path.parts
    ]
    if not images:
        raise ValueError(f"第2が読む画像がありません: {images_dir}")
    return sorted(images, key=lambda path: path.name.lower())


def recheck_targets(summary_md: Path) -> list[tuple[str, list[str]]]:
    """*_summary.md の RECHECK 画像と、NA になった主要指標。"""
    text = Path(summary_md).read_text(encoding="utf-8")
    in_section = False
    found: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            in_section = line.startswith("## RECHECK")
            continue
        if not in_section or not line.startswith("- "):
            continue
        body = line[2:].strip()
        if ":" not in body or body.startswith("(") or body.startswith("主要指標"):
            continue
        name, metrics = body.split(":", 1)
        name = name.strip()
        if name in {"(なし)", "なし"}:
            continue
        columns = [item.strip() for item in metrics.split(",") if item.strip()]
        unknown = [item for item in columns if item not in GA_PRIMARY_METRICS]
        if unknown:
            raise ValueError("RECHECK に未知の指標があります: " + ", ".join(unknown))
        if columns:
            found.append((name, columns))
    return found


def recheck_stems(summary_md: Path) -> list[str]:
    seen: list[str] = []
    for name, _columns in recheck_targets(summary_md):
        key = file_stem(name)
        if key not in seen:
            seen.append(key)
    return seen


def matches_file_stem(image: Path, file_cell: str) -> bool:
    return file_stem(image.name) == file_stem(file_cell) and bool(re.search(r"\S", file_cell))
