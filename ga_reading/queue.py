"""第2は第1の export/images だけを取る。第2・統合・最終は第1読影フォルダの中に置く。"""

from __future__ import annotations

import json
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


def _is_workflow_dir(path: Path) -> bool:
    return path.name.startswith(_WORKFLOW_PREFIXES)


def _bundle_root(path: Path) -> Path | None:
    """grader の出力そのものなら、そのフォルダ。export や images を指していれば一段上。"""
    path = Path(path)
    if path.name == "images" and path.parent.name == "export":
        path = path.parent.parent
    elif path.name == "export" and (path / "images").is_dir():
        path = path.parent
    if _is_workflow_dir(path):
        return None
    if (path / "export" / "images").is_dir():
        return path
    return None


def _child_bundles(selected: Path) -> list[Path]:
    if not selected.is_dir():
        return []
    found: list[Path] = []
    for child in sorted(selected.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir() or child.name.startswith(".") or _is_workflow_dir(child):
            continue
        root = _bundle_root(child)
        if root is not None:
            found.append(root)
    return found


def first_reader_dir(selected: Path) -> Path:
    """第1読影の出力フォルダ。画像はここの export/images だけ。

    選んだフォルダに grader.csv があれば、そこを第1とする。
    csv が無く、直下の第1フォルダが1つだけならそれを使う。上の階層は見ない。
    兄弟に複数あるときは、別の眼の画像を開かないよう止める。
    """
    selected = Path(selected).expanduser().resolve()
    if selected.is_file():
        if selected.name.endswith("_summary.md"):
            return first_reader_dir_near_summary(selected)
        selected = selected.parent
    own = _bundle_root(selected)
    children = _child_bundles(selected)
    if own is not None and (selected / "grader.csv").is_file():
        return own
    if len(children) == 1:
        return children[0]
    if own is not None and not children:
        return own
    if len(children) > 1:
        names = ", ".join(path.name for path in children)
        raise ValueError(
            "第1読影のフォルダが複数あります。読みたい眼のフォルダそのものを選んでください: " + names
        )
    if own is not None:
        return own
    raise ValueError(f"export/images が見つかりません: {selected}")


def workflow_parent(first_reader: Path) -> Path:
    """第2・統合・最終を置く場所。第1読影フォルダそのもの（例: OD_reading）。"""
    return Path(first_reader).resolve()


def workflow_parent_from_summary(summary_md: Path) -> Path:
    """summary から、第1読影フォルダ（export と grader.csv がある場所）を返す。

    見つからないとき、症例フォルダへは戻さない。
    """
    summary_md = Path(summary_md).resolve()
    try:
        return first_reader_dir_near_summary(summary_md)
    except ValueError:
        bundles = _child_bundles(summary_md.parent.parent)
        if len(bundles) == 1:
            return bundles[0]
        raise


def first_reader_dir_near_summary(summary_md: Path) -> Path:
    """最終読影が開く画像の元。source json の第1 CSV を優先し、上の階層は探さない。"""
    summary_md = Path(summary_md).resolve()
    source = summary_md.with_name(summary_md.name.replace("_summary.md", "_source_csvs.json"))
    if source.is_file():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        first_csv = Path(str(payload.get("first_csv", "")))
        if first_csv.is_file():
            root = _bundle_root(first_csv.parent)
            if root is not None:
                return root
    container = summary_md.parent.parent
    direct = _bundle_root(container)
    if direct is not None:
        return direct
    return first_reader_dir(container)


def recheck_images(summary_md: Path) -> list[Path]:
    """*_summary.md の RECHECK 画像だけを、第1の export/images から返す。"""
    summary_md = Path(summary_md)
    if not summary_md.is_file() or not summary_md.name.endswith("_summary.md"):
        raise ValueError("RECHECK の *_summary.md を選んでください。")
    stems = set(recheck_stems(summary_md))
    if not stems:
        raise ValueError("この summary に RECHECK の画像はありません。")
    try:
        images = second_reader_images(first_reader_dir_near_summary(summary_md))
    except ValueError as exc:
        if "複数" in str(exc):
            raise
        raise ValueError(
            "export/images が見つかりません。第1の出力フォルダの近くに summary があるか確認してください。"
        ) from exc
    picked = [path for path in images if file_stem(path.name) in stems]
    if not picked:
        raise ValueError("RECHECK の画像が export/images にありません。")
    return picked


def second_reader_images(selected: Path) -> list[Path]:
    """第1の export/images 直下の画像だけを返す。マスクも、隣の出力フォルダも読まない。"""
    root = first_reader_dir(selected)
    images_dir = root / "export" / "images"
    if not images_dir.is_dir():
        raise ValueError(f"export/images が見つかりません: {selected}")
    images = [
        path for path in images_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in _IMAGE
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
