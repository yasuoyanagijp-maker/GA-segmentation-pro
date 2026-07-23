"""症例グループ（フォルダ構造・未分類）の管理。"""

from __future__ import annotations

from pathlib import Path

from engine.io_utils import list_images

UNCLASSIFIED_CASE_ID = "未分類"


def discover_cases(folder: Path) -> dict[str, list[Path]]:
    """入力フォルダから症例 → 画像リストを構築する。

    - サブフォルダに画像があれば、そのフォルダ名を case_id とする
    - ルート直下の画像は「未分類」にまとめる
    - サブフォルダもルート画像も無い場合は空 dict
    """
    folder = Path(folder)
    if not folder.is_dir():
        return {}

    cases: dict[str, list[Path]] = {}
    for sub in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        imgs = list_images(sub)
        if imgs:
            cases[sub.name] = imgs

    root_imgs = list_images(folder)
    if root_imgs:
        cases[UNCLASSIFIED_CASE_ID] = root_imgs

    return cases


def ordered_case_ids(cases: dict[str, list[Path]]) -> list[str]:
    """表示用の症例 ID 順（未分類は末尾）。"""
    ids = [c for c in cases if c != UNCLASSIFIED_CASE_ID]
    ids.sort(key=str.lower)
    if UNCLASSIFIED_CASE_ID in cases:
        ids.append(UNCLASSIFIED_CASE_ID)
    return ids


def ensure_unique_case_id(cases: dict[str, list[Path]], name: str) -> str:
    """空でなければ一意な case_id を返す（重複時は _2, _3…）。"""
    base = (name or "").strip() or "case"
    if base not in cases:
        return base
    n = 2
    while f"{base}_{n}" in cases:
        n += 1
    return f"{base}_{n}"


def move_file_to_case(
    cases: dict[str, list[Path]],
    path: Path,
    from_case: str,
    to_case: str,
) -> dict[str, list[Path]]:
    """画像を症例間で移動した新しい cases dict を返す（パス自体は変えない）。"""
    return move_files_to_case(cases, [path], from_case, to_case)


def move_files_to_case(
    cases: dict[str, list[Path]],
    paths: list[Path],
    from_case: str,
    to_case: str,
) -> dict[str, list[Path]]:
    """複数画像を症例間で一括移動した新しい cases dict を返す（パス自体は変えない）。"""
    if from_case == to_case or not paths:
        return cases
    path_set = {Path(p) for p in paths}
    out: dict[str, list[Path]] = {
        cid: list(files) for cid, files in cases.items()
    }
    src = out.get(from_case) or []
    moved = [p for p in src if p in path_set]
    if not moved:
        return cases
    out[from_case] = [p for p in src if p not in path_set]
    dest = list(out.get(to_case) or [])
    known = set(dest)
    for p in moved:
        if p not in known:
            dest.append(p)
            known.add(p)
    dest.sort(key=lambda p: p.name.lower())
    out[to_case] = dest
    # 空になった症例は残す（ユーザーが再投入できるよう）。未分類以外の空は削除可。
    if from_case != UNCLASSIFIED_CASE_ID and not out[from_case]:
        del out[from_case]
    return out


def create_case(
    cases: dict[str, list[Path]],
    name: str,
) -> tuple[dict[str, list[Path]], str]:
    """新規症例を追加。戻り値: (cases, case_id)。"""
    cid = ensure_unique_case_id(cases, name)
    out = {k: list(v) for k, v in cases.items()}
    out[cid] = []
    return out, cid
