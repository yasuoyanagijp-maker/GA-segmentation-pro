"""io_utils.py – 画像入出力ユーティリティ。"""

from pathlib import Path

import cv2
import numpy as np

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

SUPPORTED_SUFFIXES = frozenset({
    '.tif', '.tiff', '.png', '.jpg', '.jpeg', '.gif',
})
SUPPORTED_EXTENSIONS = tuple(f'*{s}' for s in sorted(SUPPORTED_SUFFIXES))


def load_gray(path: Path) -> np.ndarray:
    """任意の画像形式をuint8グレースケールとして読み込む。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f'未対応の形式: {path.suffix}（対応: {", ".join(sorted(SUPPORTED_SUFFIXES))}）'
        )

    if HAS_TIFFFILE and suffix in ('.tif', '.tiff'):
        img = tifffile.imread(str(path))
    else:
        from PIL import Image
        with Image.open(path) as pil_img:
            # GIF アニメーションは先頭フレームを使用
            pil_img.seek(0)
            img = np.array(pil_img.convert('L'))

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] >= 3 else img[:, :, 0]

    img = img.astype(np.float32)
    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    return img


def classify_folder_entries(folder: Path) -> tuple[list[Path], list[tuple[str, str]]]:
    """フォルダ直下のファイルを、読影対象と除外に分ける。

    対応形式の画像は日付が同じでも、間隔が短くても、すべて対象にする。
    除外するのは画像ではないファイルだけで、理由を一緒に返す。
    隠しファイルとサブフォルダは対象の候補にしない。
    """
    folder = Path(folder)
    images: list[Path] = []
    excluded: list[tuple[str, str]] = []
    for path in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
        if path.name.startswith(".") or path.is_dir():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            kind = path.suffix or "拡張子なし"
            excluded.append((path.name, f"未対応の形式（{kind}）"))
            continue
        images.append(path)
    return images, excluded


def list_images(folder: Path) -> list[Path]:
    """フォルダ内の対応画像形式を名前順に列挙する（大文字拡張子も可）。"""
    images, _excluded = classify_folder_entries(folder)
    return images
