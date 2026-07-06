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


def list_images(folder: Path) -> list[Path]:
    """フォルダ内の対応画像形式を名前順に列挙する（大文字拡張子も可）。"""
    folder = Path(folder)
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
    )
