"""io_utils.py – 画像入出力ユーティリティ。"""

from pathlib import Path

import cv2
import numpy as np

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

SUPPORTED_EXTENSIONS = ('*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg')


def load_gray(path: Path) -> np.ndarray:
    """任意の画像形式をuint8グレースケールとして読み込む。"""
    path = Path(path)
    if HAS_TIFFFILE and path.suffix.lower() in ('.tif', '.tiff'):
        img = tifffile.imread(str(path))
    else:
        from PIL import Image
        img = np.array(Image.open(path).convert('L'))

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] >= 3 else img[:, :, 0]

    img = img.astype(np.float32)
    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    return img


def list_images(folder: Path) -> list[Path]:
    """フォルダ内の対応画像形式を名前順に列挙する。"""
    folder = Path(folder)
    return sorted(f for ext in SUPPORTED_EXTENSIONS for f in folder.glob(ext))
