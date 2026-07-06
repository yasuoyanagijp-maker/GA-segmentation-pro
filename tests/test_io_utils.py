import numpy as np
from pathlib import Path

from engine.io_utils import SUPPORTED_SUFFIXES, list_images, load_gray


def test_supported_suffixes_include_gif():
    assert '.gif' in SUPPORTED_SUFFIXES
    assert '.tif' in SUPPORTED_SUFFIXES
    assert '.jpeg' in SUPPORTED_SUFFIXES


def test_load_user_gif_if_present():
    folder = Path('/Users/yy/Desktop/名称未設定フォルダ')
    if not folder.is_dir():
        return
    files = list_images(folder)
    if not files:
        return
    img = load_gray(files[0])
    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert img.size > 0
