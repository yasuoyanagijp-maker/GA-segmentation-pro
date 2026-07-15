"""
preprocessing.py
-----------------
FAF画像の輝度正規化・コントラスト強調・ノイズ低減。
登録(registration)と境界改善(refine)の両方が、ここで作った
「解析用画像」を入力として使う。
"""

from __future__ import annotations

import numpy as np
import cv2


def to_uint8_gray(image: np.ndarray) -> np.ndarray:
    """任意のdtype/チャンネル数の画像を 8bit グレースケールへ変換する。"""
    img = image
    if img.ndim == 3:
        # 3ch (BGR/RGBいずれでも輝度としては近似的に妥当)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[..., 0]
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max() + 1e-6)
    img = np.clip((img - lo) / (hi - lo), 0, 1)
    return (img * 255).astype(np.uint8)


def apply_clahe(gray_u8: np.ndarray, clip_limit: float = 2.5, tile_grid_size: int = 8) -> np.ndarray:
    """
    CLAHE(局所適応ヒストグラム均等化)でGA境界付近のコントラストを強調する。
    FAFは撮影ごとに背景輝度差が大きく、単純な全体正規化だけでは
    境界のコントラストが不足しやすいため、登録前ではなくセグメンテーション
    改善(refine)の直前に適用するのが基本。
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(gray_u8)


def denoise(gray_u8: np.ndarray, h: float = 6.0) -> np.ndarray:
    """Non-local means によるノイズ低減。境界を過度に鈍らせない h を推奨。"""
    return cv2.fastNlMeansDenoising(gray_u8, None, h=h, templateWindowSize=7, searchWindowSize=21)


def preprocess(image: np.ndarray, clip_limit: float = 2.5, denoise_h: float = 6.0) -> dict:
    """
    登録用画像(denoiseのみ、コントラスト操作は最小限)と
    解析用画像(CLAHE込み)を分けて返す。SOP 6章に対応。
    """
    gray = to_uint8_gray(image)
    reg_img = denoise(gray, h=denoise_h)
    analysis_img = apply_clahe(reg_img, clip_limit=clip_limit)
    return {"gray": gray, "registration_image": reg_img, "analysis_image": analysis_img}
