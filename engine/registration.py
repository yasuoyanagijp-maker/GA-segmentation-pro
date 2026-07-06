"""registration.py – 多重解像度アフィン位置合わせ（stack_reg 由来）。"""

from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def build_gaussian_pyramid(img: np.ndarray, min_size: int = 64) -> list[np.ndarray]:
    pyramid = [img]
    while True:
        h, w = pyramid[-1].shape[:2]
        if h <= min_size * 2 or w <= min_size * 2:
            break
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def calculate_affine_transformations(
    reference_stack: np.ndarray,
    progress_callback: Callable[[float, str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> list[np.ndarray]:
    """各スライスを先頭スライスへ合わせる 2x3 アフィン行列を返す。"""
    num_slices = reference_stack.shape[0]
    matrices = [np.eye(2, 3, dtype=np.float32)]

    target_slice = reference_stack[0].astype(np.float32)
    target_pyramid = build_gaussian_pyramid(target_slice, min_size=64)
    num_levels = len(target_pyramid)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)

    s = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    s_inv = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    total_alignments = max(num_slices - 1, 1)
    for i in range(1, num_slices):
        source_slice = reference_stack[i].astype(np.float32)
        source_pyramid = build_gaussian_pyramid(source_slice, min_size=64)
        effective_levels = min(num_levels, len(source_pyramid))
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        for level_idx, level_res in enumerate(reversed(range(effective_levels))):
            curr_target = target_pyramid[level_res]
            curr_source = source_pyramid[level_res]
            msg = f'Aligning slice {i + 1}/{num_slices} [level {level_res}]'
            if progress_callback:
                level_prog = ((i - 1) / total_alignments) + (
                    level_idx / (effective_levels * total_alignments)
                )
                progress_callback(level_prog, msg)

            try:
                if level_idx == 0:
                    res = cv2.matchTemplate(curr_source, curr_target, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val > 0.4:
                        warp_matrix[0, 2] = float(max_loc[0])
                        warp_matrix[1, 2] = float(max_loc[1])

                _, new_warp = cv2.findTransformECC(
                    curr_target, curr_source, warp_matrix,
                    cv2.MOTION_AFFINE, criteria,
                )
                warp_matrix = new_warp
            except cv2.error:
                err_msg = f'[WARN] slice {i} level {level_res} failed; keeping previous estimate.'
                logger.warning(err_msg)
                if log_callback:
                    log_callback(err_msg)

            if level_res > 0:
                m_3x3 = np.vstack([warp_matrix.astype(np.float64), [0.0, 0.0, 1.0]])
                warp_matrix = (s @ m_3x3 @ s_inv)[:2, :].astype(np.float32)

        matrices.append(warp_matrix)
        if progress_callback:
            progress_callback(i / total_alignments, f'Slice {i + 1}/{num_slices} aligned.')

    return matrices


def apply_transformations_to_stack(
    stack: np.ndarray,
    matrices: list[np.ndarray],
    progress_callback: Callable[[float, str], None] | None = None,
) -> np.ndarray:
    num_slices = stack.shape[0]
    h, w = stack.shape[1], stack.shape[2]
    transformed_stack = np.zeros_like(stack)

    for i in range(num_slices):
        if progress_callback:
            progress_callback(i / num_slices, f'Warping slice {i + 1}/{num_slices}...')
        transformed_stack[i] = cv2.warpAffine(
            stack[i], matrices[i], (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    return transformed_stack


def register_stack(stack: np.ndarray,
                   progress_callback: Callable[[float, str], None] | None = None,
                   log_callback: Callable[[str], None] | None = None
                   ) -> tuple[np.ndarray, list[np.ndarray]]:
    """スタック全体を先頭画像へ位置合わせする。"""
    matrices = calculate_affine_transformations(
        stack, progress_callback=progress_callback, log_callback=log_callback,
    )
    aligned = apply_transformations_to_stack(stack, matrices, progress_callback=progress_callback)
    return aligned, matrices
