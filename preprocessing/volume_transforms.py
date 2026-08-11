"""
preprocessing/volume_transforms.py

Utilities for preparing a full MRI series (stack of 2D slices) as a
single fixed-depth 3D volume, for use with true 3D CNN architectures
(Phase 21), as opposed to the per-slice 2D approach used in Phases
9-20.
"""

import logging

import numpy as np
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


def resample_volume_depth(volume: np.ndarray, target_depth: int = 32) -> np.ndarray:
    """Resample a 3D volume to a fixed number of slices along the depth axis.

    Uses interpolation to stretch or compress the slice dimension to
    exactly target_depth, while leaving height and width untouched.
    This lets series with wildly different slice counts (per Step 2:
    20 to a few hundred, median ~30) all be fed to a 3D CNN that
    requires a fixed input depth.

    Args:
        volume: A 3D NumPy array of shape (D, H, W), where D is the
            original number of slices.
        target_depth: The desired number of slices after resampling.

    Returns:
        A 3D NumPy array of shape (target_depth, H, W), same dtype
        as the input.

    Raises:
        ValueError: If the input volume is not 3D, or target_depth
            is not positive.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume (D, H, W), got shape {volume.shape}.")
    if target_depth <= 0:
        raise ValueError(f"target_depth must be positive, got {target_depth}.")

    original_depth = volume.shape[0]
    zoom_factor = target_depth / original_depth

    resampled = zoom(volume, zoom=(zoom_factor, 1.0, 1.0), order=1)

    # Interpolation can occasionally produce off-by-one depth due to
    # floating point rounding -- trim or pad to guarantee exact match.
    if resampled.shape[0] != target_depth:
        resampled = _fix_depth_mismatch(resampled, target_depth)

    logger.debug(
        "Resampled volume depth: %d -> %d slices",
        original_depth, resampled.shape[0],
    )

    return resampled


def _fix_depth_mismatch(volume: np.ndarray, target_depth: int) -> np.ndarray:
    """Trim or pad a volume's depth to exactly match target_depth.

    Args:
        volume: A 3D array whose depth is off by a slice or two from
            target_depth, due to floating-point rounding in zoom().
        target_depth: The exact desired depth.

    Returns:
        A 3D array with shape[0] == target_depth exactly.
    """
    current_depth = volume.shape[0]

    if current_depth > target_depth:
        return volume[:target_depth]

    pad_amount = target_depth - current_depth
    last_slice = volume[-1:]
    padding = np.repeat(last_slice, pad_amount, axis=0)
    return np.concatenate([volume, padding], axis=0)


def build_3d_volume_from_slices(
    slice_list: list[np.ndarray],
    target_depth: int = 32,
) -> np.ndarray:
    """Stack a list of preprocessed 2D slices into a fixed-depth 3D volume.

    Args:
        slice_list: A list of 2D arrays (each shape (H, W)), typically
            already normalized via preprocessing.transforms functions,
            in slice order.
        target_depth: The desired number of slices in the output volume.

    Returns:
        A 3D float32 array of shape (target_depth, H, W).

    Raises:
        ValueError: If slice_list is empty.
    """
    if not slice_list:
        raise ValueError("Cannot build a volume from an empty slice list.")

    stacked = np.stack(slice_list, axis=0).astype(np.float32)
    resampled = resample_volume_depth(stacked, target_depth)

    return resampled