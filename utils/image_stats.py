"""
utils/image_stats.py

Utility functions for inspecting raw medical image intensity
characteristics before any preprocessing decisions are made.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IntensityStats:
    """Summary statistics describing a raw image array's intensity distribution.

    Attributes:
        min_val: Minimum pixel intensity in the array.
        max_val: Maximum pixel intensity in the array.
        mean_val: Mean pixel intensity.
        std_val: Standard deviation of pixel intensity.
        p1: 1st percentile intensity (robust lower bound).
        p99: 99th percentile intensity (robust upper bound).
        dtype: Original NumPy dtype of the array.
        implied_bit_depth: Estimated bit depth based on max value.
    """

    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    p1: float
    p99: float
    dtype: str
    implied_bit_depth: int


def compute_intensity_stats(image: np.ndarray) -> IntensityStats:
    """Compute robust intensity statistics for a raw medical image array.

    Args:
        image: A 2D or 3D NumPy array of raw pixel/voxel intensities.

    Returns:
        An IntensityStats object summarizing the array's intensity profile.

    Raises:
        ValueError: If the input array is empty.
    """
    if image.size == 0:
        raise ValueError("Cannot compute intensity stats on an empty array.")

    min_val = float(np.min(image))
    max_val = float(np.max(image))
    mean_val = float(np.mean(image))
    std_val = float(np.std(image))
    p1 = float(np.percentile(image, 1))
    p99 = float(np.percentile(image, 99))

    implied_bit_depth = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 0

    stats = IntensityStats(
        min_val=min_val,
        max_val=max_val,
        mean_val=mean_val,
        std_val=std_val,
        p1=p1,
        p99=p99,
        dtype=str(image.dtype),
        implied_bit_depth=implied_bit_depth,
    )

    logger.info(
        "Intensity stats | min=%.2f max=%.2f mean=%.2f std=%.2f "
        "p1=%.2f p99=%.2f dtype=%s implied_bits=%d",
        stats.min_val, stats.max_val, stats.mean_val, stats.std_val,
        stats.p1, stats.p99, stats.dtype, stats.implied_bit_depth,
    )

    return stats


def robust_normalize(image: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Normalize an image to [0, 1] using percentile-based clipping.

    Args:
        image: A 2D or 3D NumPy array of raw pixel/voxel intensities.
        p_low: Lower percentile bound for clipping (default: 1.0).
        p_high: Upper percentile bound for clipping (default: 99.0).

    Returns:
        A float32 NumPy array with values rescaled to [0, 1].

    Raises:
        ValueError: If p_low >= p_high, or the image is empty.
    """
    if image.size == 0:
        raise ValueError("Cannot normalize an empty array.")
    if p_low >= p_high:
        raise ValueError(f"p_low ({p_low}) must be less than p_high ({p_high}).")

    lo = np.percentile(image, p_low)
    hi = np.percentile(image, p_high)

    if hi <= lo:
        logger.warning(
            "Degenerate intensity range (p%.1f=%.2f >= p%.1f=%.2f); returning zeros.",
            p_low, lo, p_high, hi,
        )
        return np.zeros_like(image, dtype=np.float32)

    clipped = np.clip(image, lo, hi)
    normalized = (clipped - lo) / (hi - lo)

    return normalized.astype(np.float32)
