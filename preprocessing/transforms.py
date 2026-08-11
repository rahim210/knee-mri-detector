"""
preprocessing/transforms.py

Reusable preprocessing pipeline for individual MRI slices. This is the
single source of truth for how raw pixel data becomes model-ready
tensors -- used identically during training and inference.
"""

import logging

import cv2
import numpy as np

from utils.image_stats import robust_normalize

logger = logging.getLogger(__name__)


def resize_slice(image: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize a 2D image array to a fixed target size.

    Args:
        image: A 2D NumPy array (single-channel grayscale slice).
        target_size: Desired (height, width) of the output image.

    Returns:
        A resized 2D array with shape target_size.

    Raises:
        ValueError: If the input image is not 2D.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}.")

    height, width = target_size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized


def expand_to_three_channels(image: np.ndarray) -> np.ndarray:
    """Replicate a single-channel grayscale image into 3 channels.

    This lets grayscale MRI slices work with CNN backbones pretrained
    on 3-channel (RGB) natural images.

    Args:
        image: A 2D NumPy array of shape (H, W).

    Returns:
        A 3D NumPy array of shape (H, W, 3), with the same values
        repeated across all 3 channels.

    Raises:
        ValueError: If the input image is not 2D.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}.")

    return np.stack([image, image, image], axis=-1)


def preprocess_slice(
    raw_pixel_array: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Run the full preprocessing pipeline on one raw DICOM pixel array.

    Pipeline order: robust normalize -> resize -> expand to 3 channels.
    This exact order and these exact functions must be used identically
    at both training time and inference time.

    Args:
        raw_pixel_array: The raw 2D pixel array straight from
            `pydicom`'s `ds.pixel_array` (any integer dtype).
        target_size: Desired (height, width) of the final image.

    Returns:
        A float32 NumPy array of shape (H, W, 3) with values in [0, 1],
        ready to be converted into a model input tensor.
    """
    normalized = robust_normalize(raw_pixel_array)
    resized = resize_slice(normalized, target_size)
    three_channel = expand_to_three_channels(resized)

    logger.debug(
        "Preprocessed slice: input_shape=%s -> output_shape=%s",
        raw_pixel_array.shape, three_channel.shape,
    )

    return three_channel