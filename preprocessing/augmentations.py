"""
preprocessing/augmentations.py

Training-time data augmentation pipeline for knee MRI slices.
Uses only medically-sensible transformations (see Step 8 theory).
"""

import logging

import albumentations as A
import numpy as np

logger = logging.getLogger(__name__)


def get_train_augmentations(image_size: tuple[int, int] = (224, 224)) -> A.Compose:
    """Build the training-time augmentation pipeline.

    Only includes transformations that preserve valid knee anatomy:
    horizontal flip, small rotations, brightness/contrast jitter, and
    small translations. Deliberately excludes vertical flips and large
    rotations, which would produce anatomically nonsensical images.

    Args:
        image_size: Expected (height, width) of input images -- used
            to configure the shift/scale/rotate transform bounds.

    Returns:
        An Albumentations Compose pipeline to apply during training.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            rotate=(-15, 15),
            translate_percent=(0.0, 0.05),
            scale=(0.95, 1.05),
            p=0.7,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5,
        ),
    ])


def get_val_augmentations() -> A.Compose:
    """Build the validation/inference-time pipeline (no randomness).

    Validation and test data must NEVER be randomly augmented -- we
    need consistent, deterministic evaluation. This returns an
    identity pipeline for symmetry with get_train_augmentations().

    Returns:
        An Albumentations Compose pipeline that applies no transforms.
    """
    return A.Compose([])


def apply_augmentation(image: np.ndarray, pipeline: A.Compose) -> np.ndarray:
    """Apply an augmentation pipeline to a single preprocessed image.

    Args:
        image: A float32 array of shape (H, W, 3) with values in [0, 1],
            as produced by preprocessing.transforms.preprocess_slice.
        pipeline: An Albumentations Compose pipeline (train or val).

    Returns:
        The augmented image, same shape and dtype as the input.
    """
    result = pipeline(image=image)
    augmented = result["image"]

    logger.debug("Applied augmentation: input_shape=%s -> output_shape=%s", image.shape, augmented.shape)

    return augmented