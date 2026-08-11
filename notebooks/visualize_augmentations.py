"""
notebooks/visualize_augmentations.py

Generate several augmented versions of one real MRI slice, side by
side, to visually confirm our augmentation pipeline behaves sensibly.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pydicom

from preprocessing.augmentations import apply_augmentation, get_train_augmentations
from preprocessing.transforms import preprocess_slice

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def visualize_augmentations(filepath: Path, output_path: Path, num_samples: int = 5) -> None:
    """Preprocess one DICOM slice, generate several augmented versions, and save a comparison grid.

    Args:
        filepath: Path to a .dcm file.
        output_path: Where to save the resulting comparison PNG.
        num_samples: How many augmented versions to generate.

    Raises:
        FileNotFoundError: If filepath does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"DICOM file not found: {filepath}")

    ds = pydicom.dcmread(filepath)
    base_image = preprocess_slice(ds.pixel_array)

    train_pipeline = get_train_augmentations()

    fig, axes = plt.subplots(1, num_samples + 1, figsize=(4 * (num_samples + 1), 4))

    axes[0].imshow(base_image)
    axes[0].set_title("Original\n(preprocessed)")
    axes[0].axis("off")

    for i in range(num_samples):
        augmented = apply_augmentation(base_image, train_pipeline)
        axes[i + 1].imshow(augmented)
        axes[i + 1].set_title(f"Augmented #{i + 1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    logger.info("Saved augmentation comparison to: %s", output_path)


if __name__ == "__main__":
    sample_path = Path("data/sample_dicom/1.2.826.0.1.3680043.8.498.10374285052733466977592225981851620435.dcm")
    output_path = Path("outputs/augmentation_comparison.png")
    visualize_augmentations(sample_path, output_path)
    