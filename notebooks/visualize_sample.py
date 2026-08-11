"""
notebooks/visualize_sample.py

Visualize a real DICOM slice as an actual image, using both naive
and robust (percentile-based) normalization for comparison.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pydicom

from utils.image_stats import compute_intensity_stats, robust_normalize

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def visualize_dicom_slice(filepath: Path, output_path: Path) -> None:
    """Load a DICOM slice, normalize it, and save a side-by-side comparison image.

    Args:
        filepath: Path to a .dcm file.
        output_path: Where to save the resulting comparison PNG.

    Raises:
        FileNotFoundError: If filepath does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"DICOM file not found: {filepath}")

    ds = pydicom.dcmread(filepath)
    pixel_array = ds.pixel_array

    stats = compute_intensity_stats(pixel_array)
    normalized = robust_normalize(pixel_array)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(pixel_array, cmap="gray")
    axes[0].set_title(f"Raw pixel values\n(min={stats.min_val:.0f}, max={stats.max_val:.0f})")
    axes[0].axis("off")

    axes[1].imshow(normalized, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Robust normalized (p1-p99 clipped to [0,1])")
    axes[1].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    logger.info("Saved visualization to: %s", output_path)


if __name__ == "__main__":
    sample_path = Path("data/sample_dicom/1.2.826.0.1.3680043.8.498.10374285052733466977592225981851620435.dcm")
    output_path = Path("outputs/sample_slice_visualization.png")
    visualize_dicom_slice(sample_path, output_path)