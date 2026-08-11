"""
preprocessing/inspect_dicom.py

Reads one real DICOM file and prints its key metadata and image
properties. This is our first hands-on encounter with real MRI data.
"""

import logging
from pathlib import Path

import numpy as np
import pydicom

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def inspect_dicom_file(filepath: Path) -> None:
    """Load a DICOM file and print its key metadata and pixel properties.

    Args:
        filepath: Path to a .dcm file.

    Raises:
        FileNotFoundError: If the given path does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"DICOM file not found: {filepath}")

    ds = pydicom.dcmread(filepath)

    logger.info("=== DICOM Metadata ===")
    logger.info("Modality: %s", getattr(ds, "Modality", "N/A"))
    logger.info("Rows x Columns: %s x %s", ds.Rows, ds.Columns)
    logger.info("Pixel Spacing: %s", getattr(ds, "PixelSpacing", "N/A"))
    logger.info("Slice Thickness: %s", getattr(ds, "SliceThickness", "N/A"))
    logger.info("Bits Allocated: %s", ds.BitsAllocated)
    logger.info("Bits Stored: %s", ds.BitsStored)

    pixel_array = ds.pixel_array
    logger.info("=== Pixel Array ===")
    logger.info("Shape: %s", pixel_array.shape)
    logger.info("Dtype: %s", pixel_array.dtype)
    logger.info("Min value: %s", np.min(pixel_array))
    logger.info("Max value: %s", np.max(pixel_array))
    logger.info("Mean value: %.2f", np.mean(pixel_array))


if __name__ == "__main__":
    sample_path = Path("data/sample_dicom/1.2.826.0.1.3680043.8.498.10374285052733466977592225981851620435.dcm")
    inspect_dicom_file(sample_path)