"""
datasets/knee_test_dataset.py

Label-free Dataset for inference on unlabeled knee MRI studies (e.g.
the real Kaggle test set, once available, or any other directory of
studies you want predictions for without ground-truth labels).
"""

import logging
from pathlib import Path

import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

from preprocessing.augmentations import apply_augmentation, get_val_augmentations
from preprocessing.transforms import preprocess_slice

logger = logging.getLogger(__name__)


class KneeMRITestDataset(Dataset):
    """Slice-level Dataset for unlabeled inference.

    Expects the same on-disk layout as training data:
        {series_root}/{study_id}/{series_id}/{filename}.dcm
    """

    def __init__(self, series_root: Path) -> None:
        """Initialize the test dataset by scanning the DICOM folder tree."""
        self.series_root = series_root

        if not series_root.exists():
            raise FileNotFoundError(f"series_root does not exist: {series_root}")

        records = []
        for dcm_path in series_root.rglob("*.dcm"):
            series_id = dcm_path.parent.name
            study_id = dcm_path.parent.parent.name
            records.append({
                "study_id": study_id,
                "series_id": series_id,
                "filename": dcm_path.name,
            })

        if not records:
            raise FileNotFoundError(
                f"No .dcm files found under {series_root}. "
                "Check the folder structure matches "
                "{series_root}/{study_id}/{series_id}/*.dcm"
            )

        self.manifest = pd.DataFrame(records)
        self.augmentation_pipeline = get_val_augmentations()

        logger.info(
            "KneeMRITestDataset initialized | slices=%d | studies=%d | root=%s",
            len(self.manifest), self.manifest["study_id"].nunique(), series_root,
        )

    def __len__(self) -> int:
        """Return the total number of slices found."""
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        """Load and preprocess one slice (no labels, no augmentation)."""
        row = self.manifest.iloc[idx]

        dcm_path = self.series_root / row["study_id"] / row["series_id"] / row["filename"]
        ds = pydicom.dcmread(dcm_path)
        image = preprocess_slice(ds.pixel_array)
        image = apply_augmentation(image, self.augmentation_pipeline)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        return {
            "image": image_tensor,
            "study_id": row["study_id"],
        }