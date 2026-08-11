"""
datasets/knee_dataset.py

PyTorch Dataset for the RSNA Knee Abnormality Detection subset.
Each item is a single 2D DICOM slice, preprocessed and optionally
augmented, paired with its study-level multi-label target vector.

Labels are NaN for findings that weren't reported for a given study.
We return both the label vector (with NaN filled as 0) and a mask
vector (1 = labeled, 0 = missing) so the loss function can later
ignore missing labels instead of treating them as negative.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

from preprocessing.augmentations import apply_augmentation, get_train_augmentations, get_val_augmentations
from preprocessing.transforms import preprocess_slice

logger = logging.getLogger(__name__)

FINDING_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


class KneeMRIDataset(Dataset):
    """Slice-level Dataset for knee MRI multi-label abnormality detection.

    Each __getitem__ call returns one preprocessed (and optionally
    augmented) 2D slice, along with the study-level label vector and
    a mask indicating which labels are actually known (not NaN).

    Attributes:
        manifest: DataFrame of study_id, series_id, filename, path, size
            for every DICOM file included in this dataset.
        labels_df: DataFrame indexed by StudyInstanceUID with the 12
            finding columns.
        data_root: Root folder containing train_series/.
        train: Whether to apply training-time augmentation.
    """

    def __init__(
        self,
        manifest_path: Path,
        labels_csv_path: Path,
        data_root: Path,
        train: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            manifest_path: Path to subset_manifest_clean.csv.
            labels_csv_path: Path to train.csv.
            data_root: Root folder containing train_series/ (e.g. "data").
            train: If True, apply training augmentations; if False,
                apply only the deterministic validation pipeline.
        """
        self.manifest = pd.read_csv(manifest_path)
        self.data_root = data_root
        self.train = train

        labels_df = pd.read_csv(labels_csv_path)
        self.labels_df = labels_df.set_index("StudyInstanceUID")

        self.augmentation_pipeline = (
            get_train_augmentations() if train else get_val_augmentations()
        )

        logger.info(
            "KneeMRIDataset initialized | slices=%d | studies=%d | train=%s",
            len(self.manifest), self.manifest["study_id"].nunique(), train,
        )

    def __len__(self) -> int:
        """Return the total number of slices in this dataset."""
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        """Load, preprocess, and augment one slice.

        Args:
            idx: Row index into self.manifest.

        Returns:
            Dict with keys:
                image: FloatTensor, shape (3, H, W).
                label: FloatTensor, shape (12,), NaN filled as 0.0.
                label_mask: FloatTensor, shape (12,), 1.0 where the
                    label was actually known, 0.0 where it was NaN.
                study_id: str, the StudyInstanceUID (for debugging/
                    later aggregation across slices of the same study).
        """
        row = self.manifest.iloc[idx]

        dcm_path = (
            self.data_root / "train_series" / row["study_id"] / row["series_id"] / row["filename"]
        )
        ds = pydicom.dcmread(dcm_path)
        image = preprocess_slice(ds.pixel_array)
        image = apply_augmentation(image, self.augmentation_pipeline)

        label, label_mask = self._get_label_and_mask(row["study_id"])

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        return {
            "image": image_tensor,
            "label": torch.from_numpy(label).float(),
            "label_mask": torch.from_numpy(label_mask).float(),
            "study_id": row["study_id"],
        }

    def _get_label_and_mask(self, study_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Look up the label vector and missing-label mask for a study.

        Args:
            study_id: StudyInstanceUID to look up.

        Returns:
            Tuple of (label array, mask array), each shape (12,).
            label has NaN positions filled with 0.0; mask has 1.0
            where the original value was not NaN, 0.0 where it was.
        """
        row = self.labels_df.loc[study_id, FINDING_COLUMNS]
        raw_values = row.to_numpy(dtype=np.float32)

        mask = (~np.isnan(raw_values)).astype(np.float32)
        label = np.nan_to_num(raw_values, nan=0.0).astype(np.float32)

        return label, mask