"""
notebooks/test_dataset.py

Sanity check for KneeMRIDataset: load a few real samples and verify
shapes, dtypes, and label/mask values look correct.
"""

import logging
from pathlib import Path

from datasets.knee_dataset import KneeMRIDataset, FINDING_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_dataset() -> None:
    """Load the dataset and print details for the first few samples."""
    dataset = KneeMRIDataset(
        manifest_path=Path("data/subset_manifest_clean.csv"),
        labels_csv_path=Path("data/train_with_derived_labels.csv"),
        data_root=Path("data"),
        train=True,
    )

    logger.info("Dataset length: %d", len(dataset))

    for i in range(3):
        sample = dataset[i]
        logger.info("--- Sample %d ---", i)
        logger.info("image shape: %s, dtype: %s", sample["image"].shape, sample["image"].dtype)
        logger.info("label shape: %s", sample["label"].shape)
        logger.info("label values: %s", sample["label"].tolist())
        logger.info("label_mask values: %s", sample["label_mask"].tolist())
        logger.info("study_id: %s", sample["study_id"])
        logger.info("known findings: %s", [
            FINDING_COLUMNS[j] for j, m in enumerate(sample["label_mask"]) if m == 1.0
        ])


if __name__ == "__main__":
    test_dataset()