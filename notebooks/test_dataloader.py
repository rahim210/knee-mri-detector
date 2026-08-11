"""
notebooks/test_dataloader.py

Sanity check for the DataLoader: pull one batch, verify shapes,
dtypes, and that batching/collation works correctly across images,
labels, masks, and study_id strings.

IMPORTANT (Windows): code using num_workers > 0 must be guarded by
if __name__ == "__main__", or worker process spawning will raise a
RuntimeError. See the main() guard below.
"""

import logging
from pathlib import Path

from datasets.dataloader import build_dataloader

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_dataloader() -> None:
    """Build a training DataLoader and inspect the first batch."""
    loader = build_dataloader(
        manifest_path=Path("data/subset_manifest_clean.csv"),
        labels_csv_path=Path("data/train_with_derived_labels.csv"),
        data_root=Path("data"),
        train=True,
        batch_size=16,
        num_workers=4,
    )

    batch = next(iter(loader))

    logger.info("Batch keys: %s", list(batch.keys()))
    logger.info("image batch shape: %s, dtype: %s", batch["image"].shape, batch["image"].dtype)
    logger.info("label batch shape: %s, dtype: %s", batch["label"].shape, batch["label"].dtype)
    logger.info("label_mask batch shape: %s", batch["label_mask"].shape)
    logger.info("study_id batch (first 3): %s", batch["study_id"][:3])
    logger.info("Number of batches per epoch: %d", len(loader))


if __name__ == "__main__":
    test_dataloader()