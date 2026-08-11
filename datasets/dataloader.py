"""
datasets/dataloader.py

Builds PyTorch DataLoaders for the KneeMRIDataset. Tuned for CPU-only
training (no CUDA GPU detected on this machine) -- pin_memory is
disabled since it only benefits CUDA transfers, and num_workers is
set conservatively relative to available CPU cores.
"""

import logging
from pathlib import Path

from torch.utils.data import DataLoader

from datasets.knee_dataset import KneeMRIDataset

logger = logging.getLogger(__name__)

NUM_WORKERS = 4  # conservative for an 8-core CPU; leaves cores for OS/main process
BATCH_SIZE = 16  # moderate default for CPU; may need tuning once training starts


def build_dataloader(
    manifest_path: Path,
    labels_csv_path: Path,
    data_root: Path,
    train: bool,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    """Construct a DataLoader wrapping KneeMRIDataset.

    Args:
        manifest_path: Path to the file manifest CSV (e.g.
            subset_manifest_clean.csv).
        labels_csv_path: Path to the labels CSV (e.g.
            train_with_derived_labels.csv).
        data_root: Root folder containing train_series/.
        train: If True, builds a training loader (shuffled, training
            augmentations). If False, builds a validation/eval loader
            (not shuffled, deterministic preprocessing only).
        batch_size: Number of slices per batch.
        num_workers: Number of parallel subprocess workers for data
            loading. Set to 0 to disable multiprocessing entirely
            (useful for debugging, since worker process errors can be
            harder to read than main-process errors).

    Returns:
        A configured DataLoader.
    """
    dataset = KneeMRIDataset(
        manifest_path=manifest_path,
        labels_csv_path=labels_csv_path,
        data_root=data_root,
        train=train,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=False,  # no CUDA GPU on this machine; pin_memory has no benefit
        drop_last=train,  # drop incomplete final batch during training for stable batch norm stats
    )

    logger.info(
        "Built DataLoader | train=%s | batch_size=%d | num_workers=%d | batches_per_epoch=%d",
        train, batch_size, num_workers, len(loader),
    )

    return loader