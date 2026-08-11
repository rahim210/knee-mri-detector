"""
notebooks/test_baseline_model.py

Sanity check: run one real batch from our DataLoader through the
baseline model and verify output shape, dtype, and that logits look
like reasonable raw values (not NaN, not all identical).
"""

import logging
from pathlib import Path

import torch

from datasets.dataloader import build_dataloader
from models.baseline_cnn import BaselineKneeCNN

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_baseline_model() -> None:
    """Run one batch through the model and inspect the output."""
    loader = build_dataloader(
        manifest_path=Path("data/subset_manifest_clean.csv"),
        labels_csv_path=Path("data/train_with_derived_labels.csv"),
        data_root=Path("data"),
        train=True,
        batch_size=8,  # small batch for a quick CPU test
        num_workers=0,  # disable multiprocessing for simpler debugging here
    )

    batch = next(iter(loader))

    model = BaselineKneeCNN(pretrained=True)
    model.eval()  # inference mode for this quick test (disables dropout/batchnorm updates)

    with torch.no_grad():
        logits = model(batch["image"])

    probabilities = torch.sigmoid(logits)

    logger.info("Logits shape: %s, dtype: %s", logits.shape, logits.dtype)
    logger.info("Logits contain NaN: %s", torch.isnan(logits).any().item())
    logger.info("Logits sample (first item): %s", logits[0].tolist())
    logger.info("Probabilities sample (first item): %s", probabilities[0].tolist())
    logger.info("Ground truth labels (first item): %s", batch["label"][0].tolist())


if __name__ == "__main__":
    test_baseline_model()