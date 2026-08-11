"""
preprocessing/split_train_val.py

Split our local subset into training and validation sets at the
STUDY level (never at the slice level) to prevent data leakage --
all slices belonging to a given study go entirely into one split or
the other, never both.

With only 15 studies locally, this split is necessarily small and
the resulting validation signal will be noisy. This is a known,
acceptable limitation of the local subset; proper k-fold cross-
validation (Phase 15) and full-dataset training (via Kaggle) will
give more reliable estimates later.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

VAL_FRACTION = 0.2
RANDOM_SEED = 42


def split_manifest_by_study(
    manifest_path: Path,
    val_fraction: float = VAL_FRACTION,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a slice-level manifest into train/val at the study level.

    Args:
        manifest_path: Path to the full manifest CSV (e.g.
            subset_manifest_clean.csv), containing one row per slice.
        val_fraction: Fraction of STUDIES (not slices) to hold out
            for validation.
        random_seed: For reproducible splits.

    Returns:
        Tuple of (train_manifest, val_manifest) DataFrames, each a
        subset of rows from the original manifest.
    """
    manifest = pd.read_csv(manifest_path)
    unique_studies = manifest["study_id"].unique()

    train_studies, val_studies = train_test_split(
        unique_studies, test_size=val_fraction, random_state=random_seed
    )

    train_manifest = manifest[manifest["study_id"].isin(train_studies)].reset_index(drop=True)
    val_manifest = manifest[manifest["study_id"].isin(val_studies)].reset_index(drop=True)

    logger.info(
        "Split %d studies -> %d train studies (%d slices), %d val studies (%d slices)",
        len(unique_studies), len(train_studies), len(train_manifest),
        len(val_studies), len(val_manifest),
    )

    # Sanity check: confirm zero overlap between train and val studies
    overlap = set(train_studies) & set(val_studies)
    if overlap:
        raise ValueError(f"Data leakage detected! Studies in both splits: {overlap}")

    return train_manifest, val_manifest


def main() -> None:
    """Run the split and save both manifests to disk."""
    manifest_path = Path("data/subset_manifest_clean.csv")

    train_manifest, val_manifest = split_manifest_by_study(manifest_path)

    train_path = Path("data/manifest_train.csv")
    val_path = Path("data/manifest_val.csv")

    train_manifest.to_csv(train_path, index=False)
    val_manifest.to_csv(val_path, index=False)

    logger.info("Saved train manifest to %s", train_path)
    logger.info("Saved val manifest to %s", val_path)


if __name__ == "__main__":
    main()
    