"""
preprocessing/cross_validation.py

Study-level, multi-label stratified K-fold splitting for the knee MRI
dataset. Ensures (1) no data leakage -- all slices from a study stay
together in either train or validation, never split across both, and
(2) each fold has a reasonably balanced representation of all 12
findings, avoiding the "undefined AUC" problem from Phase 14.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FINDING_NAMES = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


def generate_study_level_folds(
    labels_csv_path: Path,
    n_splits: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Assign each study to one of K folds using multi-label stratification.

    Args:
        labels_csv_path: Path to the CSV with one row per study,
            containing 'StudyInstanceUID' and the 12 finding columns.
        n_splits: Number of folds (K).
        random_seed: Seed for reproducible fold assignment.

    Returns:
        A DataFrame with columns ['StudyInstanceUID', 'fold'], where
        'fold' is an integer in [0, n_splits) indicating which fold
        each study was assigned to for validation.

    Raises:
        ValueError: If any of the 12 expected finding columns are
            missing from the input CSV.
    """
    df = pd.read_csv(labels_csv_path)

    missing_cols = [c for c in FINDING_NAMES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected label columns: {missing_cols}")

    study_ids = df["StudyInstanceUID"].values
    label_matrix = df[FINDING_NAMES].values

    mskf = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_seed
    )

    fold_assignment = np.full(len(df), -1, dtype=int)

    for fold_idx, (_, val_indices) in enumerate(mskf.split(study_ids, label_matrix)):
        fold_assignment[val_indices] = fold_idx

    result = pd.DataFrame({
        "StudyInstanceUID": study_ids,
        "fold": fold_assignment,
    })

    logger.info(
        "Generated %d-fold split across %d studies. Fold sizes: %s",
        n_splits, len(df), result["fold"].value_counts().sort_index().to_dict(),
    )

    return result


def report_fold_balance(
    labels_csv_path: Path,
    fold_assignment: pd.DataFrame,
) -> None:
    """Log per-fold positive-case counts for every finding, to verify balance.

    Args:
        labels_csv_path: Path to the CSV with study-level labels.
        fold_assignment: Output of generate_study_level_folds.
    """
    df = pd.read_csv(labels_csv_path)
    merged = df.merge(fold_assignment, on="StudyInstanceUID")

    logger.info("=== Positive case count per finding, per fold ===")
    for finding in FINDING_NAMES:
        counts = merged.groupby("fold")[finding].sum().to_dict()
        logger.info("%-20s: %s", finding, counts)


if __name__ == "__main__":
    labels_path = Path("data/train_with_derived_labels.csv")
    folds_df = generate_study_level_folds(labels_path, n_splits=5)

    output_path = Path("data/study_folds.csv")
    folds_df.to_csv(output_path, index=False)
    logging.getLogger(__name__).info("Saved fold assignment to: %s", output_path)

    report_fold_balance(labels_path, folds_df)