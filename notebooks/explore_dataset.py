"""
notebooks/explore_dataset.py

Initial exploration of the RSNA Knee Abnormality Detection dataset.
Answers: how many studies have labels, what's the label distribution,
and how many series per study.
"""

import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LABEL_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


def explore_train_csv(path: str) -> pd.DataFrame:
    """Load train.csv and print summary statistics about labels.

    Args:
        path: Path to train.csv.

    Returns:
        The loaded DataFrame, for further use.
    """
    df = pd.read_csv(path)

    logger.info("=== train.csv Overview ===")
    logger.info("Total studies: %d", len(df))
    logger.info("Columns: %s", list(df.columns))

    # A study is "labeled" if at least one label column is not NaN
    labeled_mask = df[LABEL_COLUMNS].notna().any(axis=1)
    logger.info(
        "Studies WITH direct labels: %d (%.1f%%)",
        labeled_mask.sum(),
        100 * labeled_mask.mean(),
    )
    logger.info(
        "Studies WITHOUT direct labels (report-only): %d (%.1f%%)",
        (~labeled_mask).sum(),
        100 * (~labeled_mask).mean(),
    )

    logger.info("=== Label Prevalence (among labeled studies) ===")
    labeled_df = df[labeled_mask]
    for col in LABEL_COLUMNS:
        positive_count = (labeled_df[col] == 1).sum()
        total_valid = labeled_df[col].notna().sum()
        pct = 100 * positive_count / total_valid if total_valid > 0 else 0
        logger.info("%-20s: %4d / %4d positive (%.1f%%)", col, positive_count, total_valid, pct)

    return df


def explore_train_series_csv(path: str) -> pd.DataFrame:
    """Load train_series.csv and print series-level summary statistics.

    Args:
        path: Path to train_series.csv.

    Returns:
        The loaded DataFrame, for further use.
    """
    df = pd.read_csv(path)

    logger.info("=== train_series.csv Overview ===")
    logger.info("Total series: %d", len(df))
    logger.info("Unique studies: %d", df["StudyInstanceUID"].nunique())

    series_per_study = df.groupby("StudyInstanceUID").size()
    logger.info(
        "Series per study -- min: %d, max: %d, mean: %.1f",
        series_per_study.min(),
        series_per_study.max(),
        series_per_study.mean(),
    )

    logger.info("=== Anatomical Plane Distribution ===")
    logger.info("%s", df["Anatomical_Plane"].value_counts().to_string())

    return df


if __name__ == "__main__":
    train_df = explore_train_csv("data/train.csv")
    series_df = explore_train_series_csv("data/train_series.csv")