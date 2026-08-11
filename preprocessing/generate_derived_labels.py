"""
preprocessing/generate_derived_labels.py

Apply the validated rule-based extractor to every study that lacks
structured ground-truth labels, generating derived labels from the
English-translated report text. Studies that already have real
structured labels keep those original values untouched.

Output: data/train_with_derived_labels.csv -- same shape as
train.csv, but with all 12 finding columns fully populated (no NaN),
plus a 'label_source' column marking each row as either 'ground_truth'
or 'derived' so downstream code can weight/filter by confidence later.
"""

import logging
from pathlib import Path

import pandas as pd

from preprocessing.extract_labels import FINDING_COLUMNS, extract_labels_from_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in derived labels for studies missing structured ground truth.

    Args:
        df: DataFrame with report_english and the 12 finding columns
            (NaN where ground truth is missing).

    Returns:
        Copy of df with all 12 finding columns fully populated, plus
        a new 'label_source' column ('ground_truth' or 'derived').
    """
    df = df.copy()
    has_ground_truth = df[FINDING_COLUMNS].notna().any(axis=1)
    df["label_source"] = has_ground_truth.map({True: "ground_truth", False: "derived"})

    total = len(df)
    derived_count = 0

    for idx in df.index[~has_ground_truth]:
        report = df.at[idx, "report_english"]
        predicted = extract_labels_from_report(report)
        for finding in FINDING_COLUMNS:
            df.at[idx, finding] = predicted[finding]

        derived_count += 1
        if derived_count % 500 == 0:
            logger.info("[%d/%d] derived", derived_count, total - has_ground_truth.sum())

    logger.info(
        "Done. %d ground-truth rows kept as-is, %d rows filled with derived labels.",
        has_ground_truth.sum(), derived_count,
    )
    return df


def main() -> None:
    """Run label generation on the full translated dataset."""
    input_path = Path("data/train_with_english_reports.csv")
    output_path = Path("data/train_with_derived_labels.csv")

    df = pd.read_csv(input_path)
    logger.info("Loaded %d total studies", len(df))

    result_df = generate_labels(df)
    result_df.to_csv(output_path, index=False)

    logger.info("Saved to %s", output_path)
    logger.info("Label source breakdown:\n%s", result_df["label_source"].value_counts())

    logger.info("Positive rate per finding (derived + ground truth combined):")
    for finding in FINDING_COLUMNS:
        rate = result_df[finding].astype(float).mean()
        logger.info("  %s: %.1f%%", finding, rate * 100)


if __name__ == "__main__":
    main()