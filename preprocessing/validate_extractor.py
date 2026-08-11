"""
preprocessing/validate_extractor.py

Validate the rule-based label extractor against the 58 studies that
have ground-truth structured labels. Prints per-finding accuracy,
precision, and recall so we know exactly which findings the
extractor handles well and which need more work before applying it
to the full ~4,349 report-only studies.
"""

import logging

import pandas as pd

from preprocessing.extract_labels import FINDING_COLUMNS, extract_labels_from_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_ground_truth() -> pd.DataFrame:
    """Load the studies that have real structured labels, with English report text.

    Returns:
        DataFrame with StudyInstanceUID, report_english, and the 12
        finding columns (ground-truth values only, no NaN rows).
    """
    df = pd.read_csv("data/train_with_english_reports.csv")
    labeled = df[df[FINDING_COLUMNS].notna().any(axis=1)].reset_index(drop=True)
    return labeled


def evaluate(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Run the extractor on each report and compare against ground truth.

    Args:
        labeled_df: DataFrame with report_english and ground-truth
            finding columns.

    Returns:
        DataFrame with one row per finding: accuracy, precision,
        recall, and counts of TP/FP/TN/FN.
    """
    predictions = []
    for report in labeled_df["report_english"]:
        predictions.append(extract_labels_from_report(report))
    pred_df = pd.DataFrame(predictions)

    rows = []
    for finding in FINDING_COLUMNS:
        y_true = labeled_df[finding].astype(float)
        y_pred = pred_df[finding].astype(float)

        tp = ((y_true == 1.0) & (y_pred == 1.0)).sum()
        fp = ((y_true == 0.0) & (y_pred == 1.0)).sum()
        tn = ((y_true == 0.0) & (y_pred == 0.0)).sum()
        fn = ((y_true == 1.0) & (y_pred == 0.0)).sum()

        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        rows.append({
            "finding": finding,
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3) if precision == precision else None,
            "recall": round(recall, 3) if recall == recall else None,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        })

    return pd.DataFrame(rows)


def main() -> None:
    """Run validation and print a per-finding report."""
    labeled_df = load_ground_truth()
    logger.info("Validating against %d ground-truth labeled studies", len(labeled_df))

    results = evaluate(labeled_df)
    print()
    print(results.to_string(index=False))
    print()
    print(f"Overall mean accuracy: {results['accuracy'].mean():.3f}")


if __name__ == "__main__":
    main()