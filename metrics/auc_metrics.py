"""
metrics/auc_metrics.py

Per-finding and macro-averaged AUC-ROC computation, matching the
competition's official evaluation metric exactly (see
configs/competition_notes.yaml -> metric: Macro-averaged AUC-ROC).
"""

import logging

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

FINDING_NAMES = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


def compute_per_finding_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    finding_names: list[str] = FINDING_NAMES,
) -> dict[str, float]:
    """Compute AUC-ROC independently for each of the 12 findings.

    Args:
        y_true: Array of shape (N, 12) with binary ground-truth labels.
        y_pred: Array of shape (N, 12) with predicted probabilities in [0, 1].
        finding_names: Names of the 12 findings, in column order.

    Returns:
        A dict mapping finding name -> AUC score. A finding is mapped to
        NaN (and a warning is logged) if it has only one class present
        in y_true, since AUC is mathematically undefined in that case.

    Raises:
        ValueError: If y_true and y_pred shapes don't match, or don't
            match the number of finding_names.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.shape[1] != len(finding_names):
        raise ValueError(
            f"Expected {len(finding_names)} findings, got {y_true.shape[1]} columns."
        )

    results: dict[str, float] = {}

    for i, name in enumerate(finding_names):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]

        unique_classes = np.unique(col_true)
        if len(unique_classes) < 2:
            logger.warning(
                "Finding '%s' has only one class present (%s) in this set -- "
                "AUC undefined, skipping.",
                name, unique_classes,
            )
            results[name] = float("nan")
            continue

        auc = roc_auc_score(col_true, col_pred)
        results[name] = auc

    return results


def compute_macro_auc(per_finding_auc: dict[str, float]) -> float:
    """Average per-finding AUC scores into a single macro-AUC score.

    NaN findings (undefined AUC due to missing class) are excluded from
    the average rather than treated as zero, to avoid unfairly punishing
    the score for a data artifact rather than model quality.

    Args:
        per_finding_auc: Output of compute_per_finding_auc.

    Returns:
        The macro-averaged AUC across all valid (non-NaN) findings.

    Raises:
        ValueError: If every finding is NaN (nothing to average).
    """
    valid_scores = [v for v in per_finding_auc.values() if not np.isnan(v)]

    if not valid_scores:
        raise ValueError("All findings had undefined AUC -- cannot compute macro-AUC.")

    macro_auc = float(np.mean(valid_scores))
    return macro_auc


def log_auc_report(per_finding_auc: dict[str, float], macro_auc: float) -> None:
    """Print a formatted report of per-finding and macro AUC scores.

    Args:
        per_finding_auc: Output of compute_per_finding_auc.
        macro_auc: Output of compute_macro_auc.
    """
    logger.info("=== Per-Finding AUC-ROC ===")
    sorted_findings = sorted(per_finding_auc.items(), key=lambda x: (np.isnan(x[1]), x[1]))
    for name, auc in sorted_findings:
        if np.isnan(auc):
            logger.info("%-20s: undefined (single class)", name)
        else:
            logger.info("%-20s: %.4f", name, auc)
    logger.info("=== Macro-AUC (competition metric): %.4f ===", macro_auc)