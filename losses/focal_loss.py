"""
losses/focal_loss.py

Class-weighted focal loss for multi-label classification, designed
to address the class imbalance across our 12 findings (Phase 5/15
showed prevalence ranging from ~55 to ~381 positives per fold) and
to keep the model learning from hard examples throughout training,
rather than being dominated by easy, already-well-classified ones.

Fully compatible with the label_mask mechanism from
training/train_loop.py -- masked-out positions contribute zero loss,
exactly as with the existing compute_masked_bce_loss.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Class-weighted focal loss for multi-label binary classification.

    Combines two ideas:
    1. Per-class weighting (pos_weight), to counteract label imbalance
       across the 12 findings.
    2. Focal modulation ((1 - p_t)^gamma), to down-weight easy examples
       and keep gradient signal focused on hard, currently-misclassified
       examples throughout training.

    Attributes:
        gamma: Focusing parameter -- higher values down-weight easy
            examples more aggressively. gamma=0 reduces to plain
            (optionally class-weighted) BCE.
        pos_weight: Per-finding positive-class weight tensor, or None
            for unweighted loss.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        """Initialize the focal loss.

        Args:
            gamma: Focusing parameter. Standard default from the
                original Focal Loss paper is 2.0.
            pos_weight: Optional tensor of shape (num_findings,) giving
                a per-finding weight for positive examples, analogous
                to nn.BCEWithLogitsLoss's pos_weight argument. Pass
                None for no class weighting.

        Raises:
            ValueError: If gamma is negative.
        """
        super().__init__()

        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}.")

        self.gamma = gamma
        self.pos_weight = pos_weight

        logger.info(
            "FocalLoss initialized | gamma=%.2f | pos_weight=%s",
            gamma, "set" if pos_weight is not None else "None",
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-element focal loss (no reduction applied).

        Args:
            logits: Raw model outputs, shape (batch_size, num_findings).
            labels: Binary ground-truth/derived labels, same shape.

        Returns:
            Per-element loss tensor, same shape as input -- caller
            (e.g. compute_masked_bce_loss-style logic) is responsible
            for masking and reducing, exactly as with reduction="none"
            BCEWithLogitsLoss.
        """
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_weight, reduction="none",
        )

        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_modulation = (1 - p_t) ** self.gamma

        focal_loss = focal_modulation * bce
        return focal_loss


def compute_pos_weights_from_labels(
    labels_csv_path: str,
    finding_names: list[str],
) -> torch.Tensor:
    """Compute per-finding positive-class weights from a labels CSV.

    Uses the standard inverse-frequency formula:
    weight = num_negatives / num_positives, so findings with fewer
    positive examples get proportionally larger weight.

    Args:
        labels_csv_path: Path to a CSV with one row per study and one
            column per finding (values 0 or 1).
        finding_names: Names of the finding columns, in the order
            expected by the model's output.

    Returns:
        A FloatTensor of shape (len(finding_names),) with one weight
        per finding.
    """
    import pandas as pd

    df = pd.read_csv(labels_csv_path)

    weights = []
    for name in finding_names:
        num_positive = (df[name] == 1).sum()
        num_negative = (df[name] == 0).sum()

        if num_positive == 0:
            logger.warning(
                "Finding '%s' has zero positive examples in %s -- "
                "using weight=1.0 as a safe fallback.",
                name, labels_csv_path,
            )
            weights.append(1.0)
        else:
            weights.append(num_negative / num_positive)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    logger.info("Computed pos_weights: %s", dict(zip(finding_names, weights)))

    return weight_tensor