"""
losses/masked_loss.py

Masked loss wrapper that combines FocalLoss (or any per-element loss
module) with the label_mask mechanism already used by
compute_masked_bce_loss in training/train_loop.py.

Masked-out positions (label_mask == 0) contribute zero to both the
loss sum and the normalization count, so they have no effect on the
gradient -- consistent with how missing/uncertain labels are handled
everywhere else in the pipeline.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_masked_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    focal_loss_fn: nn.Module,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute focal loss with missing-label masking applied.

    Args:
        logits: Raw model outputs, shape (batch_size, num_findings).
        labels: Binary labels, same shape as logits. Values at masked
            positions are ignored and can be arbitrary (e.g. 0).
        label_mask: Binary mask, same shape as logits. 1 where the
            label is known/valid, 0 where it should be excluded from
            the loss.
        focal_loss_fn: An instance of FocalLoss (or compatible module)
            that returns per-element, unreduced loss.
        eps: Small constant to avoid division by zero if an entire
            batch happens to be fully masked for a given finding.

    Returns:
        A scalar tensor: the mean loss over all valid (unmasked)
        label positions in the batch.

    Raises:
        ValueError: If logits, labels, and label_mask shapes don't
            match.
    """
    if not (logits.shape == labels.shape == label_mask.shape):
        raise ValueError(
            "Shape mismatch: logits=%s labels=%s label_mask=%s"
            % (logits.shape, labels.shape, label_mask.shape)
        )

    per_element_loss = focal_loss_fn(logits, labels)

    masked_loss = per_element_loss * label_mask
    num_valid = label_mask.sum()

    if num_valid.item() == 0:
        logger.warning(
            "compute_masked_focal_loss: entire batch is masked -- "
            "returning zero loss with eps fallback."
        )

    total_loss = masked_loss.sum() / (num_valid + eps)
    return total_loss