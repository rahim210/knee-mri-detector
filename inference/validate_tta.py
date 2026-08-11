"""
inference/validate_tta.py

Full-validation-set evaluation using Test Time Augmentation, mirroring
training/train_loop.py's validate_with_metrics but using
predict_with_tta instead of a single plain forward pass. Used to
directly compare TTA on vs. off, measured by macro-AUC.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from inference.tta import predict_with_tta

logger = logging.getLogger(__name__)


def validate_with_tta_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_flip: bool = True,
) -> tuple[dict[str, float], float]:
    """Run full-validation-set inference with TTA and compute AUC metrics.

    Args:
        model: A trained model. Will be set to eval() mode internally.
        dataloader: Validation DataLoader (train=False, shuffle=False).
        device: The device to run computation on (cpu or cuda).
        use_flip: If True, average original + horizontally-flipped
            predictions (see inference.tta.predict_with_tta). If
            False, this is equivalent to plain (non-TTA) inference,
            useful as a same-code-path baseline for comparison.

    Returns:
        A tuple of (per_finding_auc_dict, macro_auc), matching the
        return shape of training.train_loop.validate_with_metrics
        (minus val_loss, since TTA is inference-only and has no
        associated loss).
    """
    from metrics.auc_metrics import compute_macro_auc, compute_per_finding_auc

    model.eval()

    all_probs = []
    all_labels = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        probs = predict_with_tta(model, images, use_flip=use_flip)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    per_finding_auc = compute_per_finding_auc(y_true, y_pred)
    macro_auc = compute_macro_auc(per_finding_auc)

    logger.info(
        "TTA validation complete | use_flip=%s | Macro-AUC: %.4f",
        use_flip, macro_auc,
    )

    return per_finding_auc, macro_auc