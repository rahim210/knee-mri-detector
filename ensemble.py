"""
ensemble.py

Model ensembling for the knee MRI abnormality classifier.

Combines predictions from multiple trained checkpoints (potentially
different backbone architectures) by averaging their probabilities,
optionally combined with per-model TTA (inference.tta.predict_with_tta).
This is the standard Kaggle pattern: TTA reduces variance within a
single model, ensembling reduces error by combining genuinely
different models.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.baseline_cnn import KneeMRIClassifier
from inference.tta import predict_with_tta

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load a trained KneeMRIClassifier from a checkpoint file.

    Args:
        checkpoint_path: Path to a .pt file saved by train.py. Must
            contain "backbone_name" and "model_state_dict" keys.
        device: The device to load the model onto.

    Returns:
        The reconstructed model, in eval() mode, ready for inference.

    Raises:
        KeyError: If the checkpoint is missing "backbone_name" (i.e.
            it was saved by an older version of train.py before
            Phase 26 -- retrain or manually specify the backbone).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "backbone_name" not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path} has no 'backbone_name' field. "
            "It was likely saved before Phase 26 -- retrain with the "
            "current train.py, or reconstruct the model manually."
        )

    backbone_name = checkpoint["backbone_name"]
    dropout_rate = checkpoint.get("hyperparameters", {}).get("dropout_rate", 0.3)

    model = KneeMRIClassifier(
        backbone_name=backbone_name,
        pretrained=False,  # we're loading trained weights, not ImageNet init
        dropout_rate=dropout_rate,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(
        "Loaded model from %s | backbone=%s | macro_auc=%.4f (at save time)",
        checkpoint_path, backbone_name, checkpoint.get("macro_auc", float("nan")),
    )

    return model


def ensemble_predict(
    models: list[nn.Module],
    images: torch.Tensor,
    use_tta: bool = True,
) -> torch.Tensor:
    """Average predictions across multiple models for one batch.

    Args:
        models: List of trained models, each already in eval() mode.
        images: FloatTensor, shape (batch_size, 3, H, W), on the
            correct device.
        use_tta: If True, apply TTA (original + horizontal flip)
            within each model before averaging across models. If
            False, each model contributes a single plain prediction.

    Returns:
        FloatTensor of averaged probabilities across all models,
        shape (batch_size, num_findings).

    Raises:
        ValueError: If models is empty.
    """
    if not models:
        raise ValueError("ensemble_predict requires at least one model.")

    all_model_probs = []

    for model in models:
        if use_tta:
            probs = predict_with_tta(model, images, use_flip=True)
        else:
            with torch.no_grad():
                probs = torch.sigmoid(model(images))
        all_model_probs.append(probs)

    stacked = torch.stack(all_model_probs, dim=0)  # (num_models, batch, num_findings)
    averaged = stacked.mean(dim=0)

    return averaged


def evaluate_ensemble(
    checkpoint_paths: list[Path],
    dataloader: DataLoader,
    device: torch.device,
    use_tta: bool = True,
) -> tuple[dict[str, float], float]:
    """Load multiple checkpoints and evaluate their ensemble on a full validation set.

    Args:
        checkpoint_paths: Paths to trained model checkpoints to ensemble.
        dataloader: Validation DataLoader (train=False, shuffle=False).
        device: The device to run computation on.
        use_tta: Whether to apply TTA within each model (see
            ensemble_predict).

    Returns:
        A tuple of (per_finding_auc_dict, macro_auc) for the ensemble.
    """
    from metrics.auc_metrics import compute_macro_auc, compute_per_finding_auc

    models = [load_model_from_checkpoint(path, device) for path in checkpoint_paths]

    all_probs = []
    all_labels = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        probs = ensemble_predict(models, images, use_tta=use_tta)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    per_finding_auc = compute_per_finding_auc(y_true, y_pred)
    macro_auc = compute_macro_auc(per_finding_auc)

    logger.info(
        "Ensemble evaluation complete | num_models=%d | use_tta=%s | Macro-AUC: %.4f",
        len(models), use_tta, macro_auc,
    )

    return per_finding_auc, macro_auc