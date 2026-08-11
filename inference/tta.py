"""
inference/tta.py

Test Time Augmentation (TTA) for the knee MRI classifier.

Runs each image through the model multiple times -- once as-is, once
horizontally flipped -- and averages the resulting probabilities.
Operates directly on already-loaded image tensors (post-DataLoader),
so it requires no changes to the augmentation pipeline or dataset.

Horizontal flip is used exclusively, consistent with
preprocessing/augmentations.py's training-time choices: knee
orientation (left/right) does not change the underlying pathology,
so a flipped prediction is a valid, independent "view" of the same
finding.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def predict_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    use_flip: bool = True,
) -> torch.Tensor:
    """Run TTA inference on a batch of already-loaded image tensors.

    Args:
        model: A trained model in eval() mode. Caller is responsible
            for calling model.eval() before using this function --
            TTA on a model still in train() mode would apply dropout
            randomly on top of the augmentation views, defeating the
            purpose of deterministic TTA.
        images: FloatTensor, shape (batch_size, 3, H, W), already on
            the correct device.
        use_flip: If True (default), average the original prediction
            with a horizontally-flipped-image prediction. If False,
            this function reduces to a single plain forward pass.

    Returns:
        FloatTensor of averaged probabilities, shape
        (batch_size, num_findings). Already passed through sigmoid.

    Raises:
        RuntimeError: If model is still in training mode (model.training
            is True), since TTA requires deterministic forward passes.
    """
    if model.training:
        raise RuntimeError(
            "predict_with_tta called with model in train() mode. "
            "Call model.eval() first -- TTA requires deterministic "
            "(dropout-disabled) forward passes to be meaningful."
        )

    with torch.no_grad():
        logits_original = model(images)
        probs_original = torch.sigmoid(logits_original)

        if not use_flip:
            return probs_original

        images_flipped = torch.flip(images, dims=[3])
        logits_flipped = model(images_flipped)
        probs_flipped = torch.sigmoid(logits_flipped)

        probs_averaged = (probs_original + probs_flipped) / 2.0

    logger.debug(
        "TTA prediction | batch_size=%d | use_flip=%s",
        images.shape[0], use_flip,
    )

    return probs_averaged