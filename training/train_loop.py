"""
training/train_loop.py

Core training loop for the baseline (and later, all) knee MRI
classifiers. Handles one epoch of training: forward pass, masked
BCE loss computation, backpropagation, and optimizer updates.

The label_mask is used to zero out loss contributions from any
label position that wasn't actually known for a given study -- in
our current dataset every position is populated (ground truth or
derived), but this mechanism stays in place so we can later weight
down or exclude low-confidence derived labels without changing the
training loop itself.
"""

import numpy as np
import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from losses.masked_loss import compute_masked_focal_loss

logger = logging.getLogger(__name__)


def compute_masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute BCE loss, ignoring any masked-out (unknown) label positions.

    Args:
        logits: Raw model outputs, shape (batch_size, num_findings).
        labels: Ground-truth/derived binary labels, same shape.
        label_mask: 1.0 where the label is known/should count toward
            loss, 0.0 where it should be ignored, same shape.
        criterion: A BCEWithLogitsLoss instance with reduction="none".

    Returns:
        Scalar loss tensor: the mean loss over only the masked-in
        (known) label positions.
    """
    per_element_loss = criterion(logits, labels)
    masked_loss = per_element_loss * label_mask

    total_known = label_mask.sum()
    if total_known == 0:
        # Should not happen in our current dataset, but guard against
        # a batch with zero known labels causing a divide-by-zero.
        return torch.tensor(0.0, requires_grad=True)

    return masked_loss.sum() / total_known


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_num: int,
    scaler: GradScaler | None = None,
    log_every: int = 20,
) -> float:
    """Run one full training epoch.

    Args:
        model: The model to train.
        dataloader: Training DataLoader (should have train=True,
            shuffle=True upstream).
        optimizer: The optimizer (e.g. Adam) managing model.parameters().
        criterion: A BCEWithLogitsLoss instance with reduction="none".
        device: The device to run computation on (cpu or cuda).
        epoch_num: Current epoch number, for logging.
        scaler: A GradScaler for mixed-precision training on CUDA. Pass
            None (the default) to train in standard float32 -- this is
            required on CPU, where mixed precision provides no benefit
            and autocast will simply behave as a no-op.
        log_every: Print progress every N batches.

    Returns:
        The mean training loss across the entire epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)
    epoch_start = time.time()

    for batch_idx, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        label_mask = batch["label_mask"].to(device)

        optimizer.zero_grad()

        use_amp = scaler is not None and device.type == "cuda"

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = compute_masked_focal_loss(logits, labels, label_mask, criterion)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        if batch_idx % log_every == 0 or batch_idx == num_batches:
            elapsed = time.time() - epoch_start
            avg_loss_so_far = running_loss / batch_idx
            logger.info(
                "Epoch %d | Batch %d/%d | avg_loss=%.4f | elapsed=%.1fs",
                epoch_num, batch_idx, num_batches, avg_loss_so_far, elapsed,
            )

    epoch_loss = running_loss / num_batches
    epoch_time = time.time() - epoch_start
    logger.info(
        "Epoch %d complete | avg_loss=%.4f | time=%.1fs",
        epoch_num, epoch_loss, epoch_time,
    )

    return epoch_loss


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch_num: int,
) -> float:
    """Run one full validation pass (no gradient updates).

    Args:
        model: The model to evaluate.
        dataloader: Validation DataLoader (should have train=False,
            shuffle=False upstream).
        criterion: A BCEWithLogitsLoss instance with reduction="none".
        device: The device to run computation on (cpu or cuda).
        epoch_num: Current epoch number, for logging.

    Returns:
        The mean validation loss across the entire validation set.
    """
    model.eval()
    running_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            label_mask = batch["label_mask"].to(device)

            logits = model(images)
            loss = compute_masked_focal_loss(logits, labels, label_mask, criterion)

            running_loss += loss.item()

    val_loss = running_loss / num_batches
    logger.info("Epoch %d | Validation loss: %.4f", epoch_num, val_loss)

    return val_loss


def validate_with_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch_num: int,
) -> tuple[float, dict[str, float], float]:
    """Run validation, computing both loss and per-finding/macro AUC-ROC.

    This is a superset of validate_one_epoch: it does everything that
    function does, but additionally collects all predictions and
    ground-truth labels across the epoch to compute the competition's
    actual scoring metric (macro-averaged AUC-ROC across 12 findings).

    Args:
        model: The model to evaluate.
        dataloader: Validation DataLoader (train=False, shuffle=False).
        criterion: A BCEWithLogitsLoss instance with reduction="none".
        device: The device to run computation on (cpu or cuda).
        epoch_num: Current epoch number, for logging.

    Returns:
        A tuple of (val_loss, per_finding_auc_dict, macro_auc).
    """
    from metrics.auc_metrics import compute_macro_auc, compute_per_finding_auc

    model.eval()
    running_loss = 0.0
    num_batches = len(dataloader)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            label_mask = batch["label_mask"].to(device)

            logits = model(images)
            loss = compute_masked_focal_loss(logits, labels, label_mask, criterion)
            running_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    val_loss = running_loss / num_batches
    logger.info("Epoch %d | Validation loss: %.4f", epoch_num, val_loss)

    y_pred = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    per_finding_auc = compute_per_finding_auc(y_true, y_pred)
    macro_auc = compute_macro_auc(per_finding_auc)

    logger.info("Epoch %d | Macro-AUC: %.4f", epoch_num, macro_auc)

    return val_loss, per_finding_auc, macro_auc