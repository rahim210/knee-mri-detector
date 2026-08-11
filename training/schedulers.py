"""
training/schedulers.py

Learning rate scheduling utilities. Wraps PyTorch's built-in
ReduceLROnPlateau with sensible, documented defaults for our
small-data, overfitting-prone setup (see Phase 13/14 findings).
"""

import logging

import torch

logger = logging.getLogger(__name__)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    mode: str = "max",
    factor: float = 0.5,
    patience: int = 2,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Build a ReduceLROnPlateau scheduler for adaptive learning rate decay.

    Watches a validation metric each epoch and reduces the learning
    rate when that metric stops improving -- rather than following a
    fixed decay schedule regardless of actual training behavior.

    Args:
        optimizer: The optimizer whose learning rate will be adjusted.
        mode: 'max' if higher metric values are better (e.g. macro-AUC),
            'min' if lower is better (e.g. validation loss).
        factor: Multiply the learning rate by this factor on each
            reduction (0.5 = halve the learning rate).
        patience: Number of epochs with no improvement to wait before
            reducing the learning rate.
        min_lr: Floor -- the learning rate will never be reduced below
            this value.

    Returns:
        A configured ReduceLROnPlateau scheduler.
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )

    logger.info(
        "Scheduler initialized | mode=%s | factor=%.2f | patience=%d | min_lr=%.2e",
        mode, factor, patience, min_lr,
    )

    return scheduler


def step_scheduler_and_log(
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    metric_value: float,
    optimizer: torch.optim.Optimizer,
    epoch_num: int,
) -> None:
    """Step the scheduler with the current metric and log any LR change.

    Args:
        scheduler: The ReduceLROnPlateau scheduler to step.
        metric_value: This epoch's validation metric (e.g. macro-AUC).
        optimizer: The optimizer being scheduled -- used to read the
            learning rate before/after stepping, to detect changes.
        epoch_num: Current epoch number, for logging.
    """
    lr_before = optimizer.param_groups[0]["lr"]
    scheduler.step(metric_value)
    lr_after = optimizer.param_groups[0]["lr"]

    if lr_after != lr_before:
        logger.info(
            "Epoch %d | Learning rate reduced: %.2e -> %.2e",
            epoch_num, lr_before, lr_after,
        )
    else:
        logger.info(
            "Epoch %d | Learning rate unchanged: %.2e",
            epoch_num, lr_after,
        )