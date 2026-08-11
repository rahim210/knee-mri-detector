"""
train.py

Main training entrypoint for the knee MRI abnormality classifier.

Builds the train/val dataloaders, the baseline CNN model, and a
class-weighted focal loss (using real per-finding pos_weights
computed from the training labels), then runs the full training +
validation loop for NUM_EPOCHS, saving the best checkpoint (by
macro-AUC) to checkpoints/.
"""

import logging
from pathlib import Path

import torch

from datasets.dataloader import build_dataloader
from models.baseline_cnn import BaselineKneeCNN
from losses.focal_loss import FocalLoss, compute_pos_weights_from_labels
from training.train_loop import train_one_epoch, validate_with_metrics
from training.schedulers import build_scheduler, step_scheduler_and_log
from metrics.auc_metrics import FINDING_NAMES, log_auc_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

NUM_EPOCHS = 3
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"


def main() -> None:
    """Run full training + validation and save the best checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(
        manifest_path=Path("data/manifest_train.csv"),
        labels_csv_path=Path("data/train_with_derived_labels.csv"),
        data_root=Path("data"),
        train=True,
        batch_size=16,
        num_workers=4,
    )

    val_loader = build_dataloader(
        manifest_path=Path("data/manifest_val.csv"),
        labels_csv_path=Path("data/train_with_derived_labels.csv"),
        data_root=Path("data"),
        train=False,
        batch_size=16,
        num_workers=4,
    )

    model = BaselineKneeCNN(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    pos_weights = compute_pos_weights_from_labels(
        labels_csv_path="data/train_with_derived_labels.csv",
        finding_names=FINDING_NAMES,
    ).to(device)
    criterion = FocalLoss(gamma=2.0, pos_weight=pos_weights)

    scheduler = build_scheduler(optimizer, mode="max", factor=0.5, patience=2)

    best_macro_auc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch_num=epoch,
        )
        val_loss, per_finding_auc, macro_auc = validate_with_metrics(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch_num=epoch,
        )
        logger.info(
            "=== Epoch %d summary | train_loss=%.4f | val_loss=%.4f | macro_auc=%.4f ===",
            epoch, train_loss, val_loss, macro_auc,
        )
        log_auc_report(per_finding_auc, macro_auc)

        step_scheduler_and_log(scheduler, macro_auc, optimizer, epoch)

        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "macro_auc": macro_auc,
                },
                CHECKPOINT_PATH,
            )
            logger.info(
                "New best macro_auc=%.4f -- checkpoint saved to %s",
                macro_auc, CHECKPOINT_PATH,
            )

    logger.info("Training complete. Best macro_auc=%.4f", best_macro_auc)


if __name__ == "__main__":
    main()