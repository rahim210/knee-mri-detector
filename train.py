"""
train.py

Main training entrypoint for the knee MRI abnormality classifier.

Builds the train/val dataloaders, a configurable backbone model
(via KneeMRIClassifier), and a class-weighted focal loss (using real
per-finding pos_weights computed from the training labels), then runs
the full training + validation loop, saving the best checkpoint (by
macro-AUC) to checkpoints/.

Key hyperparameters (backbone_name, dropout_rate, weight_decay,
learning_rate, num_epochs) are exposed as command-line arguments so
experiments -- including training multiple different backbones for
later ensembling (Phase 26) -- can be run without editing this file,
e.g.:

    python train.py --backbone_name efficientnet_b0 --num_epochs 3
"""
import sys
sys.path.insert(0, "/kaggle/working")
import argparse
import logging
from pathlib import Path

import torch

from datasets.dataloader import build_dataloader
from models.baseline_cnn import KneeMRIClassifier
from losses.focal_loss import FocalLoss, compute_pos_weights_from_labels
from training.train_loop import train_one_epoch, validate_with_metrics
from training.schedulers import build_scheduler, step_scheduler_and_log
from metrics.auc_metrics import FINDING_NAMES, log_auc_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")


def parse_args() -> argparse.Namespace:
    """Parse command-line hyperparameter overrides.

    Returns:
        Namespace with backbone_name, num_epochs, learning_rate,
        dropout_rate, and weight_decay attributes.
    """
    parser = argparse.ArgumentParser(
        description="Train the knee MRI abnormality classifier."
    )
    parser.add_argument(
        "--backbone_name", type=str, default="resnet18",
        help="timm backbone to use, e.g. resnet18, efficientnet_b0 (default: resnet18).",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Adam optimizer learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.3,
        help="Dropout probability before the classification head (default: 0.3).",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Adam optimizer weight decay / L2 regularization (default: 1e-4).",
    )
    return parser.parse_args()


def main() -> None:
    """Run full training + validation and save the best checkpoint."""
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info(
        "Hyperparameters | backbone_name=%s | num_epochs=%d | learning_rate=%.2e | "
        "dropout_rate=%.2f | weight_decay=%.2e",
        args.backbone_name, args.num_epochs, args.learning_rate,
        args.dropout_rate, args.weight_decay,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"best_model_{args.backbone_name}.pt"

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

    model = KneeMRIClassifier(
        backbone_name=args.backbone_name,
        pretrained=True,
        dropout_rate=args.dropout_rate,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    pos_weights = compute_pos_weights_from_labels(
        labels_csv_path="data/train_with_derived_labels.csv",
        finding_names=FINDING_NAMES,
    ).to(device)
    criterion = FocalLoss(gamma=2.0, pos_weight=pos_weights)

    scheduler = build_scheduler(optimizer, mode="max", factor=0.5, patience=2)

    best_macro_auc = 0.0

    for epoch in range(1, args.num_epochs + 1):
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
                    "backbone_name": args.backbone_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "macro_auc": macro_auc,
                    "hyperparameters": vars(args),
                },
                checkpoint_path,
            )
            logger.info(
                "New best macro_auc=%.4f -- checkpoint saved to %s",
                macro_auc, checkpoint_path,
            )

    logger.info("Training complete. Best macro_auc=%.4f", best_macro_auc)


if __name__ == "__main__":
    main()