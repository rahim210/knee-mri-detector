"""
ensemble_infer.py

CLI entrypoint for evaluating a multi-checkpoint ensemble on the
local validation set. Wraps ensemble.evaluate_ensemble with argument
parsing and reporting via metrics/auc_metrics.py.

Since only one trained checkpoint (checkpoints/best_model.pt) exists
so far, this can currently only be tested by loading that checkpoint
twice -- which validates the mechanism (multi-model loading,
averaging, evaluation) end-to-end, but does NOT yet demonstrate the
real accuracy benefit of ensembling genuinely different models. That
benefit only appears once a second, architecturally different
checkpoint (e.g. efficientnet_b0) exists from a future training run.
"""

import argparse
import logging
from pathlib import Path

import torch

from ensemble import evaluate_ensemble
from datasets.dataloader import build_dataloader
from metrics.auc_metrics import log_auc_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ensemble evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of checkpoints.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=[Path("checkpoints/best_model.pt")],
        help="One or more checkpoint .pt files to ensemble together.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest_val.csv"),
        help="Validation manifest CSV.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/train_with_derived_labels.csv"),
        help="Labels CSV (ground truth + derived).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root folder containing train_series/.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable TTA within each model (plain single-pass prediction).",
    )
    return parser.parse_args()


def main() -> None:
    """Run ensemble evaluation from CLI arguments."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Ensembling %d checkpoint(s): %s", len(args.checkpoints), args.checkpoints)

    val_loader = build_dataloader(
        manifest_path=args.manifest,
        labels_csv_path=args.labels,
        data_root=args.data_root,
        train=False,
        batch_size=16,
        num_workers=4,
    )

    per_finding_auc, macro_auc = evaluate_ensemble(
        checkpoint_paths=args.checkpoints,
        dataloader=val_loader,
        device=device,
        use_tta=not args.no_tta,
    )

    log_auc_report(per_finding_auc, macro_auc)


if __name__ == "__main__":
    main()