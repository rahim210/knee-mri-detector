"""
infer.py

Inference entrypoint: loads a trained checkpoint, runs the model over
an unlabeled test set (KneeMRITestDataset), aggregates slice-level
predictions to study-level predictions, and writes a submission CSV
matching Kaggle's expected format (StudyInstanceUID + 12 finding
probability columns).

Supports optional Test Time Augmentation (--use_tta), which averages
each slice's prediction with a horizontally-flipped view (see
inference/tta.py) for a modest, cost-free accuracy boost.

Usage:
    python infer.py --checkpoint checkpoints/best_model.pt ^
        --test_root data/test --output outputs/submission.csv --use_tta

If the checkpoint predates the backbone_name/hyperparameters fields
(older checkpoints saved before those were added to train.py's
torch.save({...}) call), pass --backbone_name and --dropout_rate
explicitly to override the defaults used as a fallback.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.knee_test_dataset import KneeMRITestDataset
from inference.tta import predict_with_tta
from models.baseline_cnn import KneeMRIClassifier
from metrics.auc_metrics import FINDING_NAMES

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference.

    Returns:
        Namespace with checkpoint, test_root, output, batch_size,
        num_workers, backbone_name, dropout_rate, and use_tta
        attributes.
    """
    parser = argparse.ArgumentParser(
        description="Run inference on the knee MRI test set."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a checkpoint saved by train.py.",
    )
    parser.add_argument(
        "--test_root", type=str, default="data/test",
        help="Root directory of test DICOMs (default: data/test).",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/submission.csv",
        help="Where to write the submission CSV (default: outputs/submission.csv).",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--backbone_name", type=str, default="resnet18",
        help="Backbone architecture, used only if the checkpoint doesn't "
             "store it (older checkpoints). Default: resnet18.",
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.3,
        help="Dropout rate, used only if the checkpoint doesn't store "
             "hyperparameters. Default: 0.3.",
    )
    parser.add_argument(
        "--use_tta", action="store_true",
        help="If set, average predictions with a horizontally-flipped "
             "view of each slice (Test Time Augmentation).",
    )
    return parser.parse_args()


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    fallback_backbone_name: str = "resnet18",
    fallback_dropout_rate: float = 0.3,
) -> torch.nn.Module:
    """Load a trained KneeMRIClassifier from a train.py-style checkpoint.

    Args:
        checkpoint_path: Path to a .pt file saved by train.py's
            torch.save({...}) call.
        device: Device to load the model onto.
        fallback_backbone_name: Backbone to use if the checkpoint
            predates the backbone_name field (older checkpoints).
        fallback_dropout_rate: Dropout rate to use if the checkpoint
            predates the hyperparameters field.

    Returns:
        The reconstructed model, in eval() mode, with trained weights
        loaded and moved to device.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "backbone_name" in checkpoint:
        backbone_name = checkpoint["backbone_name"]
    else:
        backbone_name = fallback_backbone_name
        logger.warning(
            "Checkpoint has no 'backbone_name' key (older checkpoint format) -- "
            "assuming backbone_name='%s'. Pass --backbone_name if this is wrong.",
            backbone_name,
        )

    if "hyperparameters" in checkpoint:
        dropout_rate = checkpoint["hyperparameters"].get("dropout_rate", fallback_dropout_rate)
    else:
        dropout_rate = fallback_dropout_rate
        logger.warning(
            "Checkpoint has no 'hyperparameters' key (older checkpoint format) -- "
            "assuming dropout_rate=%.2f. Pass --dropout_rate if this is wrong.",
            dropout_rate,
        )

    model = KneeMRIClassifier(
        backbone_name=backbone_name,
        pretrained=False,  # weights come from the checkpoint, not ImageNet
        dropout_rate=dropout_rate,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded checkpoint %s | backbone=%s | epoch=%d | macro_auc=%.4f",
        checkpoint_path, backbone_name, checkpoint["epoch"], checkpoint["macro_auc"],
    )
    return model


def run_inference(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_tta: bool = False,
) -> pd.DataFrame:
    """Run the model over every slice in the test set.

    Args:
        model: A trained model in eval() mode.
        test_loader: DataLoader over a KneeMRITestDataset.
        device: Device to run computation on.
        use_tta: If True, average predictions with a horizontally
            flipped view of each slice (see inference/tta.py).

    Returns:
        A DataFrame with one row per slice: study_id plus one
        probability column per finding.
    """
    all_study_ids: list[str] = []
    all_probs: list[np.ndarray] = []

    num_batches = len(test_loader)

    for batch_idx, batch in enumerate(test_loader, start=1):
        images = batch["image"].to(device)
        study_ids = batch["study_id"]

        if use_tta:
            probs = predict_with_tta(model, images, use_flip=True)
        else:
            with torch.no_grad():
                logits = model(images)
                probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_study_ids.extend(study_ids)

        if batch_idx % 20 == 0 or batch_idx == num_batches:
            logger.info("Inference batch %d/%d", batch_idx, num_batches)

    probs_array = np.concatenate(all_probs, axis=0)

    slice_df = pd.DataFrame(probs_array, columns=FINDING_NAMES)
    slice_df.insert(0, "study_id", all_study_ids)
    return slice_df


def aggregate_to_study_level(slice_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate slice-level probabilities to one row per study.

    Uses mean pooling across all slices belonging to a study -- a
    simple, solid baseline aggregation. More sophisticated pooling
    (max, attention-weighted) can replace this later without touching
    the rest of the pipeline.

    The id column is renamed to StudyInstanceUID to match Kaggle's
    expected submission format (see data/sample_submission.csv).

    Args:
        slice_df: Output of run_inference().

    Returns:
        A DataFrame with one row per study, columns
        [StudyInstanceUID, <finding_1>, ..., <finding_12>].
    """
    study_df = slice_df.groupby("study_id")[FINDING_NAMES].mean().reset_index()
    study_df = study_df.rename(columns={"study_id": "StudyInstanceUID"})
    return study_df


def main() -> None:
    """Run the full inference pipeline and write a submission CSV."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = load_model(
        Path(args.checkpoint),
        device,
        fallback_backbone_name=args.backbone_name,
        fallback_dropout_rate=args.dropout_rate,
    )

    test_dataset = KneeMRITestDataset(series_root=Path(args.test_root))
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    logger.info("TTA enabled: %s", args.use_tta)
    slice_df = run_inference(model, test_loader, device, use_tta=args.use_tta)
    study_df = aggregate_to_study_level(slice_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    study_df.to_csv(output_path, index=False)

    logger.info(
        "Inference complete | studies=%d | slices=%d | submission written to %s",
        len(study_df), len(slice_df), output_path,
    )


if __name__ == "__main__":
    main()