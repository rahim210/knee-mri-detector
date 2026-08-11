"""
preprocessing/pseudo_labeling.py

Generate pseudo-labels from a trained model's own high-confidence
predictions, and cross-check them against our existing derived
(NLP-extracted) labels to flag likely-noisy training samples.

Two distinct signal sources now exist for "unlabeled" studies:
  1. label_source == "derived": weak labels from report-text NLP
     extraction (Phase 8-9), independent of the image.
  2. Model predictions (this phase): weak labels from the trained
     image model's own confident outputs, independent of the report.

Where these two AGREE with high confidence, that's reinforcing
evidence the label is correct. Where they DISAGREE, that's a flag
for a likely-noisy sample -- useful for Phase 30 error analysis or
for down-weighting specific samples in a future training run.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets.dataloader import build_dataloader
from ensemble import load_model_from_checkpoint
from inference.tta import predict_with_tta
from metrics.auc_metrics import FINDING_NAMES

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HIGH_CONFIDENCE_THRESHOLD = 0.9
LOW_CONFIDENCE_THRESHOLD = 0.1


def generate_pseudo_labels(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    use_tta: bool = True,
    high_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
    low_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """Run inference and keep only high-confidence predictions as pseudo-labels.

    Args:
        model: A trained model in eval() mode.
        dataloader: DataLoader over the studies to pseudo-label
            (typically the derived-label subset, or a genuinely
            unlabeled subset).
        device: The device to run inference on.
        use_tta: Whether to apply TTA per prediction.
        high_threshold: Predictions above this become pseudo-label 1.0.
        low_threshold: Predictions below this become pseudo-label 0.0.
            Predictions in between are left as NaN (not confident
            enough to use).

    Returns:
        DataFrame with study_id and one pseudo_{finding} column per
        finding, containing 1.0, 0.0, or NaN (not confident).
    """
    rows = []

    for batch in dataloader:
        images = batch["image"].to(device)
        study_ids = batch["study_id"]

        if use_tta:
            probs = predict_with_tta(model, images, use_flip=True)
        else:
            with torch.no_grad():
                probs = torch.sigmoid(model(images))

        probs_np = probs.cpu().numpy()

        for i in range(len(study_ids)):
            row = {"study_id": study_ids[i]}
            for j, name in enumerate(FINDING_NAMES):
                p = probs_np[i, j]
                if p >= high_threshold:
                    row[f"pseudo_{name}"] = 1.0
                elif p <= low_threshold:
                    row[f"pseudo_{name}"] = 0.0
                else:
                    row[f"pseudo_{name}"] = np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_pseudo_labels_by_study(slice_level_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate slice-level pseudo-labels to one row per study (majority vote).

    Args:
        slice_level_df: Output of generate_pseudo_labels, one row per
            slice.

    Returns:
        DataFrame with one row per study_id, each pseudo_{finding}
        column set to the majority (mean-rounded) value across that
        study's confident slices, or NaN if no slices were confident.
    """
    grouped = slice_level_df.groupby("study_id")
    agg_rows = []

    for study_id, group in grouped:
        row = {"study_id": study_id}
        for name in FINDING_NAMES:
            col = f"pseudo_{name}"
            valid_values = group[col].dropna()
            if len(valid_values) == 0:
                row[col] = np.nan
            else:
                row[col] = round(valid_values.mean())
        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


def compare_with_derived_labels(
    pseudo_df: pd.DataFrame, labels_csv_path: Path
) -> pd.DataFrame:
    """Compare study-level pseudo-labels against existing derived labels.

    Args:
        pseudo_df: Output of aggregate_pseudo_labels_by_study.
        labels_csv_path: Path to train_with_derived_labels.csv.

    Returns:
        DataFrame with study_id, finding, derived_label, pseudo_label,
        and agrees (bool) -- one row per (study, finding) pair where
        BOTH a derived label and a confident pseudo-label exist.
    """
    derived_df = pd.read_csv(labels_csv_path)
    derived_df = derived_df.rename(columns={"StudyInstanceUID": "study_id"})

    comparison_rows = []

    merged = pseudo_df.merge(
        derived_df[["study_id", "label_source"] + FINDING_NAMES],
        on="study_id",
        how="inner",
    )

    for _, row in merged.iterrows():
        for name in FINDING_NAMES:
            pseudo_val = row[f"pseudo_{name}"]
            derived_val = row[name]

            if pd.isna(pseudo_val) or pd.isna(derived_val):
                continue

            comparison_rows.append({
                "study_id": row["study_id"],
                "finding": name,
                "label_source": row["label_source"],
                "derived_label": derived_val,
                "pseudo_label": pseudo_val,
                "agrees": derived_val == pseudo_val,
            })

    return pd.DataFrame(comparison_rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate and cross-check pseudo-labels.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest_train.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/train_with_derived_labels.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pseudo_labels"))
    return parser.parse_args()


def main() -> None:
    """Run pseudo-label generation and derived-label cross-checking."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loader = build_dataloader(
        manifest_path=args.manifest,
        labels_csv_path=args.labels,
        data_root=args.data_root,
        train=False,
        batch_size=16,
        num_workers=4,
    )

    model = load_model_from_checkpoint(args.checkpoint, device)

    slice_level = generate_pseudo_labels(model, loader, device, use_tta=True)
    study_level = aggregate_pseudo_labels_by_study(slice_level)
    study_level.to_csv(args.output_dir / "pseudo_labels_by_study.csv", index=False)

    comparison = compare_with_derived_labels(study_level, args.labels)
    comparison.to_csv(args.output_dir / "pseudo_vs_derived_comparison.csv", index=False)

    if len(comparison) > 0:
        agreement_rate = comparison["agrees"].mean()
        logger.info(
            "Pseudo-label vs derived-label agreement rate: %.1f%% (%d comparable labels)",
            agreement_rate * 100, len(comparison),
        )
        disagreements = comparison[~comparison["agrees"]]
        if len(disagreements) > 0:
            logger.info("=== Studies with disagreement (possible noisy derived labels) ===")
            logger.info("\n%s", disagreements.to_string(index=False))
    else:
        logger.warning("No overlapping confident pseudo-labels and derived labels to compare.")

    logger.info("Pseudo-labeling results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()