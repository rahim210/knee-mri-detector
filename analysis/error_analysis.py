"""
analysis/error_analysis.py

Breaks down model prediction errors on the validation set beyond a
single macro-AUC number: per-finding error rates, worst-performing
individual studies, and whether errors correlate with label_source
(ground_truth vs. derived) -- which would signal that noise from our
NLP label extraction (Phase 8-9 fallback) is hurting the model, as
opposed to the model itself being weak.
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
from metrics.auc_metrics import FINDING_NAMES, compute_per_finding_auc, compute_macro_auc

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def collect_predictions(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    use_tta: bool = True,
) -> pd.DataFrame:
    """Run inference on every batch and collect per-slice predictions with metadata.

    Args:
        model: A trained model in eval() mode.
        dataloader: A validation DataLoader (train=False, shuffle=False).
        device: The device to run inference on.
        use_tta: Whether to apply TTA per prediction.

    Returns:
        DataFrame with one row per slice: study_id, true label columns
        (prefixed "true_"), predicted probability columns (prefixed
        "pred_"), and per-slice absolute error columns (prefixed "err_").
    """
    rows = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        study_ids = batch["study_id"]

        if use_tta:
            probs = predict_with_tta(model, images, use_flip=True)
        else:
            with torch.no_grad():
                probs = torch.sigmoid(model(images))

        labels_np = labels.cpu().numpy()
        probs_np = probs.cpu().numpy()

        for i in range(len(study_ids)):
            row = {"study_id": study_ids[i]}
            for j, name in enumerate(FINDING_NAMES):
                row[f"true_{name}"] = labels_np[i, j]
                row[f"pred_{name}"] = probs_np[i, j]
                row[f"err_{name}"] = abs(labels_np[i, j] - probs_np[i, j])
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_by_finding(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean absolute error per finding, sorted worst-first.

    Args:
        predictions_df: Output of collect_predictions.

    Returns:
        DataFrame with columns: finding, mean_abs_error.
    """
    rows = []
    for name in FINDING_NAMES:
        mean_err = predictions_df[f"err_{name}"].mean()
        rows.append({"finding": name, "mean_abs_error": mean_err})

    result = pd.DataFrame(rows).sort_values("mean_abs_error", ascending=False)
    return result.reset_index(drop=True)


def find_worst_studies(predictions_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Identify the studies with the highest average prediction error.

    Args:
        predictions_df: Output of collect_predictions.
        top_n: How many worst studies to return.

    Returns:
        DataFrame with study_id and mean_abs_error, worst-first,
        aggregated across all slices and findings for that study.
    """
    err_cols = [f"err_{name}" for name in FINDING_NAMES]
    predictions_df = predictions_df.copy()
    predictions_df["overall_error"] = predictions_df[err_cols].mean(axis=1)

    per_study = (
        predictions_df.groupby("study_id")["overall_error"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    return per_study


def summarize_by_label_source(
    predictions_df: pd.DataFrame, labels_csv_path: Path
) -> pd.DataFrame:
    """Compare mean error between ground-truth-labeled and derived-labeled studies.

    This is the key diagnostic for whether our NLP label extraction
    (Phase 8-9) is introducing meaningful noise into training/eval,
    versus the model simply being weak across the board.

    Args:
        predictions_df: Output of collect_predictions.
        labels_csv_path: Path to train_with_derived_labels.csv, which
            has a label_source column ('ground_truth' or 'derived').

    Returns:
        DataFrame with label_source and mean_abs_error, one row per
        source category.
    """
    source_lookup = pd.read_csv(labels_csv_path)[["StudyInstanceUID", "label_source"]]
    source_lookup = source_lookup.rename(columns={"StudyInstanceUID": "study_id"})

    merged = predictions_df.merge(source_lookup, on="study_id", how="left")

    err_cols = [f"err_{name}" for name in FINDING_NAMES]
    merged["overall_error"] = merged[err_cols].mean(axis=1)

    summary = merged.groupby("label_source")["overall_error"].agg(["mean", "count"])
    summary = summary.rename(columns={"mean": "mean_abs_error", "count": "num_slices"})
    return summary.reset_index()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run error analysis on a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest_val.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/train_with_derived_labels.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/error_analysis"))
    return parser.parse_args()


def main() -> None:
    """Run full error analysis and save results to CSVs."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val_loader = build_dataloader(
        manifest_path=args.manifest,
        labels_csv_path=args.labels,
        data_root=args.data_root,
        train=False,
        batch_size=16,
        num_workers=4,
    )

    model = load_model_from_checkpoint(args.checkpoint, device)

    predictions_df = collect_predictions(model, val_loader, device, use_tta=True)
    predictions_df.to_csv(args.output_dir / "raw_predictions.csv", index=False)

    finding_summary = summarize_by_finding(predictions_df)
    finding_summary.to_csv(args.output_dir / "error_by_finding.csv", index=False)
    logger.info("=== Mean Absolute Error by Finding (worst first) ===")
    logger.info("\n%s", finding_summary.to_string(index=False))

    worst_studies = find_worst_studies(predictions_df, top_n=10)
    worst_studies.to_csv(args.output_dir / "worst_studies.csv", index=False)
    logger.info("=== Worst 10 Studies ===")
    logger.info("\n%s", worst_studies.to_string(index=False))

    source_summary = summarize_by_label_source(predictions_df, args.labels)
    source_summary.to_csv(args.output_dir / "error_by_label_source.csv", index=False)
    logger.info("=== Error by Label Source (ground_truth vs derived) ===")
    logger.info("\n%s", source_summary.to_string(index=False))

    logger.info("Full error analysis saved to %s", args.output_dir)


if __name__ == "__main__":
    main()