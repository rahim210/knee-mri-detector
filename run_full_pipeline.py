"""
run_full_pipeline.py

Phase 32 -- Final Competition Pipeline.

Single orchestration entrypoint chaining the entire project together:
  1. Train a model (train.py)
  2. Run error analysis on the resulting checkpoint (analysis/error_analysis.py)
  3. Generate pseudo-labels and cross-check against derived labels (preprocessing/pseudo_labeling.py)
  4. Run inference on the test set and write a submission (infer.py)

Each stage is a subprocess call to the existing, already-tested
script for that stage -- this file adds no new modeling logic, only
sequencing, logging, and stop-on-failure behavior, so a broken stage
doesn't silently continue into the next one with bad inputs.

Steps 2-3 are diagnostic/optional and can be skipped with flags if
you just want a fast train -> submit run.

Usage:
    python run_full_pipeline.py --backbone_name resnet18 --num_epochs 5
    python run_full_pipeline.py --skip_analysis --skip_pseudo_labeling
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_stage(name: str, command: list[str]) -> None:
    """Run one pipeline stage as a subprocess, streaming its output live.

    Args:
        name: Human-readable stage name, for logging.
        command: The full command (as a list of args) to run.

    Raises:
        SystemExit: If the subprocess exits with a non-zero code --
            stops the pipeline immediately rather than continuing
            into a later stage with a broken/missing prerequisite.
    """
    logger.info("=" * 70)
    logger.info("STAGE: %s", name)
    logger.info("Command: %s", " ".join(command))
    logger.info("=" * 70)

    result = subprocess.run(command)

    if result.returncode != 0:
        logger.error("Stage '%s' failed with exit code %d. Stopping pipeline.", name, result.returncode)
        sys.exit(result.returncode)

    logger.info("Stage '%s' complete.\n", name)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full pipeline run."""
    parser = argparse.ArgumentParser(description="Run the full knee MRI competition pipeline end-to-end.")

    parser.add_argument("--backbone_name", type=str, default="resnet18")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to an existing checkpoint to skip training and use directly. "
             "If not set, training runs first and the resulting "
             "checkpoints/best_model_{backbone_name}.pt is used.",
    )

    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_analysis", action="store_true")
    parser.add_argument("--skip_pseudo_labeling", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")

    parser.add_argument("--test_root", type=str, default="data/test")
    parser.add_argument("--output", type=str, default="outputs/submission.csv")
    parser.add_argument("--use_tta", action="store_true", default=True)

    return parser.parse_args()


def main() -> None:
    """Run the full pipeline, stage by stage."""
    args = parse_args()

    checkpoint_path = args.checkpoint or Path(f"checkpoints/best_model_{args.backbone_name}.pt")

    if not args.skip_training:
        run_stage(
            "Training",
            [
                sys.executable, "train.py",
                "--backbone_name", args.backbone_name,
                "--num_epochs", str(args.num_epochs),
                "--learning_rate", str(args.learning_rate),
                "--dropout_rate", str(args.dropout_rate),
                "--weight_decay", str(args.weight_decay),
            ],
        )
    else:
        logger.info("Skipping training stage (--skip_training set). Using checkpoint: %s", checkpoint_path)

    if not checkpoint_path.exists():
        logger.error(
            "Expected checkpoint not found at %s. Cannot continue -- "
            "either training failed to produce it, or --checkpoint points "
            "to the wrong path.",
            checkpoint_path,
        )
        sys.exit(1)

    if not args.skip_analysis:
        run_stage(
            "Error Analysis",
            [
                sys.executable, "-m", "analysis.error_analysis",
                "--checkpoint", str(checkpoint_path),
            ],
        )

    if not args.skip_pseudo_labeling:
        run_stage(
            "Pseudo-Labeling",
            [
                sys.executable, "-m", "preprocessing.pseudo_labeling",
                "--checkpoint", str(checkpoint_path),
            ],
        )

    if not args.skip_inference:
        infer_command = [
            sys.executable, "infer.py",
            "--checkpoint", str(checkpoint_path),
            "--test_root", args.test_root,
            "--output", args.output,
            "--backbone_name", args.backbone_name,
            "--dropout_rate", str(args.dropout_rate),
        ]
        if args.use_tta:
            infer_command.append("--use_tta")

        run_stage("Inference + Submission Generation", infer_command)

    logger.info("=" * 70)
    logger.info("FULL PIPELINE COMPLETE")
    logger.info("Checkpoint: %s", checkpoint_path)
    if not args.skip_inference:
        logger.info("Submission: %s", args.output)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()