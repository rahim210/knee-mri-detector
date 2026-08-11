"""
training/hyperparameter_search.py

Grid/random search over key hyperparameters (learning_rate,
dropout_rate, weight_decay, backbone_name), by invoking train.py as
a subprocess for each combination and parsing its final macro_auc
from stdout. Results are logged to a CSV for comparison.

NOTE: train.py saves checkpoints as best_model_{backbone_name}.pt --
runs that share a backbone_name will overwrite each other's
checkpoint file. This script only tracks *scores* per combination
(via the results CSV), not a saved checkpoint per combination. Once
you've identified the best hyperparameters here, rerun train.py once
directly with those settings to produce and keep that checkpoint.
"""

import csv
import itertools
import logging
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = Path("outputs/hyperparameter_search_results.csv")

MACRO_AUC_PATTERN = re.compile(r"Training complete\. Best macro_auc=([\d.]+)")


def build_search_space() -> list[dict]:
    """Define the grid of hyperparameter combinations to try.

    Returns:
        List of dicts, each a full set of CLI-argument values for
        one train.py run.
    """
    grid = {
        "backbone_name": ["resnet18"],  # keep fixed here; vary separately for architecture comparisons
        "learning_rate": [1e-3, 1e-4, 1e-5],
        "dropout_rate": [0.2, 0.3, 0.5],
        "weight_decay": [1e-4, 1e-5],
    }

    keys = list(grid.keys())
    combinations = list(itertools.product(*grid.values()))

    return [dict(zip(keys, combo)) for combo in combinations]


def run_single_trial(hyperparams: dict, num_epochs: int) -> float | None:
    """Run train.py as a subprocess with given hyperparameters and parse its score.

    Args:
        hyperparams: Dict with backbone_name, learning_rate,
            dropout_rate, weight_decay.
        num_epochs: How many epochs to train for this trial (kept
            low for search speed; the final chosen config should be
            retrained with more epochs).

    Returns:
        The parsed best macro_auc from this run, or None if the run
        failed or the score couldn't be parsed from output.
    """
    command = [
        sys.executable, "train.py",
        "--backbone_name", str(hyperparams["backbone_name"]),
        "--learning_rate", str(hyperparams["learning_rate"]),
        "--dropout_rate", str(hyperparams["dropout_rate"]),
        "--weight_decay", str(hyperparams["weight_decay"]),
        "--num_epochs", str(num_epochs),
    ]

    logger.info("Running trial: %s", hyperparams)

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Trial failed (exit code %d):\n%s", result.returncode, result.stderr[-2000:])
        return None

    match = MACRO_AUC_PATTERN.search(result.stdout)
    if not match:
        logger.warning("Could not parse macro_auc from trial output.")
        return None

    return float(match.group(1))


def save_result(hyperparams: dict, macro_auc: float | None, results_path: Path) -> None:
    """Append one trial's result to the results CSV.

    Args:
        hyperparams: The hyperparameter combination tried.
        macro_auc: The resulting score, or None if the trial failed.
        results_path: Path to the results CSV (created with a header
            if it doesn't exist yet).
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = results_path.exists()

    fieldnames = list(hyperparams.keys()) + ["macro_auc"]

    with results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = dict(hyperparams)
        row["macro_auc"] = macro_auc if macro_auc is not None else "FAILED"
        writer.writerow(row)


def main(num_epochs_per_trial: int = 2) -> None:
    """Run the full hyperparameter search and report the best combination."""
    search_space = build_search_space()
    logger.info("Starting hyperparameter search: %d combinations to try", len(search_space))

    best_score = -1.0
    best_config = None

    for i, hyperparams in enumerate(search_space, start=1):
        logger.info("--- Trial %d/%d ---", i, len(search_space))
        macro_auc = run_single_trial(hyperparams, num_epochs=num_epochs_per_trial)
        save_result(hyperparams, macro_auc, RESULTS_PATH)

        if macro_auc is not None and macro_auc > best_score:
            best_score = macro_auc
            best_config = hyperparams

    logger.info("Search complete. Results saved to %s", RESULTS_PATH)
    if best_config:
        logger.info("Best config: %s | macro_auc=%.4f", best_config, best_score)
    else:
        logger.warning("No successful trials completed.")


if __name__ == "__main__":
    main()