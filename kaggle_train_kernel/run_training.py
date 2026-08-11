"""
kaggle_train_kernel/run_training.py

Entrypoint for running training on Kaggle's cloud GPU.

Our code dataset (rahim02/rsna-knee-pipeline-code) stores each
subfolder as a separate zip (models.zip, datasets.zip, etc.) because
the Kaggle CLI's --dir-mode zip packs folders individually. This
script extracts them all into the working directory first, so
`import models.baseline_cnn` etc. work exactly as they do locally,
then invokes train.py as a subprocess with paths remapped to
Kaggle's actual mount points.
"""

import logging
import subprocess
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CODE_DATASET_DIR = Path("/kaggle/input/rsna-knee-pipeline-code")
WORKING_DIR = Path("/kaggle/working")
COMPETITION_DATA_DIR = Path("/kaggle/input/rsna-knee-abnormality-detection")


def extract_code_zips() -> None:
    """Extract every subfolder zip from the code dataset into /kaggle/working.

    Also copies loose root-level .py files (train.py, ensemble.py,
    etc.) directly, since those weren't zipped.
    """
    for item in CODE_DATASET_DIR.iterdir():
        if item.suffix == ".zip":
            logger.info("Extracting %s", item.name)
            with zipfile.ZipFile(item, "r") as zf:
                zf.extractall(WORKING_DIR / item.stem)
        elif item.suffix == ".py":
            dest = WORKING_DIR / item.name
            dest.write_bytes(item.read_bytes())
            logger.info("Copied %s", item.name)

    logger.info("Code extraction complete. Contents of %s:", WORKING_DIR)
    for p in sorted(WORKING_DIR.iterdir()):
        logger.info("  %s", p.name)


def run_training(backbone_name: str, num_epochs: int) -> None:
    """Invoke train.py as a subprocess with Kaggle-appropriate arguments.

    Args:
        backbone_name: Which timm backbone to train.
        num_epochs: How many epochs to train for.
    """
    command = [
        sys.executable, str(WORKING_DIR / "train.py"),
        "--backbone_name", backbone_name,
        "--num_epochs", str(num_epochs),
    ]

    logger.info("Running: %s", " ".join(command))
    result = subprocess.run(command, cwd=str(WORKING_DIR))

    if result.returncode != 0:
        logger.error("Training failed with exit code %d", result.returncode)
        sys.exit(result.returncode)

    logger.info("Training complete.")


def main() -> None:
    """Run the full Kaggle training entrypoint."""
    extract_code_zips()
    run_training(backbone_name="resnet18", num_epochs=5)


if __name__ == "__main__":
    main()