"""
kaggle_train_kernel/setup_and_train.py

Full Kaggle-side setup: extracts our code dataset, builds the full
translated + labeled dataset from ALL competition studies (not just
our local 15-study subset), builds train/val manifests by walking
the mounted competition data directly, then runs train.py.

Data is never copied -- data/train_series/ is a symlink to the
competition's mounted DICOM folder, so train.py's existing hardcoded
"data/..." paths work completely unmodified.
"""

import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CODE_DATASET_DIR = Path("/kaggle/input/datasets/rahim02/rsna-knee-pipeline-code")
WORKING_DIR = Path("/kaggle/working")
COMPETITION_DATA_DIR = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
DATA_DIR = WORKING_DIR / "data"

def install_dependencies() -> None:
    """Install packages not preinstalled on Kaggle's base image."""
    packages = ["timm", "deep-translator", "langdetect", "pydicom"]
    logger.info("Installing: %s", packages)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=True)


def extract_code_zips() -> None:
    """Extract zips, copy loose .py files, and copy real subfolders from the code dataset."""
    import shutil

    skip_folders = {"kaggle_dataset", "kaggle_kernel", "kaggle_kernel_check", "kaggle_output", "logs"}

    for item in CODE_DATASET_DIR.iterdir():
        if item.suffix == ".zip":
            if item.stem in skip_folders:
                continue
            logger.info("Extracting %s", item.name)
            with zipfile.ZipFile(item, "r") as zf:
                zf.extractall(WORKING_DIR / item.stem)
        elif item.suffix == ".py":
            (WORKING_DIR / item.name).write_bytes(item.read_bytes())
        elif item.is_dir():
            if item.name in skip_folders:
                continue
            dest = WORKING_DIR / item.name
            logger.info("Copying folder %s -> %s", item.name, dest)
            shutil.copytree(item, dest, dirs_exist_ok=True)

    logger.info("Code ready in %s", WORKING_DIR)

def setup_data_symlink() -> None:
    """Symlink data/train_series/ to the competition's mounted DICOM folder."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    link_path = DATA_DIR / "train_series"

    if link_path.exists() or link_path.is_symlink():
        logger.info("Symlink already exists at %s", link_path)
        return

    os.symlink(COMPETITION_DATA_DIR / "train_series", link_path, target_is_directory=True)
    logger.info("Symlinked %s -> %s", link_path, COMPETITION_DATA_DIR / "train_series")


def copy_train_csv() -> None:
    """Copy train.csv from the competition mount into our working data dir."""
    src = COMPETITION_DATA_DIR / "train.csv"
    dst = DATA_DIR / "train.csv"
    dst.write_bytes(src.read_bytes())
    logger.info("Copied train.csv (%d bytes)", dst.stat().st_size)


def run_translation() -> None:
    """Translate all reports to English, reusing preprocessing/translate_reports.py logic."""
    sys.path.insert(0, str(WORKING_DIR))
    from preprocessing.translate_reports import process_and_save_incrementally
    import pandas as pd

    output_path = DATA_DIR / "train_with_english_reports.csv"
    if output_path.exists():
        logger.info("Translation output already exists, skipping (delete to redo).")
        return

    df = pd.read_csv(DATA_DIR / "train.csv")
    logger.info("Translating %d reports...", len(df))
    process_and_save_incrementally(df, output_path)


def run_label_generation() -> None:
    """Generate derived labels for all report-only studies."""
    from preprocessing.generate_derived_labels import generate_labels
    import pandas as pd

    output_path = DATA_DIR / "train_with_derived_labels.csv"
    if output_path.exists():
        logger.info("Derived labels already exist, skipping.")
        return

    df = pd.read_csv(DATA_DIR / "train_with_english_reports.csv")
    result_df = generate_labels(df)
    result_df.to_csv(output_path, index=False)
    logger.info("Saved derived labels: %d studies", len(result_df))


def build_full_manifest() -> None:
    """Walk the mounted train_series/ folder directly to build a slice-level manifest.

    Unlike our local download-manifest approach (which needed the
    Kaggle API to discover file paths), on Kaggle the data is already
    a normal mounted filesystem -- we can just walk it directly with
    zero API calls and zero rate limits.
    """
    import pandas as pd

    output_path = DATA_DIR / "full_manifest.csv"
    if output_path.exists():
        logger.info("Full manifest already exists, skipping.")
        return

    train_series_dir = DATA_DIR / "train_series"
    rows = []

    logger.info("Walking %s ...", train_series_dir)
    for study_dir in train_series_dir.iterdir():
        if not study_dir.is_dir():
            continue
        for series_dir in study_dir.iterdir():
            if not series_dir.is_dir():
                continue
            for dcm_file in series_dir.glob("*.dcm"):
                rows.append({
                    "study_id": study_dir.name,
                    "series_id": series_dir.name,
                    "filename": dcm_file.name,
                    "path": f"train_series/{study_dir.name}/{series_dir.name}/{dcm_file.name}",
                    "size": dcm_file.stat().st_size,
                })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_path, index=False)
    logger.info("Built full manifest: %d files, %d studies", len(manifest), manifest["study_id"].nunique())


def build_train_val_split() -> None:
    """Split the full manifest into train/val at the study level."""
    from preprocessing.split_train_val import split_manifest_by_study

    train_path = DATA_DIR / "manifest_train.csv"
    val_path = DATA_DIR / "manifest_val.csv"
    if train_path.exists() and val_path.exists():
        logger.info("Train/val manifests already exist, skipping.")
        return

    train_manifest, val_manifest = split_manifest_by_study(DATA_DIR / "full_manifest.csv")
    train_manifest.to_csv(train_path, index=False)
    val_manifest.to_csv(val_path, index=False)


def run_training(backbone_name: str, num_epochs: int) -> None:
    """Run train.py -- all paths now resolve exactly as they do locally."""
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
    """Run the full Kaggle setup + training pipeline."""
    install_dependencies()
    extract_code_zips()
    setup_data_symlink()
    copy_train_csv()
    run_translation()
    run_label_generation()
    build_full_manifest()
    build_train_val_split()
    run_training(backbone_name="resnet18", num_epochs=1)


if __name__ == "__main__":
    main()