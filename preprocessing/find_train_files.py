"""
preprocessing/find_train_files.py

Page through the competition's file listing to find exact DICOM
file paths for a chosen subset of training studies, without
scanning the full 820k-file catalog. Stops early once all target
studies are found (alphabetical layout means train_series/ starts
early, right after the small root-level CSVs and test_series/).
"""

import csv
import logging
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COMPETITION = "rsna-knee-abnormality-detection"
PAGE_SIZE = 100  # max allowed may be higher; 100 is a safe starting point


def get_target_study_ids(train_csv_path: Path, n_studies: int) -> set[str]:
    """Load the StudyInstanceUIDs we want files for.

    Args:
        train_csv_path: Path to train.csv.
        n_studies: How many studies to select from the top of the file.

    Returns:
        Set of StudyInstanceUID strings for fast membership checks.
    """
    df = pd.read_csv(train_csv_path)
    study_ids = set(df["StudyInstanceUID"].head(n_studies).tolist())
    logger.info("Targeting %d studies", len(study_ids))
    return study_ids


def find_files_for_studies(
    api: KaggleApi, target_study_ids: set[str]
) -> list[dict]:
    """Page through the competition file listing, collecting matches.

    Stops as soon as every target study has at least one file found
    AND we've moved past the train_series/ prefix range (to avoid
    stopping too early if a study's files happen to be split across
    a page boundary).

    Args:
        api: An authenticated KaggleApi instance.
        target_study_ids: StudyInstanceUIDs to collect files for.

    Returns:
        List of dicts with keys: study_id, series_id, filename, path, size.
    """
    found_rows: list[dict] = []
    studies_seen: set[str] = set()
    page_token: str | None = None
    page_num = 0
    entered_train_series = False

    while True:
        page_num += 1
        response = api.competition_list_files(
            COMPETITION, page_token=page_token, page_size=PAGE_SIZE
        )

        for f in response.files:
            name = str(f.name)

            if name.startswith("train_series/"):
                entered_train_series = True
                parts = name.split("/")
                if len(parts) == 4:  # train_series/{study}/{series}/{file}.dcm
                    study_id = parts[1]
                    if study_id in target_study_ids:
                        found_rows.append(
                            {
                                "study_id": study_id,
                                "series_id": parts[2],
                                "filename": parts[3],
                                "path": name,
                                "size": f.total_bytes,
                            }
                        )
                        studies_seen.add(study_id)
            elif entered_train_series and not name.startswith("train_series/"):
                # We've moved past the train_series/ block entirely
                logger.info("Passed train_series/ range at page %d", page_num)
                return found_rows

        logger.info(
            "Page %d | total matches so far: %d | studies found: %d/%d",
            page_num, len(found_rows), len(studies_seen), len(target_study_ids)
        )

        if entered_train_series and studies_seen == target_study_ids:
            logger.info("Found files for all target studies. Stopping early.")
            return found_rows

        if not response.next_page_token:
            logger.warning("Reached end of file listing without finding all studies.")
            return found_rows

        page_token = response.next_page_token


def save_manifest(rows: list[dict], output_path: Path) -> None:
    """Save the discovered file list to a CSV manifest.

    Args:
        rows: List of file-info dicts from find_files_for_studies.
        output_path: Where to write the manifest CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["study_id", "series_id", "filename", "path", "size"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved manifest with %d files to %s", len(rows), output_path)


def main(n_studies: int = 25) -> None:
    """Run the full file-discovery workflow."""
    api = KaggleApi()
    api.authenticate()

    target_study_ids = get_target_study_ids(Path("data/train.csv"), n_studies)
    rows = find_files_for_studies(api, target_study_ids)
    save_manifest(rows, Path("data/subset_manifest.csv"))


if __name__ == "__main__":
    main(n_studies=25)