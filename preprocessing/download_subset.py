"""
preprocessing/download_subset.py

Download a local subset of RSNA Knee Abnormality Detection training
DICOM files using exact file paths from data/subset_manifest.csv
(built by find_train_files.py). Downloading exact paths -- rather
than folder paths -- is required; the Kaggle API only supports
single-file downloads, not directory downloads.

Includes rate-limit handling: a delay between requests, plus
automatic retry with capped exponential backoff on HTTP 429s.
"""

import logging
import time
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from requests.exceptions import HTTPError

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COMPETITION = "rsna-knee-abnormality-detection"
REQUEST_DELAY_SECONDS = 2.0
MAX_RETRIES = 6
COOLDOWN_AFTER_RATE_LIMIT_SECONDS = 30


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load the file manifest produced by find_train_files.py.

    Args:
        manifest_path: Path to subset_manifest.csv.

    Returns:
        DataFrame with columns: study_id, series_id, filename, path, size.

    Raises:
        FileNotFoundError: If the manifest doesn't exist yet.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            f"Run 'python -m preprocessing.find_train_files' first."
        )
    df = pd.read_csv(manifest_path)
    logger.info(
        "Loaded manifest: %d files, %d studies, %d series",
        len(df), df["study_id"].nunique(), df["series_id"].nunique(),
    )
    return df


def download_one_file_with_retry(
    api: KaggleApi, remote_path: str, staging_dir: Path
) -> None:
    """Download a single file, retrying with capped backoff on rate limits.

    Args:
        api: An authenticated KaggleApi instance.
        remote_path: Exact remote file path on Kaggle.
        staging_dir: Local folder to download the raw file into.

    Raises:
        HTTPError: If all retries are exhausted without success.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            api.competition_download_file(
                competition=COMPETITION,
                file_name=remote_path,
                path=str(staging_dir),
                force=False,
                quiet=True,
            )
            return
        except HTTPError as exc:
            is_rate_limit = "429" in str(exc)
            if is_rate_limit and attempt < MAX_RETRIES:
                wait_time = min(2 ** attempt, COOLDOWN_AFTER_RATE_LIMIT_SECONDS)
                logger.warning(
                    "Rate limited (attempt %d/%d). Waiting %ds before retry...",
                    attempt, MAX_RETRIES, wait_time,
                )
                time.sleep(wait_time)
                continue
            raise


def download_manifest_files(
    df: pd.DataFrame, output_dir: Path, api: KaggleApi
) -> None:
    """Download every file listed in the manifest into correct nested folders.

    The Kaggle API's competition_download_file() always saves files
    flat into the given path, ignoring the remote folder structure.
    So we download into a temporary staging area using the exact
    remote path, then move each file into the correct local nested
    path (data/train_series/{study}/{series}/{filename}) ourselves.

    Skips files that already exist at their correct final location,
    so this function is safe to re-run if interrupted partway through.
    A delay is added between requests to avoid rate limiting.

    Args:
        df: Manifest DataFrame with study_id, series_id, filename,
            and path columns.
        output_dir: Local root folder to download into (e.g. data/).
        api: An authenticated KaggleApi instance.
    """
    staging_dir = output_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    total = len(df)
    skipped = 0
    downloaded = 0

    for i, row in enumerate(df.itertuples(), start=1):
        final_path = output_dir / "train_series" / row.study_id / row.series_id / row.filename

        if final_path.exists():
            skipped += 1
            continue

        final_path.parent.mkdir(parents=True, exist_ok=True)

        download_one_file_with_retry(api, row.path, staging_dir)

        staged_file = staging_dir / row.filename
        if not staged_file.exists():
            raise FileNotFoundError(
                f"Expected downloaded file not found at {staged_file}. "
                f"Kaggle API's flat-download behavior may have changed."
            )

        staged_file.rename(final_path)
        downloaded += 1

        time.sleep(REQUEST_DELAY_SECONDS)

        if i % 100 == 0 or i == total:
            logger.info(
                "[%d/%d] downloaded=%d skipped=%d", i, total, downloaded, skipped
            )

    staging_dir.rmdir()
    logger.info(
        "Done. Downloaded %d new files, skipped %d already-present files.",
        downloaded, skipped,
    )


def main() -> None:
    """Run the full subset download."""
    api = KaggleApi()
    api.authenticate()

    df = load_manifest(Path("data/subset_manifest.csv"))
    download_manifest_files(df, output_dir=Path("data"), api=api)


if __name__ == "__main__":
    main()