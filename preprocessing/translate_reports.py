"""
preprocessing/translate_reports.py

Detect the language of each radiology report and translate non-English
reports to English. Processes and saves incrementally (one row at a
time, appended to a CSV) so the script can be safely interrupted and
resumed without losing progress or re-translating completed rows.

Uses deep-translator's GoogleTranslator backend (free, no API key
needed) and langdetect for language identification.
"""

import csv
import logging
import time
from pathlib import Path

import pandas as pd
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MAX_CHARS_PER_TRANSLATE_CALL = 4500  # GoogleTranslator's free tier has a ~5000 char limit
REQUEST_DELAY_SECONDS = 0.5

INPUT_PATH = Path("data/train.csv")
OUTPUT_PATH = Path("data/train_with_english_reports.csv")


def detect_language(text: str) -> str:
    """Detect the language of a report.

    Args:
        text: Raw report text.

    Returns:
        ISO 639-1 language code (e.g. 'en', 'es', 'de'), or 'unknown'
        if detection fails (e.g. text too short or ambiguous).
    """
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def translate_to_english(text: str, source_lang: str) -> str:
    """Translate a report to English, chunking if it's too long.

    Args:
        text: Report text in its original language.
        source_lang: ISO 639-1 language code.

    Returns:
        English translation of the text.
    """
    if len(text) <= MAX_CHARS_PER_TRANSLATE_CALL:
        return GoogleTranslator(source=source_lang, target="en").translate(text)

    chunks = []
    current_chunk = ""
    for line in text.split("\n"):
        if len(current_chunk) + len(line) + 1 > MAX_CHARS_PER_TRANSLATE_CALL:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk = current_chunk + "\n" + line if current_chunk else line
    if current_chunk:
        chunks.append(current_chunk)

    translated_chunks = []
    for chunk in chunks:
        translated_chunks.append(GoogleTranslator(source=source_lang, target="en").translate(chunk))
        time.sleep(REQUEST_DELAY_SECONDS)

    return "\n".join(translated_chunks)


def get_already_processed_ids(output_path: Path) -> set[str]:
    """Read which StudyInstanceUIDs have already been translated.

    Args:
        output_path: Path to the incremental output CSV.

    Returns:
        Set of StudyInstanceUIDs already present in the output file.
        Empty set if the file doesn't exist yet.
    """
    if not output_path.exists():
        return set()
    existing = pd.read_csv(output_path)
    return set(existing["StudyInstanceUID"].tolist())


def process_and_save_incrementally(df: pd.DataFrame, output_path: Path) -> None:
    """Translate each report and append results to the output CSV one row at a time.

    Args:
        df: DataFrame with StudyInstanceUID and Report columns (plus
            the 12 finding columns, carried through unchanged).
        output_path: Where to append results. Created with a header
            if it doesn't exist yet.
    """
    already_done = get_already_processed_ids(output_path)
    logger.info("%d rows already processed, will skip those", len(already_done))

    file_exists = output_path.exists()
    fieldnames = list(df.columns) + ["detected_language", "report_english"]

    total = len(df)
    processed_this_run = 0

    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            row_dict = row.to_dict()

            if row_dict["StudyInstanceUID"] in already_done:
                continue

            report_text = str(row_dict["Report"])
            lang = detect_language(report_text)

            if lang == "en" or lang == "unknown":
                translated = report_text
            else:
                try:
                    translated = translate_to_english(report_text, source_lang=lang)
                except Exception as exc:
                    logger.warning(
                        "Translation failed for %s (lang=%s): %s",
                        row_dict["StudyInstanceUID"], lang, exc,
                    )
                    translated = report_text

            row_dict["detected_language"] = lang
            row_dict["report_english"] = translated
            writer.writerow(row_dict)
            f.flush()  # ensure it's written to disk immediately, not buffered

            processed_this_run += 1
            time.sleep(REQUEST_DELAY_SECONDS)

            if i % 20 == 0 or i == total:
                logger.info("[%d/%d] processed (this run: %d)", i, total, processed_this_run)

    logger.info("Done. Processed %d new rows this run.", processed_this_run)


def main() -> None:
    """Run translation on the full train.csv, resuming if interrupted."""
    df = pd.read_csv(INPUT_PATH)
    logger.info("Loaded %d total reports", len(df))

    process_and_save_incrementally(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()