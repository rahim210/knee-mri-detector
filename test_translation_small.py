"""
test_translation_small.py

Quick validation: translate only the 58 labeled reports first,
before committing to the full 4,407-report translation run.
"""

import logging
from pathlib import Path

import pandas as pd

from preprocessing.translate_reports import process_reports

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

finding_cols = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

df = pd.read_csv("data/train.csv")
labeled = df[df[finding_cols].notna().any(axis=1)].reset_index(drop=True)

logger.info("Testing translation on %d labeled reports", len(labeled))

result = process_reports(labeled)
result.to_csv("data/labeled_reports_translated_test.csv", index=False)

logger.info("Done. Language distribution:")
logger.info("\n%s", result["detected_language"].value_counts())

# Show a couple of non-English examples so we can eyeball translation quality
non_english = result[result["detected_language"] != "en"]
for i in range(min(3, len(non_english))):
    row = non_english.iloc[i]
    print(f"\n--- {row['detected_language']} example ---")
    print("ORIGINAL:", str(row["Report"])[:300])
    print("TRANSLATED:", str(row["report_english"])[:300])