import pandas as pd

from preprocessing.extract_labels import FINDING_COLUMNS, extract_labels_from_report

df = pd.read_csv("data/train_with_english_reports.csv")
labeled = df[df[FINDING_COLUMNS].notna().any(axis=1)].reset_index(drop=True)

for finding in ["PF OA", "Synovitis", "Effusion"]:
    print(f"\n{'=' * 20} {finding} FALSE NEGATIVES/POSITIVES {'=' * 20}")
    for _, row in labeled.iterrows():
        pred = extract_labels_from_report(row["report_english"])[finding]
        true = row[finding]
        if true == 1.0 and pred == 0.0:
            print(f"\n--- FALSE NEGATIVE (missed a real positive) ---")
            print(row["report_english"][:400])
        elif true == 0.0 and pred == 1.0:
            print(f"\n--- FALSE POSITIVE (wrongly flagged) ---")
            print(row["report_english"][:400])