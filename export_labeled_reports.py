import pandas as pd

df = pd.read_csv('data/train.csv')
finding_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture',
]

labeled = df[df[finding_cols].notna().any(axis=1)]
labeled[['StudyInstanceUID', 'Report'] + finding_cols].to_csv(
    'data/labeled_reports_export.csv', index=False
)
print(f"Exported {len(labeled)} labeled reports to data/labeled_reports_export.csv")