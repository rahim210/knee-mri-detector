import pandas as pd

df = pd.read_csv('data/train.csv')
finding_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture',
]

has_label = df[finding_cols].notna().any(axis=1)
print('Total studies:', len(df))
print('Studies WITH at least 1 label:', has_label.sum())
print('Studies with NO labels (report-only):', (~has_label).sum())
print('Percent labeled:', round(100 * has_label.sum() / len(df), 1), '%')