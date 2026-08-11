import pandas as pd

df = pd.read_csv('data/train.csv')
finding_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture',
]

labeled = df[df[finding_cols].notna().any(axis=1)]
row = labeled.iloc[0]

print("FULL REPORT:")
print(row["Report"])
print()
print("=" * 60)
print("LABELS:")
for col in finding_cols:
    print(f"  {col}: {row[col]}")