import pandas as pd

df = pd.read_csv('data/train.csv')
finding_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture',
]

labeled = df[df[finding_cols].notna().any(axis=1)]
print(f"Total labeled studies: {len(labeled)}")
print()

for i in range(3):
    row = labeled.iloc[i]
    print(f"--- Example {i} ---")
    print("Report:", row["Report"])
    print()
    print("Labels:")
    for col in finding_cols:
        print(f"  {col}: {row[col]}")
    print()