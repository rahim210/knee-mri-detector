import pandas as pd

labels = pd.read_csv('data/train.csv')
manifest = pd.read_csv('data/subset_manifest_clean.csv')
our_studies = manifest['study_id'].unique()
subset_labels = labels[labels['StudyInstanceUID'].isin(our_studies)]

finding_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture',
]

has_any_label = subset_labels[finding_cols].notna().any(axis=1)
print('Studies with at least 1 real label:', has_any_label.sum(), '/', len(subset_labels))
print()
print(subset_labels[['StudyInstanceUID'] + finding_cols].to_string())