import pandas as pd
import re

df = pd.read_csv('data/train.csv')
finding_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture',
]

labeled = df[df[finding_cols].notna().any(axis=1)]


def guess_language(text: str) -> str:
    """Very rough language guess based on common Spanish/English words."""
    text_lower = str(text).lower()
    spanish_markers = ['tróclea', 'derrame', 'leve', 'rodilla', 'lesión', 'grado', 'articular']
    english_markers = ['impression', 'findings', 'tear', 'the', 'and', 'moderate']

    es_count = sum(text_lower.count(w) for w in spanish_markers)
    en_count = sum(text_lower.count(w) for w in english_markers)

    if es_count > en_count:
        return 'likely_spanish'
    elif en_count > es_count:
        return 'likely_english'
    return 'unclear'


labeled = labeled.copy()
labeled['guessed_language'] = labeled['Report'].apply(guess_language)
print(labeled['guessed_language'].value_counts())
print()
print("Sample report lengths (characters):")
print(labeled['Report'].str.len().describe())