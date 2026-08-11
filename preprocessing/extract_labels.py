"""
preprocessing/extract_labels.py

Rule-based extraction of the 12 RSNA knee finding labels from
English-translated radiology report text. Built and validated
against the 58 studies that already have ground-truth structured
labels (see validate_extractor.py).

Each finding has a list of regex patterns. A finding is marked
positive (1.0) if the report contains a positive-context match for
it, and negative (0.0) if the report explicitly states it's normal/
absent, or simply never mentions it (the common case for unaffected
structures in a normal report).
"""

import logging
import re

logger = logging.getLogger(__name__)

FINDING_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

POSITIVE_PATTERNS: dict[str, list[str]] = {
    "ACL": [
        r"\bACL\b.{0,60}\b(tear|rupture|torn|ruptured|injury|sprain|disruption)\b",
        r"\banterior cruciate ligament\b.{0,60}\b(tear|rupture|torn|ruptured|injury|sprain|disruption)\b",
        r"\b(tear|rupture|injury)\b.{0,60}\banterior cruciate ligament\b",
        r"\bACL\b.{0,40}\b(grade\s*[1-3]|partial|complete|full[- ]thickness)\b",
    ],
    "MCL": [
        r"\bMCL\b.{0,60}\b(tear|rupture|torn|ruptured|injury|sprain|strain)\b",
        r"\bmedial collateral ligament\b.{0,60}\b(tear|rupture|torn|ruptured|injury|sprain|strain)\b",
        r"\bMCL\b.{0,40}\b(grade\s*[1-3]|partial|complete)\b",
    ],
    "Medial Meniscus": [
        r"\bmedial meniscus\b.{0,80}\b(tear|torn|rupture|degenerat\w*|extrusion|maceration)\b",
        r"\b(tear|torn|rupture)\b.{0,60}\bmedial meniscus\b",
        r"\bmedial meniscal\b.{0,60}\btear\b",
    ],
    "Lateral Meniscus": [
        r"\blateral meniscus\b.{0,80}\b(tear|torn|rupture|degenerat\w*|extrusion|maceration)\b",
        r"\b(tear|torn|rupture)\b.{0,60}\blateral meniscus\b",
        r"\blateral meniscal\b.{0,60}\btear\b",
    ],
    "Medial OA": [
        r"\bmedial\b.{0,30}\b(osteoarthritis|OA|compartment\w*)\b.{0,40}\b(osteoarthritis|chondrosis|cartilage loss|chondropathy|chondromalacia)\b",
        r"\btricompartmental\b.{0,40}\b(osteoarthritis|chondrosis)\b",
        r"\ball\s+three\s+compartment\w*\b",
        r"\bmedial\b.{0,10}(femorotibial|tibiofemoral)\b.{0,40}\b(osteoarthritis|chondrosis|cartilage loss|OA)\b",
        r"\bmedial\s+(and|&)\s+lateral\s+femorotibial\s+OA\b",
    ],
    "Lateral OA": [
        r"\blateral\b.{0,30}\b(osteoarthritis|OA|compartment\w*)\b.{0,40}\b(osteoarthritis|chondrosis|cartilage loss|chondropathy|chondromalacia)\b",
        r"\btricompartmental\b.{0,40}\b(osteoarthritis|chondrosis)\b",
        r"\ball\s+three\s+compartment\w*\b",
        r"\blateral\b.{0,10}(femorotibial|tibiofemoral)\b.{0,40}\b(osteoarthritis|chondrosis|cartilage loss|OA)\b",
        r"\bmedial\s+(and|&)\s+lateral\s+femorotibial\s+OA\b",
    ],
    "PF OA": [
        r"\bpatellofemoral\b.{0,60}\b(osteoarthritis|OA|chondrosis|cartilage loss|chondropathy|chondromalacia|arthrotic)\b",
        r"\bPF\b.{0,20}\b(OA|osteoarthritis|arthrotic)\b",
        r"\btricompartmental\b.{0,40}\b(osteoarthritis|chondrosis)\b",
        r"\ball\s+three\s+compartment\w*\b",
        r"\btrochle\w*\b.{0,60}\b(chondropathy|chondrosis|cartilage loss|chondromalacia)\b",
        r"\bpatell\w*\b.{0,40}\b(chondropathy|chondromalacia|cartilage loss|chondrosis)\b",
        r"\bchondromalacia\b.{0,60}\b(patella|trochlea|facet)\w*\b",
        r"\bfacets?\s+of\s+the\s+patella\b.{0,40}\btrochlea\b",
    ],
    "Effusion": [
        r"\b(joint\s+)?effusion\b",
        r"\bderrame\b",
        r"\bhemarthrosis\b",
    ],
    "Synovitis": [
        r"\bsynovitis\b",
        r"\bsinovitis\b",
        r"\bsynovial\s+(thickening|hypertrophy|proliferation)\b",
        r"\bhypertrophy of the synovium\b",
        r"\bproliferation of (the\s+)?synovium\b",
        r"\bthickened\s+synovi\w*\b",
        r"\breactive\s+synovi\w*\b",
        r"\bthickened\s+synovial\s+tissue\b",
    ],
    "Baker's": [
        r"\bbaker'?s?\s+cyst\b",
        r"\bpopliteal\s+cyst\b",
    ],
    "Contusion": [
        r"\b(bone\s+)?contusion\b",
        r"\bbone\s+bruise\b",
        r"\bosteochondral\s+impact\w*\b",
        r"\bbone\s+marrow\s+edema\b.{0,40}\b(contusion|trauma\w*)\b",
    ],
    "Fracture": [
        r"\bfracture\b",
        r"\bavulsion\b.{0,20}\bfractur\w*\b",
        r"\bsegond\s+fracture\b",
    ],
}

NEGATION_WORDS = [
    "no ", "not ", "without ", "negative for ", "absence of ", "denies ",
    "no evidence of ", "no significant ", "no acute ",
]


def _has_negated_match(text: str, match_start: int, window: int = 60) -> bool:
    """Check if a match is preceded by a negation word within a small window.

    Only counts a negation if there is no sentence-ending punctuation
    (period, newline) between the negation word and the match, so a
    negation in one sentence cannot suppress a positive finding in
    the next sentence.
    """
    context_start = max(0, match_start - window)
    preceding_text = text[context_start:match_start]

    # Only look at text after the last sentence boundary within the window
    last_boundary = max(
        preceding_text.rfind("."),
        preceding_text.rfind("\n"),
    )
    same_sentence_text = preceding_text[last_boundary + 1:]

    return any(neg in same_sentence_text for neg in NEGATION_WORDS)


def extract_labels_from_report(report_text: str) -> dict[str, float]:
    """Extract the 12 finding labels from a single English report."""
    text = str(report_text).lower()
    results: dict[str, float] = {}

    for finding, patterns in POSITIVE_PATTERNS.items():
        found_positive = False
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                if not _has_negated_match(text, match.start()):
                    found_positive = True
                    break
            if found_positive:
                break
        results[finding] = 1.0 if found_positive else 0.0

    return results

