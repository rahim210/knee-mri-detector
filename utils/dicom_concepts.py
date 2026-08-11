"""
utils/dicom_concepts.py

Reference definitions for core medical imaging concepts used throughout
this project. This module documents terminology as executable code so
it stays precise and testable, rather than living only in comments.
"""

from dataclasses import dataclass
from enum import Enum


class AnatomicalPlane(str, Enum):
    """The three standard MRI viewing planes.

    In our competition's train_series.csv, each series is tagged with
    exactly one of these planes.
    """

    SAGITTAL = "Sagittal"
    CORONAL = "Coronal"
    AXIAL = "Axial"


@dataclass
class SeriesInfo:
    """Describes one MRI series (a single stack of slices).

    A 'study' (one knee exam) contains multiple series -- for example,
    a sagittal T2 series and an axial PD series -- each capturing the
    knee from a different plane and/or with different tissue contrast.

    Attributes:
        plane: Which of the three standard planes this series was
            acquired in.
        num_slices: How many 2D slices make up this series (its "depth").
        is_fluid_sensitive: True if the sequence emphasizes fluid signal
            (T2, PD, STIR-like), making fluid (e.g. effusion) appear bright.
        has_fat_suppression: True if fat signal was suppressed during
            acquisition, which increases contrast for certain findings.
    """

    plane: AnatomicalPlane
    num_slices: int
    is_fluid_sensitive: bool
    has_fat_suppression: bool


def describe_series(series: SeriesInfo) -> str:
    """Produce a human-readable description of a series' key properties.

    Args:
        series: The series to describe.

    Returns:
        A one-line plain-English summary of the series' imaging properties.
    """
    fluid_desc = "fluid-sensitive" if series.is_fluid_sensitive else "not fluid-sensitive"
    fat_desc = "with fat suppression" if series.has_fat_suppression else "without fat suppression"

    return (
        f"{series.plane.value} series, {series.num_slices} slices, "
        f"{fluid_desc}, {fat_desc}"
    )