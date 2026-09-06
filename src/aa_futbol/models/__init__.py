"""Custom classifiers required by the assignment."""

from aa_futbol.models.baseline import TenYearWinRateClassifier
from aa_futbol.models.id3 import CategoricalDecisionTreeClassifier
from aa_futbol.models.naive_bayes import MEstimateCategoricalNB

__all__ = [
    "CategoricalDecisionTreeClassifier",
    "MEstimateCategoricalNB",
    "TenYearWinRateClassifier",
]
