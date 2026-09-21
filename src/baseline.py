"""Baseline on the same pre-match, online histories as the learned models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

BASELINE_FEATURES = ["home_win_rate_10y", "away_win_rate_10y"]


class TenYearWinRateClassifier(ClassifierMixin, BaseEstimator):
    """Compare causal ten-year win rates produced by features.py.

    predict never reads target labels or mutates a history. The shared feature
    builder includes only earlier dates, including earlier validation/test dates,
    and uses 0.5 for both teams without history. fit records the input contract;
    there are no learned parameters. Ties favor home unless configured otherwise.
    """

    def __init__(self, tie_break: str = "home") -> None:
        self.tie_break = tie_break

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("El baseline requiere tasas causales en un DataFrame.")
        missing = set(BASELINE_FEATURES).difference(X.columns)
        if missing:
            raise ValueError("Faltan tasas causales: " + ", ".join(sorted(missing)))
        rates = X.loc[:, BASELINE_FEATURES].astype(float)
        if not np.isfinite(rates.to_numpy()).all() or ((rates < 0) | (rates > 1)).any().any():
            raise ValueError("Las tasas deben ser finitas y estar entre 0 y 1.")
        return rates

    def fit(self, X: pd.DataFrame, y):
        if self.tie_break not in {"home", "away"}:
            raise ValueError("tie_break debe ser 'home' o 'away'.")
        rates = self._validate_frame(X)
        target = np.asarray(y)
        if target.ndim != 1 or len(rates) != len(target) or not len(target):
            raise ValueError("X e y deben tener igual cantidad no nula de filas.")
        if not np.isin(target, ["E", "L", "V"]).all():
            raise ValueError("y solo puede contener E, L y V.")
        self.classes_ = np.unique(target)
        self.n_features_in_ = len(BASELINE_FEATURES)
        self.feature_names_in_ = np.asarray(BASELINE_FEATURES, dtype=object)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "classes_")
        rates = self._validate_frame(X)
        home, away = (rates[column].to_numpy() for column in BASELINE_FEATURES)
        return np.where(home > away, "L", np.where(away > home, "V",
                        "L" if self.tie_break == "home" else "V"))
