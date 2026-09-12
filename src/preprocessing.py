"""Fold-fitted discretization for the categorical custom classifiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class MixedTypeDiscretizer(TransformerMixin, BaseEstimator):
    """Encode strings and bin numbers as non-negative integers.

    Numeric columns not listed in ``fixed_cuts`` are split into equal-frequency
    bins using quantile edges fit on the training data (no leakage). Numeric
    columns listed in ``fixed_cuts`` use user-provided constant edges that do
    not depend on the data, so they introduce no leakage and need no refitting.
    Value zero is reserved for unseen or missing values. Fitting this
    transformer inside a Pipeline prevents category and quantile leakage across
    temporal folds.
    """

    def __init__(
        self,
        categorical_features: Sequence[str],
        numeric_features: Sequence[str],
        n_bins: int = 5,
        fixed_cuts: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.n_bins = n_bins
        self.fixed_cuts = fixed_cuts

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MixedTypeDiscretizer requiere un pandas DataFrame.")
        if self.n_bins < 2:
            raise ValueError("n_bins debe ser al menos 2.")

        self.feature_names_in_ = np.asarray(
            [*self.categorical_features, *self.numeric_features], dtype=object
        )
        missing = set(self.feature_names_in_).difference(X.columns)
        if missing:
            raise ValueError("Faltan atributos: " + ", ".join(sorted(missing)))

        self.category_maps_ = {}
        for column in self.categorical_features:
            values = sorted(X[column].dropna().astype(str).unique())
            self.category_maps_[column] = {
                value: index + 1 for index, value in enumerate(values)
            }

        self.numeric_edges_ = {}
        self.numeric_medians_ = {}
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        for column in self.numeric_features:
            values = pd.to_numeric(X[column], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if finite.size else 0.0
            self.numeric_medians_[column] = median
            if self.fixed_cuts and column in self.fixed_cuts:
                edges = np.asarray(self.fixed_cuts[column], dtype=float)
                if edges.ndim != 1 or edges.size < 1 or np.any(np.diff(edges) <= 0):
                    raise ValueError(
                        f"fixed_cuts[{column}] debe ser ascendente y no vacio."
                    )
                self.numeric_edges_[column] = edges
            else:
                self.numeric_edges_[column] = (
                    np.unique(np.quantile(finite, quantiles))
                    if finite.size
                    else np.asarray([], dtype=float)
                )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["category_maps_", "numeric_edges_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MixedTypeDiscretizer requiere un pandas DataFrame.")

        columns: list[np.ndarray] = []
        for column in self.categorical_features:
            encoded = (
                X[column]
                .astype("string")
                .map(self.category_maps_[column])
                .fillna(0)
                .to_numpy(dtype=np.int64)
            )
            columns.append(encoded)

        for column in self.numeric_features:
            values = pd.to_numeric(X[column], errors="coerce").to_numpy(dtype=float)
            values = np.where(
                np.isfinite(values), values, self.numeric_medians_[column]
            )
            encoded = np.digitize(values, self.numeric_edges_[column]) + 1
            columns.append(encoded.astype(np.int64))

        return np.column_stack(columns)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_.copy()
