"""A compact categorical ID3 decision tree with minimum information gain."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


@dataclass
class _Node:
    prediction_index: int
    class_counts: np.ndarray
    feature: int | None = None
    gain: float = 0.0
    children: dict[int, _Node] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class CategoricalDecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    """ID3-style multiway tree for discrete non-negative attributes.

    A node is expanded only when its best information gain is strictly greater
    than ``min_info_gain``. An attribute is used at most once along a branch.
    Unknown values fall back to the majority distribution stored at that node.
    """

    def __init__(
        self,
        min_info_gain: float = 0.0,
        max_depth: int | None = 8,
        min_samples_split: int = 2,
    ) -> None:
        self.min_info_gain = min_info_gain
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    @staticmethod
    def _validate_X(X) -> np.ndarray:
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("X debe estar discretizada como valores enteros.")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("X debe contener enteros no negativos y finitos.")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError("X debe contener valores enteros discretos.")
        return values.astype(np.int64, copy=False)

    @staticmethod
    def _entropy(encoded_y: np.ndarray, n_classes: int) -> float:
        counts = np.bincount(encoded_y, minlength=n_classes)
        probabilities = counts[counts > 0] / len(encoded_y)
        return float(-(probabilities * np.log2(probabilities)).sum())

    def _information_gain(
        self, values: np.ndarray, encoded_y: np.ndarray, feature: int
    ) -> float:
        parent_entropy = self._entropy(encoded_y, len(self.classes_))
        conditional_entropy = 0.0
        for value in np.unique(values[:, feature]):
            mask = values[:, feature] == value
            conditional_entropy += mask.mean() * self._entropy(
                encoded_y[mask], len(self.classes_)
            )
        return parent_entropy - conditional_entropy

    def _build(
        self,
        values: np.ndarray,
        encoded_y: np.ndarray,
        available_features: tuple[int, ...],
        depth: int,
    ) -> _Node:
        counts = np.bincount(encoded_y, minlength=len(self.classes_))
        node = _Node(prediction_index=int(np.argmax(counts)), class_counts=counts)
        reached_max_depth = self.max_depth is not None and depth >= self.max_depth
        if (
            np.count_nonzero(counts) == 1
            or len(encoded_y) < self.min_samples_split
            or not available_features
            or reached_max_depth
        ):
            return node

        gains = [
            (self._information_gain(values, encoded_y, feature), feature)
            for feature in available_features
        ]
        best_gain, best_feature = max(gains, key=lambda item: (item[0], -item[1]))
        if best_gain <= self.min_info_gain:
            return node

        node.feature = best_feature
        node.gain = best_gain
        self._feature_gain_totals[best_feature] += best_gain * len(encoded_y)
        remaining = tuple(
            feature for feature in available_features if feature != best_feature
        )
        for value in np.unique(values[:, best_feature]):
            mask = values[:, best_feature] == value
            node.children[int(value)] = self._build(
                values[mask], encoded_y[mask], remaining, depth + 1
            )
        return node

    def fit(self, X, y):
        if self.min_info_gain < 0:
            raise ValueError("min_info_gain no puede ser negativo.")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth debe ser al menos 1 o None.")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split debe ser al menos 2.")

        values = self._validate_X(X)
        target = np.asarray(y)
        if len(values) != len(target):
            raise ValueError("X e y deben tener la misma cantidad de filas.")
        self.classes_, encoded_y = np.unique(target, return_inverse=True)
        self.n_features_in_ = values.shape[1]
        self._feature_gain_totals = np.zeros(self.n_features_in_, dtype=float)
        self.tree_ = self._build(
            values, encoded_y, tuple(range(self.n_features_in_)), depth=0
        )
        total_gain = self._feature_gain_totals.sum()
        self.feature_importances_ = (
            self._feature_gain_totals / total_gain
            if total_gain > 0
            else self._feature_gain_totals.copy()
        )
        return self

    def _leaf_for_row(self, row: np.ndarray) -> _Node:
        node = self.tree_
        while not node.is_leaf:
            child = node.children.get(int(row[node.feature]))
            if child is None:
                break
            node = child
        return node

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "tree_")
        values = self._validate_X(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("X tiene una cantidad de atributos inesperada.")
        indices = [self._leaf_for_row(row).prediction_index for row in values]
        return self.classes_[indices]

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "tree_")
        values = self._validate_X(X)
        probabilities = []
        for row in values:
            counts = self._leaf_for_row(row).class_counts.astype(float)
            probabilities.append(counts / counts.sum())
        return np.asarray(probabilities)

    def get_depth(self) -> int:
        check_is_fitted(self, "tree_")

        def depth(node: _Node) -> int:
            return (
                0
                if node.is_leaf
                else 1 + max(depth(child) for child in node.children.values())
            )

        return depth(self.tree_)

    def get_n_leaves(self) -> int:
        check_is_fitted(self, "tree_")

        def leaves(node: _Node) -> int:
            return 1 if node.is_leaf else sum(leaves(c) for c in node.children.values())

        return leaves(self.tree_)
