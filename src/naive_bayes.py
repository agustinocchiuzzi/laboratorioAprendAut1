"""Categorical Naive Bayes with the assignment's m-estimate parameter."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_is_fitted


class MEstimateCategoricalNB(ClassifierMixin, BaseEstimator):
    r"""Categorical Naive Bayes using an m-estimate for each conditional.

    For attribute value ``v`` and class ``c`` the estimator uses

    ``P(v|c) = (count(v,c) + m * p(v)) / (count(c) + m)``

    with a uniform ``p(v)`` over the observed cardinality plus the reserved
    unknown value. The interpretation is documented because the assignment does
    not state which prior distribution should accompany ``m``.
    """

    def __init__(self, m: float = 1.0) -> None:
        self.m = m

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

    def fit(self, X, y):
        if self.m <= 0:
            raise ValueError("m debe ser estrictamente positivo.")
        values = self._validate_X(X)
        target = np.asarray(y)
        if len(values) != len(target):
            raise ValueError("X e y deben tener la misma cantidad de filas.")

        self.classes_, encoded_y = np.unique(target, return_inverse=True)
        class_counts = np.bincount(encoded_y, minlength=len(self.classes_))
        self.class_count_ = class_counts
        self.class_log_prior_ = np.log(class_counts / class_counts.sum())
        self.n_features_in_ = values.shape[1]
        self.feature_log_prob_: list[np.ndarray] = []
        self.category_count_: list[np.ndarray] = []

        for feature in range(values.shape[1]):
            cardinality = int(values[:, feature].max(initial=0)) + 1
            counts = np.zeros((len(self.classes_), cardinality), dtype=float)
            for class_index in range(len(self.classes_)):
                counts[class_index] = np.bincount(
                    values[encoded_y == class_index, feature],
                    minlength=cardinality,
                )
            prior_probability = 1.0 / cardinality
            probabilities = (counts + self.m * prior_probability) / (
                class_counts[:, np.newaxis] + self.m
            )
            self.category_count_.append(counts)
            self.feature_log_prob_.append(np.log(probabilities))
        return self

    def _joint_log_likelihood(self, X) -> np.ndarray:
        check_is_fitted(self, "feature_log_prob_")
        values = self._validate_X(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("X tiene una cantidad de atributos inesperada.")

        joint = np.tile(self.class_log_prior_, (len(values), 1))
        for feature, log_probabilities in enumerate(self.feature_log_prob_):
            feature_values = values[:, feature].copy()
            feature_values[feature_values >= log_probabilities.shape[1]] = 0
            joint += log_probabilities[:, feature_values].T
        return joint

    def predict(self, X) -> np.ndarray:
        joint = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(joint, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        return softmax(self._joint_log_likelihood(X), copy=True)

    def predict_log_proba(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.log(probabilities)
