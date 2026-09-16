"""Naive Bayes categorico: aprende frecuencias y las suaviza con m-estimate."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_is_fitted


class MEstimateCategoricalNB(ClassifierMixin, BaseEstimator):
    r"""Predice E, L o V combinando la frecuencia de clase y de cada atributo.

    X tiene una fila por partido y una columna por atributo ya discretizado.
    Los codigos 1, 2, 3 representan baja, media, alta; 0 queda para desconocidos.
    Se supone independencia entre atributos dada la clase (la parte "naive").

    Para cada atributo, codigo v y clase c:
    ``P(v|c) = (count(v,c) + m / K) / (count(c) + m)``.

    K abarca los codigos desde 0 hasta el maximo observado en ese atributo.
    m equivale a agregar m observaciones ficticias repartidas entre K codigos:
    evita probabilidades cero; un m mayor acerca la distribucion a la uniforme.
    Por ejemplo, con count(v,c)=3, count(c)=10, K=4 y m=2: P(v|c)=3.5/12.

    BaseEstimator y ClassifierMixin permiten usar la interfaz de scikit-learn.
    Los atributos terminados en _ guardan lo aprendido durante fit.
    """

    def __init__(self, m: float = 1.0) -> None:
        self.m = m

    @staticmethod
    def _validate_X(X) -> np.ndarray:
        """Acepta una matriz de codigos enteros, finitos y no negativos."""
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
        """Cuenta clases y codigos en train; guarda sus probabilidades en log."""
        if self.m <= 0:
            raise ValueError("m debe ser estrictamente positivo.")
        values = self._validate_X(X)
        target = np.asarray(y)
        if len(values) != len(target):
            raise ValueError("X e y deben tener la misma cantidad de filas.")

        # Convierte etiquetas en indices: E, L, V -> 0, 1, 2 si estan presentes.
        self.classes_, encoded_y = np.unique(target, return_inverse=True)
        class_counts = np.bincount(encoded_y, minlength=len(self.classes_))
        self.class_count_ = class_counts
        # P(c) es la proporcion de partidos de cada clase, sin suavizado.
        self.class_log_prior_ = np.log(class_counts / class_counts.sum())
        self.n_features_in_ = values.shape[1]
        self.feature_log_prob_: list[np.ndarray] = []
        self.category_count_: list[np.ndarray] = []

        for feature in range(values.shape[1]):
            # +1 incluye el codigo 0, aunque no aparezca en train.
            cardinality = int(values[:, feature].max(initial=0)) + 1
            # Una tabla por atributo: filas = clases, columnas = codigos.
            counts = np.zeros((len(self.classes_), cardinality), dtype=float)
            for class_index in range(len(self.classes_)):
                # Filtra partidos de esta clase y cuenta cada codigo del atributo.
                counts[class_index] = np.bincount(
                    values[encoded_y == class_index, feature],
                    minlength=cardinality,
                )
            prior_probability = 1.0 / cardinality
            # np.newaxis permite dividir cada fila por su propio n_c + m.
            probabilities = (counts + self.m * prior_probability) / (
                class_counts[:, np.newaxis] + self.m
            )
            self.category_count_.append(counts)
            # Luego sumamos logs en lugar de multiplicar probabilidades pequenas.
            self.feature_log_prob_.append(np.log(probabilities))
        return self

    def _joint_log_likelihood(self, X) -> np.ndarray:
        """Calcula log P(c) + suma_j log P(x_j|c) para cada partido y clase."""
        check_is_fitted(self, "feature_log_prob_")
        values = self._validate_X(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("X tiene una cantidad de atributos inesperada.")

        # Cada partido empieza con los mismos puntajes iniciales log P(c).
        joint = np.tile(self.class_log_prior_, (len(values), 1))
        for feature, log_probabilities in enumerate(self.feature_log_prob_):
            feature_values = values[:, feature].copy()
            # Un codigo fuera de la tabla usa la columna reservada 0.
            feature_values[feature_values >= log_probabilities.shape[1]] = 0
            # Busca el log de cada codigo; .T ordena como partidos x clases.
            joint += log_probabilities[:, feature_values].T
        return joint

    def predict(self, X) -> np.ndarray:
        """Devuelve la etiqueta con mayor puntaje para cada partido."""
        joint = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(joint, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        """Normaliza los puntajes: cada fila suma 1, en el orden de classes_."""
        # softmax exponencia y normaliza de forma numericamente estable.
        return softmax(self._joint_log_likelihood(X), copy=True)

    def predict_log_proba(self, X) -> np.ndarray:
        """Devuelve el log de las probabilidades normalizadas de cada clase."""
        probabilities = self.predict_proba(X)
        return np.log(probabilities)
