"""Naive Bayes categorico: aprende frecuencias y las suaviza con m-estimate."""

from __future__ import annotations

from numbers import Real

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import softmax
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d


class MEstimateCategoricalNB(ClassifierMixin, BaseEstimator):
    r"""Predice E, L o V combinando la frecuencia de clase y de cada atributo.

    X tiene una fila por partido y una columna por atributo ya discretizado.
    Los codigos son enteros no negativos; 0 queda para desconocidos.
    En el pipeline de tres bines, 1, 2, 3 representan baja, media, alta.
    Se supone independencia entre atributos dada la clase (la parte "naive").

    Para cada atributo, codigo v y clase c:
    ``P(v|c) = (count(v,c) + m / K) / (count(c) + m)``.

    K es propio de cada atributo: maximo codigo de train + 1, incluyendo cero
    y los codigos intermedios aunque no aparezcan. ``min_categories`` permite
    declarar un dominio mayor (un entero comun o un entero por atributo), sin
    mirar validacion/test. Por defecto se infiere exclusivamente de train.
    m equivale a agregar m observaciones ficticias repartidas entre K codigos:
    evita probabilidades cero; un m mayor acerca la distribucion a la uniforme.
    Este prior uniforme de categorias no es el prior de clase P(c), que se
    estima por frecuencia y no se suaviza, ni la frecuencia marginal de X_j.
    Por ejemplo, con count(v,c)=3, count(c)=10, K=4 y m=2: P(v|c)=3.5/12.

    CategoricalNB usa (count(v,c) + alpha) / (count(c) + alpha*K).
    Equivale con alpha=m/K solo si ambos usan el mismo dominio, prior de clase
    y codificacion de desconocidos. Un alpha escalar comun exige el mismo K
    en todos los atributos; K no es necesariamente cuatro.

    BaseEstimator y ClassifierMixin permiten usar la interfaz de scikit-learn.
    Los atributos terminados en _ guardan lo aprendido durante fit.
    """

    def __init__(self, m: float = 1.0, min_categories=None) -> None:
        self.m = m
        self.min_categories = min_categories

    @staticmethod
    def _validate_X(X) -> np.ndarray:
        """Acepta una matriz de codigos enteros, finitos y no negativos."""
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")
        if not all(values.shape):
            raise ValueError("X debe contener al menos una fila y un atributo.")
        if values.dtype.kind not in "iuf":
            raise TypeError("X debe estar discretizada como valores enteros.")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("X debe contener enteros no negativos y finitos.")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError("X debe contener valores enteros discretos.")
        if (values >= 2**63).any():
            raise ValueError("Los codigos de X deben ser representables como int64.")
        return values.astype(np.int64, copy=False)

    def fit(self, X, y):
        """Cuenta clases y codigos en train; guarda sus probabilidades en log."""
        if not isinstance(self.m, Real) or not np.isfinite(self.m) or self.m <= 0:
            raise ValueError("m debe ser un numero finito y estrictamente positivo.")
        values = self._validate_X(X)
        target = check_array(column_or_1d(y, warn=True), ensure_2d=False, dtype=None)
        check_classification_targets(target)
        if len(values) != len(target):
            raise ValueError("X e y deben tener la misma cantidad de filas.")

        cardinalities = values.max(axis=0) + 1
        if self.min_categories is not None:
            minimum = np.asarray(self.min_categories)
            if (minimum.ndim > 1
                    or (minimum.ndim == 1 and minimum.shape != cardinalities.shape)
                    or minimum.dtype.kind not in "iu"
                    or (minimum < 1).any()
                    or (minimum >= 2**63).any()):
                raise ValueError(
                    "min_categories debe ser un entero positivo o un entero "
                    "positivo por atributo."
                )
            cardinalities = np.maximum(cardinalities, minimum).astype(np.int64)

        # Convierte etiquetas en indices: E, L, V -> 0, 1, 2 si estan presentes.
        self.classes_, encoded_y = np.unique(target, return_inverse=True)
        class_counts = np.bincount(encoded_y, minlength=len(self.classes_))
        self.class_count_ = class_counts
        # P(c) es la proporcion de partidos de cada clase, sin suavizado.
        self.class_log_prior_ = np.log(class_counts / class_counts.sum())
        self.n_features_in_ = values.shape[1]
        self.n_categories_ = cardinalities
        self.feature_log_prob_: list[np.ndarray] = []
        self.category_count_: list[np.ndarray] = []

        for feature in range(values.shape[1]):
            cardinality = int(self.n_categories_[feature])
            # Una tabla por atributo: filas = clases, columnas = codigos.
            counts = np.zeros((len(self.classes_), cardinality), dtype=float)
            for class_index in range(len(self.classes_)):
                # Filtra partidos de esta clase y cuenta cada codigo del atributo.
                counts[class_index] = np.bincount(
                    values[encoded_y == class_index, feature],
                    minlength=cardinality,
                )
            prior_probability = 1.0 / cardinality
            self.category_count_.append(counts)
            # log(n_jvc + m/K) - log(n_c + m) evita formar cocientes diminutos.
            # logaddexp tambien admite conteos cero sin perder el pseudoconteo.
            with np.errstate(divide="ignore"):
                log_counts = np.log(counts)
            log_smoothed = np.logaddexp(
                log_counts, np.log(self.m) + np.log(prior_probability)
            )
            log_totals = np.logaddexp(np.log(class_counts), np.log(self.m))
            # np.newaxis resta a cada fila su propio log(n_c + m).
            self.feature_log_prob_.append(log_smoothed - log_totals[:, np.newaxis])
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
        joint = self._joint_log_likelihood(X)
        # No usar log(predict_proba): una probabilidad puede redondearse a cero
        # aunque su logaritmo sea finito. Normalizamos directamente en log.
        return joint - logsumexp(joint, axis=1, keepdims=True)
