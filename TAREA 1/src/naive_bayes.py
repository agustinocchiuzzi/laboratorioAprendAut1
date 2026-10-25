"""Naive Bayes categorico: cuenta frecuencias y las suaviza con m-estimate.

Es una clase Python simple (sin herencia de scikit-learn): se ajusta y se
predice a mano en cada fold. Los atributos terminados en _ guardan lo
aprendido durante fit.
"""

import numpy as np


class MEstimateCategoricalNB:
    """Predice E, L o V combinando la frecuencia de clase y la de cada atributo.

    X tiene una fila por partido y una columna por atributo ya discretizado
    (enteros no negativos. El 0 queda para desconocidos). Se supone independencia
    entre atributos dada la clase, la parte naive del modelo.

    El suavizado con m funciona asi: a cada codigo se le suma m dividido la
    cantidad de codigos del atributo. Con m chico los datos mandan. Con m
    grande las probabilidades se acercan a la uniforme. La clase si se estima
    por frecuencia, sin suavizar. min_categories declara un dominio mayor que
    el de train (un entero o uno por atributo) por si un codigo no aparece. Por
    defecto se infiere de train.
    """

    def __init__(self, m=1.0, min_categories=None):
        self.m = m
        self.min_categories = min_categories

    def fit(self, X, y):
        """Cuenta clases y codigos en train y guarda sus log-probabilidades."""
        if self.m <= 0:
            raise ValueError("m debe ser estrictamente positivo.")
        values = np.asarray(X).astype(np.int64, copy=False)
        target = np.asarray(y)
        if target.ndim == 2 and target.shape[1] == 1:
            target = target[:, 0]

        cardinalities = values.max(axis=0) + 1
        if self.min_categories is not None:
            cardinalities = np.maximum(
                cardinalities, np.asarray(self.min_categories)
            ).astype(np.int64)

        # Convierte etiquetas en indices numericos en orden de aparicion.
        self.classes_, encoded_y = np.unique(target, return_inverse=True)
        class_counts = np.bincount(encoded_y, minlength=len(self.classes_))
        # P(c) es la proporcion de partidos de cada clase, sin suavizado.
        self.class_log_prior_ = np.log(class_counts / class_counts.sum())
        self.feature_log_prob_ = []

        for feature in range(values.shape[1]):
            cardinality = int(cardinalities[feature])
            # Una tabla por atributo: una fila por clase y una columna por codigo.
            counts = np.zeros((len(self.classes_), cardinality), dtype=float)
            for class_index in range(len(self.classes_)):
                counts[class_index] = np.bincount(
                    values[encoded_y == class_index, feature],
                    minlength=cardinality,
                )
            prior_probability = 1.0 / cardinality
            # Se resta en log para no formar cocientes chicos. Logaddexp
            # maneja los conteos cero sin perder el pseudoconteo.
            with np.errstate(divide="ignore"):
                log_counts = np.log(counts)
            log_smoothed = np.logaddexp(
                log_counts, np.log(self.m) + np.log(prior_probability)
            )
            log_totals = np.logaddexp(np.log(class_counts), np.log(self.m))
            # np.newaxis le resta a cada fila su propio total.
            self.feature_log_prob_.append(log_smoothed - log_totals[:, np.newaxis])
        return self

    def _joint_log_likelihood(self, X):
        """Suma el log de la clase y el log de cada atributo, por partido."""
        values = np.asarray(X).astype(np.int64, copy=False)
        # Cada partido arranca con la probabilidad de cada clase en log.
        joint = np.tile(self.class_log_prior_, (len(values), 1))
        for feature, log_probabilities in enumerate(self.feature_log_prob_):
            feature_values = values[:, feature].copy()
            # Un codigo fuera de la tabla usa la columna reservada 0.
            feature_values[feature_values >= log_probabilities.shape[1]] = 0
            # .T deja la tabla ordenada como partidos por clase.
            joint += log_probabilities[:, feature_values].T
        return joint

    def predict(self, X):
        """Devuelve la etiqueta con mayor puntaje para cada partido."""
        joint = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(joint, axis=1)]

    def predict_proba(self, X):
        """Normaliza los puntajes: cada fila suma 1, en el orden de classes_."""
        joint = self._joint_log_likelihood(X)
        restado = joint - joint.max(axis=1, keepdims=True)
        exp = np.exp(restado)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_log_proba(self, X):
        """Devuelve el log de las probabilidades de cada clase por partido.

        No se usa el log de predict_proba porque una probabilidad puede
        redondearse a cero aunque su log sea finito. Se normaliza en log.
        """
        joint = self._joint_log_likelihood(X)
        restado = joint - joint.max(axis=1, keepdims=True)
        log_sum = joint.max(axis=1, keepdims=True) + np.log(
            np.exp(restado).sum(axis=1, keepdims=True)
        )
        return joint - log_sum
