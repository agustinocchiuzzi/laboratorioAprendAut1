"""Discretizacion ajustada con el train de cada fold, para los clasificadores
propios (ID3 y Naive Bayes categorico).

Es una clase Python simple: se ajusta (fit) y se transforma a mano en cada
fold de la validacion, sin herencia de scikit-learn ni Pipeline (la
validacion temporal se hace a mano en evaluation.py).
"""

import numpy as np
import pandas as pd


class MixedTypeDiscretizer:
    """Convierte tasas numericas en codigos (bins) y categorias en codigos.

    Las columnas numericas que no estan en ``fixed_cuts`` se parten con
    cuantiles (igual frecuencia) calculados en fit() sobre el train de cada
    fold. Las que figuran en ``fixed_cuts`` usan cortes fijos dados por el
    usuario, que no dependen de los datos, asi que no introducen leakage y no
    requieren reajustarse. El valor 0 queda reservado para valores
    desconocidos o faltantes; los numeros no finitos se imputan con la mediana
    del train (0 si la columna de train no tiene valores finitos).
    """

    def __init__(self, categorical_features, numeric_features, n_bins=5,
                 fixed_cuts=None):
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.n_bins = n_bins
        self.fixed_cuts = fixed_cuts

    def fit(self, X, y=None):
        # Categorias: mapa {valor: codigo} en orden alfabetico; el 0 queda
        # reservado para valores que no se vieron en train.
        self.category_maps_ = {}
        for column in self.categorical_features:
            values = sorted(X[column].dropna().astype(str).unique())
            self.category_maps_[column] = {
                value: index + 1 for index, value in enumerate(values)
            }

        # Numerico: mediana del train (para imputar en transform) y bordes
        # de los bines. Los bordes de los cuantiles se calculan solo con train.
        self.numeric_medians_ = {}
        self.numeric_edges_ = {}
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        for column in self.numeric_features:
            values = pd.to_numeric(X[column], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            self.numeric_medians_[column] = (
                float(np.median(finite)) if finite.size else 0.0
            )
            if self.fixed_cuts and column in self.fixed_cuts:
                self.numeric_edges_[column] = np.asarray(
                    self.fixed_cuts[column], dtype=float
                )
            else:
                self.numeric_edges_[column] = (
                    np.unique(np.quantile(finite, quantiles))
                    if finite.size
                    else np.asarray([], dtype=float)
                )
        return self

    def transform(self, X):
        columns = []
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
            # La mediana es la de train y no cambia: nunca se aprende de
            # validacion/test.
            values = np.where(
                np.isfinite(values), values, self.numeric_medians_[column]
            )
            encoded = np.digitize(values, self.numeric_edges_[column]) + 1
            columns.append(encoded.astype(np.int64))

        return np.column_stack(columns)