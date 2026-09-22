"""Setup temporal compartido; importar este modulo no ajusta ningun modelo.

Los insumos provienen del armador de atributos causales auditado. La
validacion es manual (sin Pipeline de scikit-learn): en cada fold se ajusta
el preprocesado con el train del fold y se evaluan validacion y train.
"""

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

try:  # Notebook imports src modules directly; tests can import as a package.
    from .baseline import BASELINE_FEATURES
    from .features import NUMERIC_FEATURES
    from .preprocessing import MixedTypeDiscretizer
except ImportError:
    from baseline import BASELINE_FEATURES
    from features import NUMERIC_FEATURES
    from preprocessing import MixedTypeDiscretizer

VALIDATION_YEARS = (2021, 2022, 2023)
CLASSES = ("E", "L", "V")
LAST_5_CUTS = (0.3, 0.6)


@dataclass(frozen=True)
class TemporalFold:
    validation_year: int
    train_positions: tuple
    validation_positions: tuple


def _dates(frame):
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError(
            "Se requieren fechas de partido completas, sin horas ni faltantes."
        )
    return dates


def make_temporal_folds(frame):
    """Entrenamiento expandido antes de cada ano completo 2021, 2022 y 2023.

    Las posiciones refieren al frame recibido, sin importar su orden. Se
    rechaza el test 2024-2025 en lugar de dejarlo pasar por la CV.
    """
    dates = _dates(frame)
    if dates.dt.year.gt(2023).any():
        raise ValueError("La validacion cruzada no acepta el test de 2024--2025.")
    folds = []
    for year in VALIDATION_YEARS:
        train = np.flatnonzero(dates.dt.year.lt(year).to_numpy())
        valid = np.flatnonzero(dates.dt.year.eq(year).to_numpy())
        if not len(train) or not len(valid):
            raise ValueError(f"El fold {year} necesita entrenamiento y validacion.")
        folds.append(TemporalFold(year, tuple(map(int, train)), tuple(map(int, valid))))
    return tuple(folds)


def temporal_holdout(frame):
    dates = _dates(frame)
    if dates.dt.year.gt(2025).any():
        raise ValueError("Hay fechas posteriores al horizonte de la consigna.")
    train = frame.loc[dates.dt.year.le(2023)].reset_index(drop=True)
    test = frame.loc[dates.dt.year.between(2024, 2025)].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Entrenamiento y evaluacion deben contener partidos.")
    return train, test


def new_discretizer(feature_columns=None):
    """Devuelve un discretizador sin ajustar para cada particion de train."""
    return MixedTypeDiscretizer(
        categorical_features=[], numeric_features=(
            list(NUMERIC_FEATURES) if feature_columns is None else list(feature_columns)
        ),
        n_bins=3, fixed_cuts={column: LAST_5_CUTS for column in NUMERIC_FEATURES[:2]},
    )


def evaluate_temporal_cv(estimator, frame, folds, representation,
                         feature_columns=None):
    """Evalua solo cuando se la llama; modelo nuevo y ajustado en cada fold.

    Se revalidan las posiciones contra el frame exacto para que ningun modelo
    use un split distinto. representation indica el tipo de input:
    "discrete" aplica la discretizacion ajustada por fold, "continuous" pasa
    las tasas crudas y "baseline" las tasas de diez anos sin discretizar.
    feature_columns permite comparar las entradas con el mismo preprocesado.
    """
    if folds != make_temporal_folds(frame):
        raise ValueError("Todos los modelos deben usar los folds anuales compartidos.")
    columns = list(NUMERIC_FEATURES if feature_columns is None else feature_columns)
    rows = []
    for fold in folds:
        train = frame.iloc[list(fold.train_positions)]
        valid = frame.iloc[list(fold.validation_positions)]

        if representation == "discrete":
            discretizer = new_discretizer(columns)
            discretizer.fit(train[columns])
            X_train = discretizer.transform(train[columns])
            X_valid = discretizer.transform(valid[columns])
        elif representation == "continuous":
            X_train = train[columns]
            X_valid = valid[columns]
        elif representation == "baseline":
            X_train = train[BASELINE_FEATURES]
            X_valid = valid[BASELINE_FEATURES]
        else:
            raise ValueError("Representacion desconocida: " + representation)

        model = copy.deepcopy(estimator)
        model.fit(X_train, train["winner"])
        prediction = model.predict(X_valid)
        train_prediction = model.predict(X_train)
        rows.append({
            "validacion": fold.validation_year,
            "n_train": len(train), "n_validacion": len(valid),
            "accuracy": accuracy_score(valid["winner"], prediction),
            "validation_error": 1.0 - accuracy_score(valid["winner"], prediction),
            "train_accuracy": accuracy_score(train["winner"], train_prediction),
            "train_error": 1.0 - accuracy_score(train["winner"], train_prediction),
            "train_macro_f1": f1_score(train["winner"], train_prediction,
                                       labels=list(CLASSES), average="macro",
                                       zero_division=0),
            "macro_f1": f1_score(valid["winner"], prediction,
                                 labels=list(CLASSES), average="macro",
                                 zero_division=0),
        })
    return pd.DataFrame(rows)


def reporte_por_clases(nombre, y_true, y_pred, digits=4):
    """Classification report por clase (precision, recall, F1, support).

    Se reusa para todos los modelos para que midan lo mismo en el mismo orden
    (E, L, V); zero_division=0 mantiene definida la tabla si una clase nunca
    se predice.
    """
    reporte = classification_report(
        y_true,
        y_pred,
        labels=list(CLASSES),
        target_names=list(CLASSES),
        digits=digits,
        zero_division=0,
    )
    print(f"### {nombre}\n{reporte}")
    return reporte