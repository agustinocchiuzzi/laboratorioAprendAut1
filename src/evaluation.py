"""Shared temporal setup; importing this module does not fit any model.

Inputs must come from the audited causal feature builder. Histories update after
whole match dates; learned preprocessing and estimators stay fixed in each fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

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
    train_positions: tuple[int, ...]
    validation_positions: tuple[int, ...]


def _dates(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Se requieren fechas de partido completas, sin horas ni faltantes.")
    return dates


def make_temporal_folds(frame: pd.DataFrame) -> tuple[TemporalFold, ...]:
    """Expanding training before each complete year 2021, 2022 and 2023.

    Positions refer to the supplied frame, irrespective of its index/order.
    Reject final evaluation rows instead of silently allowing them in CV.
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


def temporal_holdout(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = _dates(frame)
    if dates.dt.year.gt(2025).any():
        raise ValueError("Hay fechas posteriores al horizonte de la consigna.")
    train = frame.loc[dates.dt.year.le(2023)].reset_index(drop=True)
    test = frame.loc[dates.dt.year.between(2024, 2025)].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Entrenamiento y evaluacion deben contener partidos.")
    return train, test


def new_discretizer() -> MixedTypeDiscretizer:
    """Return a fresh, unfitted transformer for each training partition."""
    return MixedTypeDiscretizer(
        categorical_features=[], numeric_features=NUMERIC_FEATURES,
        n_bins=3, fixed_cuts={column: LAST_5_CUTS for column in NUMERIC_FEATURES[:2]},
    )


def make_model_pipeline(estimator, representation: str) -> Pipeline:
    """Clone the estimator and select only allowed inputs, never labels/dates.

    discrete: ID3/categorical NB; continuous: reference trees/forest;
    baseline: shared causal ten-year rates, without discretization.
    """
    if representation == "discrete":
        preprocessing = new_discretizer()
    elif representation in {"continuous", "baseline"}:
        columns = NUMERIC_FEATURES if representation == "continuous" else BASELINE_FEATURES
        preprocessing = ColumnTransformer(
            [("inputs", "passthrough", columns)], remainder="drop",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
    else:
        raise ValueError("Representacion desconocida: " + representation)
    return Pipeline([("preprocessing", preprocessing), ("model", clone(estimator))])


def evaluate_temporal_cv(estimator, frame: pd.DataFrame,
                         folds: tuple[TemporalFold, ...], representation: str) -> pd.DataFrame:
    """Evaluate only when explicitly called; use a NEW pipeline in every fold.

    Revalidate positions against the exact frame so a reordered/stale split or
    an ad-hoc fold splitting a match date cannot be used by a single model.
    """
    if folds != make_temporal_folds(frame):
        raise ValueError("Todos los modelos deben usar los folds anuales compartidos.")
    rows = []
    for fold in folds:
        train = frame.iloc[list(fold.train_positions)]
        valid = frame.iloc[list(fold.validation_positions)]
        pipeline = make_model_pipeline(estimator, representation)
        pipeline.fit(train, train["winner"])
        prediction = pipeline.predict(valid)
        train_prediction = pipeline.predict(train)
        rows.append({
            "validacion": fold.validation_year,
            "n_train": len(train), "n_validacion": len(valid),
            "accuracy": accuracy_score(valid["winner"], prediction),
            "validation_error": 1.0 - accuracy_score(valid["winner"], prediction),
            "train_accuracy": accuracy_score(train["winner"], train_prediction),
            "train_error": 1.0 - accuracy_score(train["winner"], train_prediction),
            "train_macro_f1": f1_score(train["winner"], train_prediction,
                                       labels=list(CLASSES), average="macro", zero_division=0),
            "macro_f1": f1_score(valid["winner"], prediction,
                                 labels=list(CLASSES), average="macro", zero_division=0),
        })
    return pd.DataFrame(rows)
