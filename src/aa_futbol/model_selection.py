"""Temporal cross-validation that never separates matches from the same date."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class DateBlockedTimeSeriesSplit(BaseCrossValidator):
    """Expanding-window CV over contiguous blocks of unique match dates."""

    def __init__(self, n_splits: int = 5, gap_dates: int = 0) -> None:
        if n_splits < 2:
            raise ValueError("n_splits debe ser al menos 2.")
        if gap_dates < 0:
            raise ValueError("gap_dates no puede ser negativo.")
        self.n_splits = n_splits
        self.gap_dates = gap_dates

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if groups is not None:
            dates = pd.to_datetime(pd.Series(groups), errors="raise")
        elif isinstance(X, pd.DataFrame) and "date" in X.columns:
            dates = pd.to_datetime(X["date"], errors="raise").reset_index(drop=True)
        else:
            raise ValueError(
                "X debe ser un DataFrame con columna 'date' o usar groups."
            )

        if not dates.is_monotonic_increasing:
            raise ValueError("Las filas deben estar ordenadas cronologicamente.")

        unique_dates = np.asarray(pd.Index(dates.unique()).sort_values())
        if len(unique_dates) < self.n_splits + 1:
            raise ValueError(
                "No hay suficientes fechas unicas para la cantidad de folds."
            )

        blocks = np.array_split(unique_dates, self.n_splits + 1)
        date_values = dates.to_numpy()
        for validation_block_index in range(1, len(blocks)):
            validation_dates = blocks[validation_block_index]
            validation_start_position = int(
                np.searchsorted(unique_dates, validation_dates[0])
            )
            train_end_position = validation_start_position - self.gap_dates
            train_dates = unique_dates[:train_end_position]
            if len(train_dates) == 0:
                raise ValueError("El gap deja un fold sin datos de entrenamiento.")

            train_index = np.flatnonzero(np.isin(date_values, train_dates))
            validation_index = np.flatnonzero(np.isin(date_values, validation_dates))
            yield train_index, validation_index
