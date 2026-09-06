"""Assignment baseline based on each team's win rate in the preceding decade."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class TenYearWinRateClassifier(ClassifierMixin, BaseEstimator):
    """Predict that the team with the larger ten-year win proportion wins.

    The estimator deliberately never predicts a draw, matching the assignment's
    baseline wording. Equal rates are resolved deterministically in favor of the
    home team by default.
    """

    def __init__(self, tie_break: str = "home") -> None:
        self.tie_break = tie_break

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TenYearWinRateClassifier requiere un pandas DataFrame.")
        required = {"date", "home", "away"}
        missing = required.difference(X.columns)
        if missing:
            raise ValueError("Faltan atributos: " + ", ".join(sorted(missing)))
        frame = X.loc[:, ["date", "home", "away"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        return frame

    def fit(self, X: pd.DataFrame, y):
        if self.tie_break not in {"home", "away"}:
            raise ValueError("tie_break debe ser 'home' o 'away'.")
        frame = self._validate_frame(X).reset_index(drop=True)
        target = np.asarray(y)
        if len(frame) != len(target):
            raise ValueError("X e y deben tener la misma cantidad de filas.")
        if not np.isin(target, ["E", "L", "V"]).all():
            raise ValueError("y solo puede contener E, L y V.")

        events: dict[str, list[tuple[np.datetime64, int]]] = {}
        for match, winner in zip(  # noqa: B905
            frame.itertuples(index=False), target
        ):
            match_date = np.datetime64(match.date, "D")
            events.setdefault(str(match.home), []).append(
                (match_date, int(winner == "L"))
            )
            events.setdefault(str(match.away), []).append(
                (match_date, int(winner == "V"))
            )

        self.histories_ = {}
        for team, team_events in events.items():
            team_events.sort(key=lambda event: event[0])
            dates = np.asarray(
                [event[0] for event in team_events], dtype="datetime64[D]"
            )
            wins = np.asarray([event[1] for event in team_events], dtype=np.int64)
            self.histories_[team] = (dates, np.concatenate(([0], np.cumsum(wins))))

        self.classes_ = np.asarray(sorted(np.unique(target)))
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def _win_rate(self, team: str, match_date: pd.Timestamp) -> float:
        history = self.histories_.get(team)
        if history is None:
            return 0.0
        dates, cumulative_wins = history
        cutoff = np.datetime64(match_date - pd.DateOffset(years=10), "D")
        current = np.datetime64(match_date, "D")
        left = int(np.searchsorted(dates, cutoff, side="left"))
        right = int(np.searchsorted(dates, current, side="left"))
        games = right - left
        if games == 0:
            return 0.0
        wins = int(cumulative_wins[right] - cumulative_wins[left])
        return wins / games

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "histories_")
        frame = self._validate_frame(X)
        predictions: list[str] = []
        for match in frame.itertuples(index=False):
            home_rate = self._win_rate(str(match.home), match.date)
            away_rate = self._win_rate(str(match.away), match.date)
            if home_rate > away_rate:
                predictions.append("L")
            elif away_rate > home_rate:
                predictions.append("V")
            else:
                predictions.append("L" if self.tie_break == "home" else "V")
        return np.asarray(predictions)
