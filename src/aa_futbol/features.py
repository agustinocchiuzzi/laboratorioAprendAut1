"""Leakage-safe, causal features computed from matches strictly in the past."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ["home_ident", "away_ident", "month"]
NUMERIC_FEATURES = [
    "home_prior_matches",
    "away_prior_matches",
    "home_win_rate_all",
    "away_win_rate_all",
    "home_win_rate_10y",
    "away_win_rate_10y",
    "home_points_per_match_5",
    "away_points_per_match_5",
    "home_goal_diff_per_match_5",
    "away_goal_diff_per_match_5",
    "win_rate_10y_diff",
    "form_points_diff",
    "form_goal_diff_diff",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass
class _TeamState:
    matches: int = 0
    wins: int = 0
    last_ten_years: deque[tuple[pd.Timestamp, int]] = field(default_factory=deque)
    last_five: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=5))

    def snapshot(self, match_date: pd.Timestamp) -> dict[str, float]:
        cutoff = match_date - pd.DateOffset(years=10)
        while self.last_ten_years and self.last_ten_years[0][0] < cutoff:
            self.last_ten_years.popleft()

        wins_10y = sum(win for _, win in self.last_ten_years)
        games_10y = len(self.last_ten_years)
        form_games = len(self.last_five)
        return {
            "prior_matches": float(self.matches),
            "win_rate_all": self.wins / self.matches if self.matches else 0.0,
            "win_rate_10y": wins_10y / games_10y if games_10y else 0.0,
            "points_per_match_5": (
                sum(points for points, _ in self.last_five) / form_games
                if form_games
                else 0.0
            ),
            "goal_diff_per_match_5": (
                sum(goal_difference for _, goal_difference in self.last_five)
                / form_games
                if form_games
                else 0.0
            ),
        }

    def update(
        self,
        match_date: pd.Timestamp,
        *,
        won: bool,
        points: int,
        goals_for: int,
        goals_against: int,
    ) -> None:
        self.matches += 1
        self.wins += int(won)
        self.last_ten_years.append((match_date, int(won)))
        self.last_five.append((points, goals_for - goals_against))


def build_causal_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Return one modeling row per match using only earlier match dates.

    All matches played on the same date are featurized before any result from that
    date is added to team history. During 2024-2025 evaluation, results of earlier
    dates are therefore available, which models the realistic online prediction
    setting without ever using the current or future match result.
    """
    required = {"date", "home_ident", "away_ident", "gh", "ga", "winner"}
    missing = required.difference(matches.columns)
    if missing:
        raise ValueError("Faltan columnas para crear atributos: " + ", ".join(missing))

    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values(
        ["date", "home_ident", "away_ident"], kind="stable"
    ).reset_index(drop=True)

    histories: defaultdict[str, _TeamState] = defaultdict(_TeamState)
    feature_rows: list[dict[str, object]] = []

    for match_date, same_day in frame.groupby("date", sort=True):
        pending_updates: list[pd.Series] = []
        for _, match in same_day.iterrows():
            home = histories[str(match["home_ident"])].snapshot(match_date)
            away = histories[str(match["away_ident"])].snapshot(match_date)
            feature_rows.append(
                {
                    "date": match_date,
                    "year": int(match_date.year),
                    "month": int(match_date.month),
                    "home_ident": str(match["home_ident"]),
                    "away_ident": str(match["away_ident"]),
                    "home_prior_matches": home["prior_matches"],
                    "away_prior_matches": away["prior_matches"],
                    "home_win_rate_all": home["win_rate_all"],
                    "away_win_rate_all": away["win_rate_all"],
                    "home_win_rate_10y": home["win_rate_10y"],
                    "away_win_rate_10y": away["win_rate_10y"],
                    "home_points_per_match_5": home["points_per_match_5"],
                    "away_points_per_match_5": away["points_per_match_5"],
                    "home_goal_diff_per_match_5": home["goal_diff_per_match_5"],
                    "away_goal_diff_per_match_5": away["goal_diff_per_match_5"],
                    "win_rate_10y_diff": home["win_rate_10y"] - away["win_rate_10y"],
                    "form_points_diff": home["points_per_match_5"]
                    - away["points_per_match_5"],
                    "form_goal_diff_diff": home["goal_diff_per_match_5"]
                    - away["goal_diff_per_match_5"],
                    "winner": str(match["winner"]),
                }
            )
            pending_updates.append(match)

        for match in pending_updates:
            winner = str(match["winner"])
            home_points = 3 if winner == "L" else 1 if winner == "E" else 0
            away_points = 3 if winner == "V" else 1 if winner == "E" else 0
            histories[str(match["home_ident"])].update(
                match_date,
                won=winner == "L",
                points=home_points,
                goals_for=int(match["gh"]),
                goals_against=int(match["ga"]),
            )
            histories[str(match["away_ident"])].update(
                match_date,
                won=winner == "V",
                points=away_points,
                goals_for=int(match["ga"]),
                goals_against=int(match["gh"]),
            )

    featured = pd.DataFrame(feature_rows)
    if not np.isfinite(featured[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Los atributos numericos contienen valores no finitos.")
    return featured
