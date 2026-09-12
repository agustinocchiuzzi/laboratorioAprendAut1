"""Leakage-safe, causal features computed from matches strictly in the past.

Cuando el denominador es 0 (sin historial previo: debut del equipo, sin partidos
en la ventana, primer partido del ano, primer head-to-head) se imputa un valor
neutro constante distinto de ``0.0`` (ver ``NEUTRAL_WIN_RATE``,
``NEUTRAL_POINTS_PER_MATCH`` y ``NEUTRAL_GOAL_DIFF_PER_MATCH``). Un ``0.0`` real
solo aparece cuando hubo partidos con historial y todos se perdieron.
"""

from __future__ import annotations

import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = {"home", "away", "date", "gh", "ga"}

_UNUSED_COLUMNS = {
    "competition",
    "level",
    "home_country",
    "away_country",
    "home_code",
    "away_code",
    "home_continent",
    "away_continent",
    "continent",
    "home_ident",
    "away_ident",
}


def load_clean_matches(source: str | Path) -> pd.DataFrame:
    """Read the original ZIP (one CSV) and return a cleaned matches frame.

    ``gh``/``ga``/``winner`` are kept only for target construction and auditing;
    they must never enter a model as input attributes.
    """
    path = Path(source)
    if path.suffix.lower() != ".zip":
        raise ValueError(f"Se espera un archivo ZIP; recibido: {path.name}.")
    with zipfile.ZipFile(path) as archive:
        csv_members = [
            name for name in archive.namelist() if name.lower().endswith(".csv")
        ]
        if len(csv_members) != 1:
            raise ValueError(
                "El ZIP debe contener exactamente un CSV; "
                f"se encontraron {len(csv_members)}: {csv_members}."
            )
        with archive.open(csv_members[0]) as stream:
            raw = pd.read_csv(stream, encoding="utf-8-sig")
    return _clean_matches(raw)


def _clean_matches(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    missing_columns = _REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas: " + ", ".join(sorted(missing_columns))
        )

    text_columns = frame.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        frame[column] = frame[column].astype("string").str.strip()

    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    for column in ("gh", "ga"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if (~np.isfinite(numeric) | (numeric < 0) | (numeric % 1 != 0)).any():
            raise ValueError(
                f"{column} debe contener enteros no negativos y finitos."
            )
        frame[column] = numeric.astype("int64")

    frame = frame.drop_duplicates(keep="first").reset_index(drop=True)
    frame["winner"] = np.select(
        [frame["gh"] > frame["ga"], frame["gh"] < frame["ga"]],
        ["L", "V"],
        default="E",
    )
    frame["year"] = frame["date"].dt.year.astype("int64")
    frame["month"] = frame["date"].dt.month.astype("int64")
    frame = frame.sort_values(["date", "home", "away"], kind="stable").reset_index(
        drop=True
    )
    return frame.drop(columns=_UNUSED_COLUMNS.intersection(frame.columns))


CATEGORICAL_FEATURES = []

# Atributos numericos completos que calcula build_causal_match_features.
# Se usan para auditoria y para los datasets procesados; el modelado usa solo
# el subconjunto definido en NUMERIC_FEATURES.
_ALL_COMPUTED_NUMERIC_FEATURES = [
    "home_prior_matches",
    "away_prior_matches",
    "home_win_rate_all",
    "away_win_rate_all",
    "home_win_rate_10y",
    "away_win_rate_10y",
    "home_win_rate_last_5",
    "away_win_rate_last_5",
    "home_win_rate_season",
    "away_win_rate_season",
    "home_win_rate_as_home_all",
    "home_win_rate_h2h_as_home",
    "home_points_per_match_5",
    "away_points_per_match_5",
    "home_goal_diff_per_match_5",
    "away_goal_diff_per_match_5",
    "win_rate_10y_diff",
    "form_points_diff",
    "form_goal_diff_diff",
]

# Conjunto con el que se modela: tasas de victoria (local y visitante donde
# existen). No se usan nombres de equipos ni mes como atributos.
NUMERIC_FEATURES = [
    "home_win_rate_last_5",
    "away_win_rate_last_5",
    "home_win_rate_season",
    "away_win_rate_season",
    "home_win_rate_as_home_all",
    "home_win_rate_h2h_as_home",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Imputacion para denominador = 0 (sin historial disponible). Un valor neutro
# distinto de 0.0 evita que el modelo confunda "nunca jugo" con "siempre perdio".
NEUTRAL_WIN_RATE = 0.5
NEUTRAL_POINTS_PER_MATCH = 4.0 / 3.0
NEUTRAL_GOAL_DIFF_PER_MATCH = 0.0


@dataclass
class _TeamState:
    matches: int = 0
    wins: int = 0
    last_ten_years: deque[tuple[pd.Timestamp, int]] = field(default_factory=deque)
    last_five: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=5))
    last_five_wins: deque[int] = field(default_factory=lambda: deque(maxlen=5))

    def snapshot(self, match_date: pd.Timestamp) -> dict[str, float]:
        cutoff = match_date - pd.DateOffset(years=10)
        while self.last_ten_years and self.last_ten_years[0][0] < cutoff:
            self.last_ten_years.popleft()

        wins_10y = sum(win for _, win in self.last_ten_years)
        games_10y = len(self.last_ten_years)
        form_games = len(self.last_five)
        return {
            "prior_matches": float(self.matches),
            "win_rate_all": (
                self.wins / self.matches if self.matches else NEUTRAL_WIN_RATE
            ),
            "win_rate_10y": (
                wins_10y / games_10y if games_10y else NEUTRAL_WIN_RATE
            ),
            "win_rate_last_5": (
                sum(self.last_five_wins) / len(self.last_five_wins)
                if self.last_five_wins
                else NEUTRAL_WIN_RATE
            ),
            "points_per_match_5": (
                sum(points for points, _ in self.last_five) / form_games
                if form_games
                else NEUTRAL_POINTS_PER_MATCH
            ),
            "goal_diff_per_match_5": (
                sum(goal_difference for _, goal_difference in self.last_five)
                / form_games
                if form_games
                else NEUTRAL_GOAL_DIFF_PER_MATCH
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
        self.last_five_wins.append(int(won))


def build_causal_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Return one modeling row per match using only earlier match dates.

    All matches played on the same date are featurized before any result from that
    date is added to team history. During 2024-2025 evaluation, results of earlier
    dates are therefore available, which models the realistic online prediction
    setting without ever using the current or future match result.
    """
    required = {"date", "home", "away", "gh", "ga", "winner"}
    missing = required.difference(matches.columns)
    if missing:
        raise ValueError("Faltan columnas para crear atributos: " + ", ".join(missing))

    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values(
        ["date", "home", "away"], kind="stable"
    ).reset_index(drop=True)

    histories: defaultdict[str, _TeamState] = defaultdict(_TeamState)
    season_histories: defaultdict[tuple[str, int], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    home_histories: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    head_to_head_histories: defaultdict[tuple[str, str], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    feature_rows: list[dict[str, object]] = []

    for match_date, same_day in frame.groupby("date", sort=True):
        pending_updates: list[pd.Series] = []
        for _, match in same_day.iterrows():
            home_name = str(match["home"])
            away_name = str(match["away"])
            year = int(match_date.year)
            home = histories[home_name].snapshot(match_date)
            away = histories[away_name].snapshot(match_date)
            home_season_matches, home_season_wins = season_histories[(home_name, year)]
            away_season_matches, away_season_wins = season_histories[(away_name, year)]
            home_matches, home_wins = home_histories[home_name]
            h2h_matches, h2h_home_wins = head_to_head_histories[
                (home_name, away_name)
            ]
            feature_rows.append(
                {
                    "date": match_date,
                    "year": year,
                    "month": int(match_date.month),
                    "home": home_name,
                    "away": away_name,
                    "home_prior_matches": home["prior_matches"],
                    "away_prior_matches": away["prior_matches"],
                    "home_win_rate_all": home["win_rate_all"],
                    "away_win_rate_all": away["win_rate_all"],
                    "home_win_rate_10y": home["win_rate_10y"],
                    "away_win_rate_10y": away["win_rate_10y"],
                    "home_win_rate_last_5": home["win_rate_last_5"],
                    "away_win_rate_last_5": away["win_rate_last_5"],
                    "home_win_rate_season": (
                        home_season_wins / home_season_matches
                        if home_season_matches
                        else NEUTRAL_WIN_RATE
                    ),
                    "away_win_rate_season": (
                        away_season_wins / away_season_matches
                        if away_season_matches
                        else NEUTRAL_WIN_RATE
                    ),
                    "home_win_rate_as_home_all": (
                        home_wins / home_matches
                        if home_matches
                        else NEUTRAL_WIN_RATE
                    ),
                    "home_win_rate_h2h_as_home": (
                        h2h_home_wins / h2h_matches
                        if h2h_matches
                        else NEUTRAL_WIN_RATE
                    ),
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
            histories[str(match["home"])].update(
                match_date,
                won=winner == "L",
                points=home_points,
                goals_for=int(match["gh"]),
                goals_against=int(match["ga"]),
            )
            histories[str(match["away"])].update(
                match_date,
                won=winner == "V",
                points=away_points,
                goals_for=int(match["ga"]),
                goals_against=int(match["gh"]),
            )
            year = int(match_date.year)
            home_name = str(match["home"])
            away_name = str(match["away"])
            season_histories[(home_name, year)][0] += 1
            season_histories[(home_name, year)][1] += int(winner == "L")
            season_histories[(away_name, year)][0] += 1
            season_histories[(away_name, year)][1] += int(winner == "V")
            home_histories[home_name][0] += 1
            home_histories[home_name][1] += int(winner == "L")
            head_to_head_histories[(home_name, away_name)][0] += 1
            head_to_head_histories[(home_name, away_name)][1] += int(winner == "L")

    featured = pd.DataFrame(feature_rows)
    numeric_frame = featured[_ALL_COMPUTED_NUMERIC_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(numeric_frame).all():
        raise ValueError("Los atributos numericos contienen valores no finitos.")
    return featured
