"""Atributos causales sin leakage, calculados con partidos estrictamente
anteriores.

Cuando el denominador es 0 (sin historial previo: debut del equipo, sin
partidos en la ventana, primer partido del ano, primer head-to-head) se
imputa un valor neutro constante distinto de 0.0 (ver NEUTRAL_WIN_RATE,
NEUTRAL_POINTS_PER_MATCH y NEUTRAL_GOAL_DIFF_PER_MATCH). Un 0.0 real solo
aparece cuando hubo partidos con historial y ninguna victoria (puede haber
empates).
"""

import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

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


def load_raw_matches(source):
    """Lee el ZIP original (un CSV) sin modificar los registros.

    gh/ga/winner se usan solo para construir la etiqueta y auditar; nunca
    entran al modelo como atributos.
    """
    path = Path(source)
    with zipfile.ZipFile(path) as archive:
        csv_members = [
            name for name in archive.namelist() if name.lower().endswith(".csv")
        ]
        with archive.open(csv_members[0]) as stream:
            return pd.read_csv(stream, encoding="utf-8-sig")


def load_clean_matches(source):
    """Aplica la politica de auditoria antes de crear targets o historiales."""
    return _clean_matches(load_raw_matches(source))


def _normalize_matches(raw):
    """Normaliza tipos y rechaza goles faltantes, no finitos o no enteros."""
    frame = raw.copy().reset_index(drop=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].astype("string").str.strip()
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    for column in ("gh", "ga"):
        scores = pd.to_numeric(frame[column], errors="raise")
        if (scores.isna().any() or not np.isfinite(scores).all()
                or scores.lt(0).any() or scores.mod(1).ne(0).any()):
            raise ValueError(f"{column} debe contener goles enteros no negativos y finitos.")
        frame[column] = scores.astype("int64")
    return frame


def audit_match_records(raw):
    """Audita cada fila del CSV original (la linea 1 es el encabezado).

    Se mantienen el orden y los goles originales del archivador. Quedan en
    cuarentena todos los partidos duplicados exactos y los ambiguos (mismo par
    de equipos y misma fecha, sin importar el orden local/visitante), mas los
    definidos en alargue o penales (full_time E o P). Que un equipo juegue dos
    veces el mismo dia contra rivales distintos es solo una advertencia, no
    un duplicado.
    """
    frame = _normalize_matches(raw)
    exact_duplicate = frame.duplicated(keep="first")
    unique = frame.loc[~exact_duplicate]

    # Pares de equipos (sin importar el orden) por fecha: si hay mas de un
    # partido con el mismo par, todos son ambiguos, no solo el segundo.
    pares = defaultdict(list)
    for indice, fila in unique.iterrows():
        pareja = tuple(sorted([str(fila["home"]), str(fila["away"])]))
        pares[(fila["date"], pareja)].append(indice)
    ambiguos = {
        indice
        for indices in pares.values()
        for indice in indices
        if len(indices) > 1
    }

    # Mismo equipo dos veces en la misma fecha contra rivales distintos.
    equipos = defaultdict(list)
    for indice, fila in unique.iterrows():
        for equipo in (str(fila["home"]), str(fila["away"])):
            equipos[(fila["date"], equipo)].append(indice)
    warnings = {
        indice
        for indices in equipos.values()
        for indice in indices
        if len(indices) > 1
    }

    audit = frame[["date", "home", "away", "gh", "ga", "full_time"]].copy()
    audit.insert(0, "source_line", frame.index + 2)
    audit["exact_duplicate"] = exact_duplicate
    audit["ambiguous_fixture"] = audit.index.isin(ambiguos)
    audit["extra_time_or_penalties"] = frame["full_time"].isin(["E", "P"])
    audit["team_date_warning"] = audit.index.isin(warnings)
    audit["included"] = ~audit[["exact_duplicate", "ambiguous_fixture",
                                "extra_time_or_penalties"]].any(axis=1)
    audit["reason"] = [
        ";".join(name for name in ("exact_duplicate", "ambiguous_fixture",
                                  "extra_time_or_penalties") if row[name])
        or "included"
        for _, row in audit.iterrows()
    ]
    return audit


def _clean_matches(raw):
    """Filtra la auditada y deriva winner antes de pasar a los atributos."""
    frame = _normalize_matches(raw)
    audit = audit_match_records(raw)
    frame = frame.loc[audit["included"]].copy()
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
# distinto de 0.0 separa falta de historial de cero victorias, pero coincide
# con tasas observadas de 0.5: no es un indicador exclusivo de ausencia.
NEUTRAL_WIN_RATE = 0.5
NEUTRAL_POINTS_PER_MATCH = 4.0 / 3.0
NEUTRAL_GOAL_DIFF_PER_MATCH = 0.0


@dataclass
class _TeamState:
    matches: int = 0
    wins: int = 0
    last_ten_years: deque = field(default_factory=deque)
    last_five: deque = field(default_factory=lambda: deque(maxlen=5))
    last_five_wins: deque = field(default_factory=lambda: deque(maxlen=5))

    def snapshot(self, match_date):
        """Tasas hasta match_date; los partidos de hace mas de 10 anos salen."""
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

    def update(self, match_date, *, won, points, goals_for, goals_against):
        self.matches += 1
        self.wins += int(won)
        self.last_ten_years.append((match_date, int(won)))
        self.last_five.append((points, goals_for - goals_against))
        self.last_five_wins.append(int(won))


def build_causal_match_features(matches):
    """Devuelve una fila de modelado por partido usando solo fechas anteriores.

    Todos los partidos de la misma fecha se procesan antes de que sus
    resultados entren al historial de los equipos, asi un partido jamas usa el
    resultado de otro partido del mismo dia (no hay leakage). Durante la
    evaluacion de 2024-2025 los resultados de fechas anteriores siguen
    disponibles, como en el uso real de prediccion online.
    """
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values(
        ["date", "home", "away"], kind="stable"
    ).reset_index(drop=True)

    histories = defaultdict(_TeamState)
    season_histories = defaultdict(lambda: [0, 0])
    home_histories = defaultdict(lambda: [0, 0])
    head_to_head_histories = defaultdict(lambda: [0, 0])
    feature_rows = []

    for match_date, same_day in frame.groupby("date", sort=True):
        pending_updates = []
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
            home_name = str(match["home"])
            away_name = str(match["away"])
            year = int(match_date.year)
            histories[home_name].update(
                match_date,
                won=winner == "L",
                points=home_points,
                goals_for=int(match["gh"]),
                goals_against=int(match["ga"]),
            )
            histories[away_name].update(
                match_date,
                won=winner == "V",
                points=away_points,
                goals_for=int(match["ga"]),
                goals_against=int(match["gh"]),
            )
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


def features_for_new_match(home_team, away_team, match_date, historical_df):
    """Atributos causales que tendria un partido hipotetico en match_date.

    historical_df debe ser el dataset con gh/ga/winner (no los atributos ya
    calculados). Solo se usan partidos con fecha anterior y el partido se
    procesa igual que cualquier otro en build_causal_match_features, asi da
    exactamente los atributos que recibiria un partido real en esa fecha. Sin
    historial se imputan los neutros habituales (NEUTRAL_WIN_RATE = 0.5, etc.).
    """
    required = {"date", "home", "away", "gh", "ga", "winner"}
    match_date = pd.Timestamp(match_date)
    past = pd.to_datetime(historical_df["date"], errors="raise")
    history = historical_df.loc[past.lt(match_date), list(required)].copy()
    synthetic = pd.DataFrame([{
        "date": match_date,
        "home": str(home_team),
        "away": str(away_team),
        "gh": 0,
        "ga": 0,
        "winner": "E",
    }])
    frame = pd.concat([history, synthetic], ignore_index=True)
    featured = build_causal_match_features(frame)
    row = featured.loc[
        featured["date"].eq(match_date)
        & featured["home"].eq(str(home_team))
        & featured["away"].eq(str(away_team))
    ]
    if len(row) != 1:
        raise ValueError("No se pudieron calcular los atributos del partido hipotetico.")
    return row.iloc[0]


def predict_new_match(home_team, away_team, match_date, historical_df, model,
                      feature_columns=None, discretizer=None):
    """Predice un partido hipotetico con el mismo calculo de atributos de train.

    El modelo debe estar ya entrenado. feature_columns elige los atributos que
    espera el modelo (NUMERIC_FEATURES para ID3, Naive Bayes o Random Forest;
    BASELINE_FEATURES para el clasificador base). Si se pasa discretizer
    (ajustado solo con train), los atributos se discretizan; si no, se pasan
    las tasas crudas, como esperan el base y el random forest.
    """
    row = features_for_new_match(home_team, away_team, match_date, historical_df)
    if feature_columns is None:
        feature_columns = list(NUMERIC_FEATURES)
    input_frame = row[list(feature_columns)].to_frame().T
    X = discretizer.transform(input_frame) if discretizer is not None else input_frame

    classes = getattr(model, "classes_", [])
    predicted = str(model.predict(X)[0])
    result = {
        "features": {column: float(row[column]) for column in feature_columns},
        "prediccion": predicted,
        "probabilidades": None,
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        result["probabilidades"] = {
            str(class_label): float(value)
            for class_label, value in zip(classes, probabilities)
        }
    return result
