"""Carga, validacion y limpieza reproducible del dataset de partidos."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/raw/futbol_uruguayo.zip"
DEFAULT_OUTPUT = ROOT / "data/processed/futbol_uruguayo_clean.csv"
DEFAULT_REPORT = ROOT / "data/processed/cleaning_report.json"

REQUIRED_COLUMNS = {
    "home",
    "away",
    "date",
    "gh",
    "ga",
}

UNUSED_COLUMNS = {
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


def _read_source(path: Path) -> pd.DataFrame:
    """Read a CSV or a ZIP containing exactly one CSV."""
    if path.suffix.lower() != ".zip":
        return pd.read_csv(path, encoding="utf-8-sig")

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
            return pd.read_csv(stream, encoding="utf-8-sig")


def clean_matches(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate and clean matches without mutating ``raw``.

    Goal columns are retained only for target construction and auditing. They must
    never be included in a model's input features.
    """
    frame = raw.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas: " + ", ".join(sorted(missing_columns))
        )

    text_columns = frame.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        frame[column] = frame[column].astype("string").str.strip()

    required_missing = frame[list(REQUIRED_COLUMNS)].isna() | (
        frame[list(REQUIRED_COLUMNS)].astype("string") == ""
    )
    if required_missing.any(axis=None):
        bad_rows = (required_missing.any(axis=1)).loc[lambda values: values].index
        raise ValueError(
            "Hay valores faltantes en columnas requeridas; "
            f"primeras filas: {list(bad_rows[:5])}."
        )

    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    for column in ("gh", "ga"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if (~np.isfinite(numeric) | (numeric < 0) | (numeric % 1 != 0)).any():
            raise ValueError(f"{column} debe contener enteros no negativos y finitos.")
        frame[column] = numeric.astype("int64")

    duplicate_rows = int(frame.duplicated(keep="first").sum())
    frame = frame.drop_duplicates(keep="first").copy()

    frame["winner"] = np.select(
        [frame["gh"] > frame["ga"], frame["gh"] < frame["ga"]],
        ["L", "V"],
        default="E",
    )
    frame["year"] = frame["date"].dt.year.astype("int64")
    frame["month"] = frame["date"].dt.month.astype("int64")
    frame = frame.sort_values(
        ["date", "home", "away"], kind="stable"
    ).reset_index(drop=True)
    frame = frame.drop(columns=UNUSED_COLUMNS.intersection(frame.columns))

    target_counts = frame["winner"].value_counts().sort_index()
    report: dict[str, object] = {
        "rows_written": int(len(frame)),
        "exact_duplicates_removed": duplicate_rows,
        "date_range": {
            "min": frame["date"].min().date().isoformat(),
            "max": frame["date"].max().date().isoformat(),
        },
        "target_counts": {key: int(value) for key, value in target_counts.items()},
        "target_percentages": {
            key: round(100 * int(value) / len(frame), 2)
            for key, value in target_counts.items()
        },
        "model_feature_warning": [
            "No usar gh, ga, winner ni full_time como atributos del modelo.",
            (
                "Ajustar codificacion y discretizacion solamente con cada fold "
                "de entrenamiento."
            ),
        ],
    }
    return frame, report


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def prepare_dataset(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the cleaned CSV and its audit report from the raw course file."""
    cleaned, report = clean_matches(_read_source(input_path))
    report = {
        "input": _display_path(input_path),
        "output": _display_path(output_path),
        **report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.assign(date=cleaned["date"].dt.strftime("%Y-%m-%d")).to_csv(
        output_path, index=False
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return cleaned, report


def load_clean_matches(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Load a generated clean CSV and restore its date dtype."""
    return pd.read_csv(path, parse_dates=["date"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, report = prepare_dataset(args.input, args.output, args.report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
