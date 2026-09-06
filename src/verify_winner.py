#!/usr/bin/env python3
"""Verifica que la columna ``winner`` coincida con los goles de cada partido.

Uso:
    python3 src/verify_winner.py
    python3 src/verify_winner.py ruta/al/archivo.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = (
    ROOT / "data/processed/futbol_uruguayo_hasta_2023.csv",
    ROOT / "data/processed/futbol_uruguayo_2024_2025.csv",
)


def expected_winner(home_goals: int, away_goals: int) -> str:
    """Return L, V, or E according to the final score."""
    if home_goals > away_goals:
        return "L"
    if home_goals < away_goals:
        return "V"
    return "E"


def verify_file(path: Path, max_errors: int = 10) -> tuple[int, list[str]]:
    """Return the number of invalid rows and a sample of diagnostic messages."""
    errors: list[str] = []
    error_count = 0

    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required_columns = {"gh", "ga", "winner"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{path}: faltan columnas requeridas: "
                f"{', '.join(sorted(missing_columns))}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                home_goals = int(row["gh"])
                away_goals = int(row["ga"])
                if home_goals < 0 or away_goals < 0:
                    raise ValueError("los goles deben ser no negativos")
                expected = expected_winner(home_goals, away_goals)
                actual = (row["winner"] or "").strip()
                if actual != expected:
                    raise ValueError(f"winner={actual!r}; debería ser {expected!r}")
            except ValueError as error:
                error_count += 1
                if len(errors) < max_errors:
                    errors.append(f"línea {line_number}: {error}")

    return error_count, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        default=list(DEFAULT_FILES),
        help="CSV a verificar. Por defecto verifica los dos conjuntos separados.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    has_errors = False

    for path in args.files:
        if not path.is_file():
            print(f"ERROR: no existe el archivo {path}")
            has_errors = True
            continue

        error_count, errors = verify_file(path)
        if error_count:
            has_errors = True
            print(f"ERROR: {path}: {error_count} filas inválidas.")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"OK: {path}: winner es correcto en todas las filas.")

    if has_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
