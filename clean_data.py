#!/usr/bin/env python3
"""Create a validated, duplicate-free version of the Uruguay football dataset.

The script accepts either the course ZIP file or a CSV file. It does not modify
the source data. The output retains all source columns for traceability and adds
three useful columns:

* winner: L (home win), V (away win), or E (draw)
* year and month: extracted from the match date

Goal columns remain in the cleaned dataset so the target can be audited. They
must not be used as model features because they reveal the final result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable, TextIO


REQUIRED_COLUMNS = {
    "home",
    "away",
    "date",
    "gh",
    "ga",
    "home_ident",
    "away_ident",
}


def clean_text(value: str) -> str:
    """Normalize surrounding whitespace without changing a team's spelling."""
    return value.strip()


def parse_goal(value: str, column: str, row_number: int) -> int:
    """Return a non-negative whole-number goal count or raise a clear error."""
    try:
        numeric_value = float(value)
    except ValueError as error:
        raise ValueError(
            f"Row {row_number}: {column!r} must be numeric; found {value!r}."
        ) from error

    if not math.isfinite(numeric_value) or numeric_value < 0:
        raise ValueError(
            f"Row {row_number}: {column!r} must be a non-negative finite number."
        )
    if not numeric_value.is_integer():
        raise ValueError(
            f"Row {row_number}: {column!r} must be a whole number; found {value!r}."
        )
    return int(numeric_value)


def target_from_goals(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "L"
    if home_goals < away_goals:
        return "V"
    return "E"


def open_input(input_path: Path) -> tuple[TextIO, zipfile.ZipFile | None]:
    """Open a CSV directly or the single CSV stored in a ZIP archive."""
    if input_path.suffix.lower() != ".zip":
        return input_path.open("r", encoding="utf-8-sig", newline=""), None

    archive = zipfile.ZipFile(input_path)
    csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if len(csv_members) != 1:
        archive.close()
        raise ValueError(
            "The ZIP must contain exactly one CSV file; "
            f"found {len(csv_members)}: {csv_members}."
        )
    return (
        archive.open(csv_members[0], "r"),  # type: ignore[return-value]
        archive,
    )


def text_stream(binary_or_text_stream: TextIO) -> Iterable[str]:
    """Yield decoded text for both a regular CSV and a CSV opened from a ZIP."""
    for line in binary_or_text_stream:
        if isinstance(line, bytes):
            yield line.decode("utf-8-sig")
        else:
            yield line


def clean_dataset(input_path: Path, output_path: Path, report_path: Path) -> dict:
    source_stream, archive = open_input(input_path)
    try:
        reader = csv.DictReader(text_stream(source_stream))
        if reader.fieldnames is None:
            raise ValueError("The input CSV has no header row.")

        source_columns = [clean_text(column) for column in reader.fieldnames]
        missing_columns = REQUIRED_COLUMNS.difference(source_columns)
        if missing_columns:
            raise ValueError(
                "Missing required columns: " + ", ".join(sorted(missing_columns))
            )

        output_columns = [*source_columns, "winner", "year", "month"]
        seen_rows: set[tuple[str, ...]] = set()
        cleaned_rows: list[dict[str, str | int]] = []
        duplicate_rows = 0
        class_counts: Counter[str] = Counter()
        years: list[int] = []

        for row_number, raw_row in enumerate(reader, start=2):
            row = {
                column: clean_text(raw_row.get(original_column, ""))
                for column, original_column in zip(source_columns, reader.fieldnames)
            }

            missing_values = [
                column for column in REQUIRED_COLUMNS if not row[column]
            ]
            if missing_values:
                raise ValueError(
                    f"Row {row_number}: missing required values in {missing_values}."
                )

            try:
                match_date = date.fromisoformat(row["date"])
            except ValueError as error:
                raise ValueError(
                    f"Row {row_number}: invalid ISO date {row['date']!r}."
                ) from error

            home_goals = parse_goal(row["gh"], "gh", row_number)
            away_goals = parse_goal(row["ga"], "ga", row_number)
            row["gh"] = str(home_goals)
            row["ga"] = str(away_goals)

            row_key = tuple(row[column] for column in source_columns)
            if row_key in seen_rows:
                duplicate_rows += 1
                continue
            seen_rows.add(row_key)

            winner = target_from_goals(home_goals, away_goals)
            row["winner"] = winner
            row["year"] = match_date.year
            row["month"] = match_date.month
            cleaned_rows.append(row)
            class_counts[winner] += 1
            years.append(match_date.year)

        if not cleaned_rows:
            raise ValueError("No valid rows remained after cleaning.")

        cleaned_rows.sort(
            key=lambda row: (str(row["date"]), str(row["home_ident"]), str(row["away_ident"]))
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=output_columns)
            writer.writeheader()
            writer.writerows(cleaned_rows)

        report = {
            "input": str(input_path),
            "output": str(output_path),
            "rows_written": len(cleaned_rows),
            "exact_duplicates_removed": duplicate_rows,
            "date_range": {"min": min(row["date"] for row in cleaned_rows), "max": max(row["date"] for row in cleaned_rows)},
            "target_counts": dict(sorted(class_counts.items())),
            "target_percentages": {
                label: round(100 * count / len(cleaned_rows), 2)
                for label, count in sorted(class_counts.items())
            },
            "model_feature_warning": [
                "Do not use gh, ga, winner, or full_time as model features.",
                "Use transformations such as encoding and discretization inside a training pipeline.",
            ],
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return report
    finally:
        source_stream.close()
        if archive is not None:
            archive.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("futbol_uruguayo.zip"),
        help="Course ZIP or CSV input file (default: futbol_uruguayo.zip).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/futbol_uruguayo_clean.csv"),
        help="Cleaned CSV output path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/cleaning_report.json"),
        help="JSON report output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = clean_dataset(args.input, args.output, args.report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Cleaning failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
