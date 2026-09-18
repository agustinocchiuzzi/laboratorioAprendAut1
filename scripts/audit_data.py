"""Reproduce the audit only: no features, model fits or experiments.

Run: python scripts/audit_data.py
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import audit_match_records, load_raw_matches  # noqa: E402


def main():
    source = ROOT / "data/raw/futbol_uruguayo.zip"
    audit = audit_match_records(load_raw_matches(source))
    output = ROOT / "docs/data_audit"
    output.mkdir(parents=True, exist_ok=True)
    flagged = ~audit["included"] | audit["team_date_warning"]
    audit.loc[flagged].to_csv(output / "flagged_records.csv", index=False, date_format="%Y-%m-%d")
    counts = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "raw_rows": len(audit),
        "exact_duplicate_rows_removed": int(audit.exact_duplicate.sum()),
        "ambiguous_fixture_rows_removed": int(audit.ambiguous_fixture.sum()),
        "extra_time_rows_removed": int(audit.full_time.eq("E").sum()),
        "penalty_rows_removed": int(audit.full_time.eq("P").sum()),
        "team_date_warning_rows_before_exclusions": int(audit.team_date_warning.sum()),
        "retained_team_date_warning_rows": int((audit.team_date_warning & audit.included).sum()),
        "included_rows": int(audit.included.sum()),
        "train_rows": int((audit.included & audit.date.dt.year.le(2023)).sum()),
        "test_rows": int((audit.included & audit.date.dt.year.between(2024, 2025)).sum()),
    }
    (output / "summary.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
