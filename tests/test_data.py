from __future__ import annotations

import pandas as pd

from aa_futbol.data import clean_matches


def test_clean_matches_adds_target_and_removes_exact_duplicates() -> None:
    raw = pd.DataFrame(
        [
            {
                "home": " A ",
                "away": "B",
                "date": "2023-01-01",
                "gh": 2.0,
                "ga": 1.0,
            },
            {
                "home": " A ",
                "away": "B",
                "date": "2023-01-01",
                "gh": 2.0,
                "ga": 1.0,
            },
            {
                "home": "B",
                "away": "A",
                "date": "2023-01-02",
                "gh": 0,
                "ga": 0,
            },
        ]
    )

    cleaned, report = clean_matches(raw)

    assert cleaned["winner"].tolist() == ["L", "E"]
    assert cleaned["home"].tolist() == ["A", "B"]
    assert report["exact_duplicates_removed"] == 1
