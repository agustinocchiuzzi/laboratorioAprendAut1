from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score

from aa_futbol.data import DEFAULT_INPUT, prepare_dataset
from aa_futbol.model_selection import DateBlockedTimeSeriesSplit
from aa_futbol.models import TenYearWinRateClassifier


def test_course_dataset_is_rebuilt_from_raw_zip(tmp_path) -> None:
    cleaned, report = prepare_dataset(
        DEFAULT_INPUT,
        tmp_path / "clean.csv",
        tmp_path / "report.json",
    )

    assert len(cleaned) == 15_206
    assert report["exact_duplicates_removed"] == 1
    assert cleaned["date"].min().date().isoformat() == "1932-03-05"
    assert cleaned["date"].max().date().isoformat() == "2025-06-30"
    assert report["target_counts"] == {"E": 4172, "L": 6745, "V": 4289}


def test_baseline_supports_temporal_cross_validation() -> None:
    X = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=9, freq="D"),
            "home": ["A", "B", "C"] * 3,
            "away": ["B", "C", "A"] * 3,
        }
    )
    y = np.asarray(["L", "V", "E"] * 3)
    scorer = make_scorer(
        f1_score,
        average="macro",
        labels=["E", "L", "V"],
        pos_label=None,
        zero_division=0,
    )

    scores = cross_val_score(
        TenYearWinRateClassifier(),
        X,
        y,
        cv=DateBlockedTimeSeriesSplit(n_splits=2),
        scoring=scorer,
    )

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
