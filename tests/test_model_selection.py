from __future__ import annotations

import pandas as pd

from aa_futbol.model_selection import DateBlockedTimeSeriesSplit


def test_split_is_chronological_and_date_blocked() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                    "2020-01-04",
                ]
            )
        }
    )
    splitter = DateBlockedTimeSeriesSplit(n_splits=2)

    for train_index, validation_index in splitter.split(frame):
        train_dates = set(frame.loc[train_index, "date"])
        validation_dates = set(frame.loc[validation_index, "date"])
        assert train_dates.isdisjoint(validation_dates)
        assert max(train_dates) < min(validation_dates)
