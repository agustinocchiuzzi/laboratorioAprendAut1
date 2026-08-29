from __future__ import annotations

import numpy as np
import pandas as pd

from aa_futbol.models import (
    CategoricalDecisionTreeClassifier,
    MEstimateCategoricalNB,
    TenYearWinRateClassifier,
)


def test_ten_year_baseline_picks_team_with_higher_rate() -> None:
    X_train = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "home_ident": ["A", "A", "B"],
            "away_ident": ["B", "C", "C"],
        }
    )
    y_train = np.array(["L", "L", "V"])
    model = TenYearWinRateClassifier().fit(X_train, y_train)
    future = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-01"]),
            "home_ident": ["A", "B"],
            "away_ident": ["B", "A"],
        }
    )

    assert model.predict(future).tolist() == ["L", "V"]


def test_m_estimate_naive_bayes_learns_simple_pattern() -> None:
    X = np.array([[1], [1], [2], [2]])
    y = np.array(["L", "L", "V", "V"])
    model = MEstimateCategoricalNB(m=1.0).fit(X, y)

    assert model.predict([[1], [2]]).tolist() == ["L", "V"]
    np.testing.assert_allclose(model.predict_proba([[1]]).sum(axis=1), 1.0)


def test_id3_respects_information_gain_threshold() -> None:
    X = np.array([[1], [1], [2], [2]])
    y = np.array(["L", "L", "V", "V"])

    split_tree = CategoricalDecisionTreeClassifier(min_info_gain=0.0).fit(X, y)
    stopped_tree = CategoricalDecisionTreeClassifier(min_info_gain=1.0).fit(X, y)

    assert split_tree.predict(X).tolist() == y.tolist()
    assert split_tree.get_depth() == 1
    assert stopped_tree.get_depth() == 0


def test_id3_falls_back_to_node_majority_for_unseen_value() -> None:
    X = np.array([[1], [1], [1], [2]])
    y = np.array(["L", "L", "L", "V"])
    model = CategoricalDecisionTreeClassifier().fit(X, y)

    assert model.predict([[99]]).tolist() == ["L"]
