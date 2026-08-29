from __future__ import annotations

import pandas as pd

from aa_futbol.features import build_causal_match_features


def test_same_day_results_do_not_leak_between_matches() -> None:
    matches = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "home_ident": "A",
                "away_ident": "B",
                "gh": 1,
                "ga": 0,
                "winner": "L",
            },
            {
                "date": "2020-01-01",
                "home_ident": "C",
                "away_ident": "A",
                "gh": 0,
                "ga": 2,
                "winner": "V",
            },
            {
                "date": "2020-01-02",
                "home_ident": "A",
                "away_ident": "B",
                "gh": 0,
                "ga": 0,
                "winner": "E",
            },
        ]
    )

    featured = build_causal_match_features(matches)

    assert featured.loc[0, "home_prior_matches"] == 0
    assert featured.loc[1, "away_prior_matches"] == 0
    assert featured.loc[2, "home_prior_matches"] == 2
    assert featured.loc[2, "home_win_rate_10y"] == 1.0


def test_current_match_goals_do_not_affect_its_features() -> None:
    first = pd.DataFrame(
        [
            {
                "date": "2021-01-01",
                "home_ident": "A",
                "away_ident": "B",
                "gh": 1,
                "ga": 0,
                "winner": "L",
            }
        ]
    )
    changed = first.assign(gh=10, ga=9)

    left = build_causal_match_features(first).drop(columns="winner")
    right = build_causal_match_features(changed).drop(columns="winner")

    pd.testing.assert_frame_equal(left, right)
