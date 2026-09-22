"""Focused checks for the frozen final evaluation contract."""

import unittest

import numpy as np
import pandas as pd

from src.features import NUMERIC_FEATURES, build_causal_match_features
from src.id3 import ID3


class FinalizedExperimentChecks(unittest.TestCase):
    def test_current_match_and_same_day_scores_do_not_enter_inputs(self):
        matches = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "home": ["A", "C", "A"], "away": ["B", "A", "D"],
            "gh": [3, 2, 1], "ga": [0, 0, 0], "winner": ["L", "V", "L"],
        })
        changed = matches.copy()
        changed.loc[:1, ["gh", "ga"]] = changed.loc[:1, ["ga", "gh"]].to_numpy()
        changed.loc[:1, "winner"] = ["V", "L"]
        before = build_causal_match_features(matches)
        after = build_causal_match_features(changed)
        day_one = before["date"].eq("2024-01-01")
        pd.testing.assert_frame_equal(
            before.loc[day_one, NUMERIC_FEATURES].reset_index(drop=True),
            after.loc[day_one, NUMERIC_FEATURES].reset_index(drop=True),
        )
        self.assertTrue((before.loc[day_one, ["home_prior_matches", "away_prior_matches"]] == 0).all().all())

    def test_id3_stops_and_falls_back_for_an_unseen_category(self):
        X = np.array([[1], [1], [2], [2]])
        y = np.array(["L", "L", "V", "V"])
        self.assertEqual(ID3(min_info_gain=1.0).fit(X, y).get_depth(), 0)
        self.assertEqual(ID3().fit(X, y).predict([[99]])[0], "L")


if __name__ == "__main__":
    unittest.main()
