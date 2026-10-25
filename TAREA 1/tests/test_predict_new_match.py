"""Predicting a hypothetical match must reuse the exact causal featurization.

For a real match present in the dataset, ``features_for_new_match`` must return
the same features that ``build_causal_match_features`` computed for that row,
otherwise the model sold as "demo of a new match" would disagree with training.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import (  # noqa: E402
    NUMERIC_FEATURES,
    build_causal_match_features,
    features_for_new_match,
    load_clean_matches,
)

DATA = ROOT / "data/raw/futbol_uruguayo.zip"


class PredictNewMatchFeaturesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.matches = load_clean_matches(DATA)

    def test_features_matches_a_real_row(self):
        featured = build_causal_match_features(self.matches)
        real = featured.sort_values("date").iloc[-1]  # ultima fecha: maximo historial
        fresh = features_for_new_match(
            real["home"], real["away"], real["date"], self.matches,
        )
        for column in NUMERIC_FEATURES:
            self.assertAlmostEqual(
                fresh[column], real[column], places=12,
                msg=f"{column} difiere para el partido {real['home']}-{real['away']}",
            )

    def test_first_everything_neutral(self):
        fresh = features_for_new_match("Un Equipo Nuevo", "Otro Nuevo",
                                       "1990-01-01", self.matches)
        for column in NUMERIC_FEATURES:
            self.assertEqual(fresh[column], 0.5, msg=column)


if __name__ == "__main__":
    unittest.main()