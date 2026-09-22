"""Regression checks for validation selection and the disabled test boundary."""
import ast
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import evaluation
from src.features import NUMERIC_FEATURES, _normalize_matches
from src.naive_bayes import MEstimateCategoricalNB

EXTRA = [
    "home_points_per_match_5", "away_points_per_match_5",
    "home_goal_diff_per_match_5", "away_goal_diff_per_match_5",
]
NB = json.loads((ROOT / "entrega/notebook.ipynb").read_text())
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]


def definition(name):
    for source in CODE:
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
                return ast.get_source_segment(source, node)
    raise AssertionError(name)


def fixture():
    frame = pd.DataFrame({
        name: np.tile([0.1, 0.2, 0.4, 0.7, 0.8, 0.9], 4)
        for name in NUMERIC_FEATURES + EXTRA
    })
    frame["date"] = pd.to_datetime([
        f"{year}-01-{day:02d}" for year in (2020, 2021, 2022, 2023)
        for day in range(1, 7)
    ])
    frame["winner"] = np.tile(["E", "L", "V"], 8)
    return frame


class ExperimentalConsistencyTest(unittest.TestCase):
    def test_score_validation_rejects_invalid_values_in_both_workflows(self):
        namespace = dict(np=np, pd=pd)
        exec(definition("_normalize_matches"), namespace)
        for normalize in (_normalize_matches, namespace["_normalize_matches"]):
            for column in ("gh", "ga"):
                for value in (-1, 1.9, np.nan, np.inf, -np.inf, "invalid"):
                    with self.subTest(column=column, value=value, normalize=normalize):
                        raw = pd.DataFrame({"date": ["2024-01-01"], "gh": [2], "ga": [0]})
                        raw[column] = value
                        with self.assertRaises((ValueError, TypeError)):
                            normalize(raw)

    def test_score_validation_preserves_valid_integer_strings(self):
        namespace = dict(np=np, pd=pd)
        exec(definition("_normalize_matches"), namespace)
        raw = pd.DataFrame({"date": ["2024-01-01"], "gh": ["2"], "ga": ["0"]})
        for normalize in (_normalize_matches, namespace["_normalize_matches"]):
            normalized = normalize(raw)
            self.assertEqual(normalized["gh"].tolist(), [2])
            self.assertEqual(normalized["ga"].tolist(), [0])
            self.assertEqual(str(normalized["gh"].dtype), "int64")

    def test_notebook_and_source_discretizers_match_training_quantiles(self):
        namespace = dict(np=np, pd=pd, NUMERIC_FEATURES=NUMERIC_FEATURES,
                         LAST_5_CUTS=evaluation.LAST_5_CUTS)
        exec(definition("MixedTypeDiscretizer"), namespace)
        exec(definition("new_discretizer"), namespace)
        frame = fixture()
        for columns in (NUMERIC_FEATURES, NUMERIC_FEATURES + EXTRA):
            left = evaluation.new_discretizer(columns).fit(frame)
            right = namespace["new_discretizer"](columns).fit(frame)
            for column in columns:
                expected = ([0.3, 0.6] if column in NUMERIC_FEATURES[:2]
                            else np.quantile(frame[column], [1/3, 2/3]))
                np.testing.assert_allclose(left.numeric_edges_[column], expected)
                np.testing.assert_allclose(right.numeric_edges_[column], expected)
            edges = copy.deepcopy(left.numeric_edges_)
            shifted = frame.copy()
            shifted[columns] += 20
            np.testing.assert_array_equal(left.transform(shifted), right.transform(shifted))
            for column in columns:
                np.testing.assert_array_equal(left.numeric_edges_[column], edges[column])

    def test_cv_fits_only_each_fold_training_rows(self):
        frame = fixture()
        frame.loc[frame.date.dt.year.eq(2023), EXTRA] = 100
        folds = evaluation.make_temporal_folds(frame)
        observed = []
        factory = evaluation.new_discretizer

        def recording_factory(columns):
            disc = factory(columns)
            fit = disc.fit

            def recording_fit(data, y=None):
                observed.append(data.index.tolist())
                return fit(data, y)
            disc.fit = recording_fit
            return disc

        with patch.object(evaluation, "new_discretizer", recording_factory):
            own = evaluation.evaluate_temporal_cv(
                MEstimateCategoricalNB(m=0.1, min_categories=4),
                frame, folds, "discrete", NUMERIC_FEATURES + EXTRA)
        self.assertEqual(observed, [list(f.train_positions) for f in folds])
        ref = evaluation.evaluate_temporal_cv(
            CategoricalNB(alpha=0.025, min_categories=4, force_alpha=True),
            frame, folds, "discrete", NUMERIC_FEATURES + EXTRA)
        pd.testing.assert_frame_equal(own, ref)
        polluted = frame.copy()
        polluted.loc[0, "date"] = pd.Timestamp("2024-01-01")
        with self.assertRaises(ValueError):
            evaluation.evaluate_temporal_cv(
                MEstimateCategoricalNB(), polluted, folds, "discrete")

    def test_six_features_can_win_and_exact_ties_prefer_six(self):
        namespace = {}
        exec(definition("select_feature_columns"), namespace)
        variants = {"six": NUMERIC_FEATURES, "ten": NUMERIC_FEATURES + EXTRA}
        for six_score, ten_score, expected in [(0.5, 0.4, 6), (0.4, 0.4, 6), (0.4, 0.5, 10)]:
            rows = [
                dict(modelo=model, variante=variant, n_features=n,
                     tie_rank=rank, macro_f1=score)
                for model in ("ID3", "NB propio", "sklearn CategoricalNB")
                for variant, n, rank, score in [
                    ("ten", 10, 1, ten_score), ("six", 6, 0, six_score)]
            ]
            chosen = namespace["select_feature_columns"](pd.DataFrame(rows), variants)
            self.assertTrue(all(len(columns) == expected for columns in chosen.values()))

    def test_selected_columns_and_forest_parameters_reach_final_fit(self):
        class Recorder:
            def __init__(self, **params):
                self.params = params

            def fit(self, X, y):
                self.shape = X.shape
                return self

        final_source = next(
            source for source in CODE
            if source.startswith("discretizer = new_discretizer(id3_columns)"))
        # Exercise both outcomes, not only the ten-feature winner on real data.
        for id3_columns, nb_columns in [
            (NUMERIC_FEATURES, NUMERIC_FEATURES + EXTRA),
            (NUMERIC_FEATURES + EXTRA, NUMERIC_FEATURES),
        ]:
            frame = fixture()
            frame["home_win_rate_10y"] = 0.5
            frame["away_win_rate_10y"] = 0.5
            namespace = {
                "train": frame, "id3_columns": id3_columns,
                "nb_columns": nb_columns, "nb_ref_columns": nb_columns,
                "new_discretizer": evaluation.new_discretizer,
                "id3_gain": 0.005, "nb_m": 0.1,
                "NUMERIC_FEATURES": NUMERIC_FEATURES,
                "BASELINE_FEATURES": ["home_win_rate_10y", "away_win_rate_10y"],
                "RANDOM_STATE": 42,
                "rf_params": {"max_depth": 8, "min_samples_leaf": 20},
                "ID3": Recorder, "MEstimateCategoricalNB": Recorder,
                "CategoricalNB": Recorder, "DecisionTreeClassifier": Recorder,
                "RandomForestClassifier": Recorder, "TenYearWinRateClassifier": Recorder,
            }
            exec(final_source, namespace)
            self.assertEqual(namespace["arbol_id3"].shape[1], len(id3_columns))
            self.assertEqual(namespace["nb_final"].shape[1], len(nb_columns))
            self.assertEqual(namespace["nb_ref"].shape[1], len(nb_columns))
            rf = namespace["competidores"]["sklearn RandomForest (tasas crudas)"]
            self.assertEqual(rf.shape[1], 6)
            self.assertEqual(rf.params["max_depth"], 8)
            self.assertEqual(rf.params["min_samples_leaf"], 20)
            self.assertEqual(rf.params["random_state"], 42)
            self.assertNotIn("y_test", namespace)

    def test_all_test_cells_are_disabled_without_accessing_test(self):
        guarded = [source for source in CODE if source.startswith("if RUN_FINAL_TEST:")]
        self.assertEqual(len(guarded), 7)
        for source in guarded:
            namespace = {"RUN_FINAL_TEST": False}
            exec(source, namespace)  # No test frame, models or predictions exist here.
            self.assertNotIn("y_test", namespace)
        setup = next(source for source in CODE if "RUN_FINAL_TEST = True" in source)
        self.assertIn("RUN_FINAL_TEST = True", setup)

    def test_shared_cv_and_notebook_workflows_stay_synchronized(self):
        import inspect
        for name in ("new_discretizer", "evaluate_temporal_cv"):
            inline = ast.dump(ast.parse(definition(name)), include_attributes=False)
            shared = ast.dump(ast.parse(inspect.getsource(getattr(evaluation, name))),
                              include_attributes=False)
            self.assertEqual(inline, shared)
        root_nb = json.loads((ROOT / "notebook.ipynb").read_text())
        root_code = ["".join(c["source"]) for c in root_nb["cells"] if c["cell_type"] == "code"]
        # The inline definitions are the only extra code cells in entrega.
        self.assertEqual(root_code[1:], CODE[6:])


if __name__ == "__main__":
    unittest.main()
