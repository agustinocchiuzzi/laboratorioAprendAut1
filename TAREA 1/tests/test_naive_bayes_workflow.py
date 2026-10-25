"""Regression checks for NB preprocessing and validation-selected inputs."""

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
from src.features import NUMERIC_FEATURES
from src.naive_bayes import MEstimateCategoricalNB

EXTRA = [
    "home_points_per_match_5", "away_points_per_match_5",
    "home_goal_diff_per_match_5", "away_goal_diff_per_match_5",
]


def code_cells(path):
    notebook = json.loads(path.read_text())
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def definition(name):
    for source in code_cells(ROOT / "entrega/notebook.ipynb"):
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


class NaiveBayesWorkflowTest(unittest.TestCase):
    def test_source_and_delivery_use_fixed_cuts_only_for_recent_win_rates(self):
        namespace = dict(np=np, pd=pd, NUMERIC_FEATURES=NUMERIC_FEATURES,
                         LAST_5_CUTS=evaluation.LAST_5_CUTS)
        exec(definition("MixedTypeDiscretizer"), namespace)
        exec(definition("new_discretizer"), namespace)
        frame = fixture()
        for columns in (NUMERIC_FEATURES, NUMERIC_FEATURES + EXTRA):
            for factory in (evaluation.new_discretizer, namespace["new_discretizer"]):
                with self.subTest(columns=len(columns), factory=factory):
                    disc = factory(columns).fit(frame[columns])
                    for column in columns:
                        expected = ([0.3, 0.6] if column in NUMERIC_FEATURES[:2]
                                    else np.quantile(frame[column], [1/3, 2/3]))
                        np.testing.assert_allclose(disc.numeric_edges_[column], expected)
                    edges = copy.deepcopy(disc.numeric_edges_)
                    disc.transform(frame[columns] + 100)
                    for column in columns:
                        np.testing.assert_array_equal(disc.numeric_edges_[column], edges[column])

    def test_default_discretizer_preserves_existing_six_feature_workflow(self):
        frame = fixture()
        default = evaluation.new_discretizer().fit(frame)
        explicit = evaluation.new_discretizer(NUMERIC_FEATURES).fit(frame)
        self.assertEqual(default.numeric_features, NUMERIC_FEATURES)
        np.testing.assert_array_equal(default.transform(frame), explicit.transform(frame))

    def test_feature_validation_fits_only_the_training_rows_of_each_fold(self):
        frame = fixture()
        frame.loc[frame.date.dt.year.eq(2023), EXTRA] = 100
        folds = evaluation.make_temporal_folds(frame)
        observed = []
        factory = evaluation.new_discretizer

        def recording_factory(columns):
            disc = factory(columns)
            original_fit = disc.fit

            def fit(data, y=None):
                observed.append((list(data.index), list(data.columns)))
                return original_fit(data, y)

            disc.fit = fit
            return disc

        columns = NUMERIC_FEATURES + EXTRA
        with patch.object(evaluation, "new_discretizer", recording_factory):
            own = evaluation.evaluate_temporal_cv(
                MEstimateCategoricalNB(m=0.1, min_categories=4), frame, folds,
                "discrete", feature_columns=columns)
        self.assertEqual(observed, [(list(f.train_positions), columns) for f in folds])
        reference = evaluation.evaluate_temporal_cv(
            CategoricalNB(alpha=0.025, min_categories=4, force_alpha=True),
            frame, folds, "discrete", feature_columns=columns)
        pd.testing.assert_frame_equal(own, reference)
        frame.loc[0, "date"] = pd.Timestamp("2024-01-01")
        with self.assertRaises(ValueError):
            evaluation.evaluate_temporal_cv(
                MEstimateCategoricalNB(), frame, folds, "discrete", columns)

    def test_selected_feature_set_reaches_both_final_models_and_ties_prefer_six(self):
        # Execute the actual selection and final-fit block from the delivery.
        source = next(s for s in code_cells(ROOT / "entrega/notebook.ipynb")
                      if s.startswith("columnas_ampliadas ="))
        fit_source = source.split("pred_nb =", 1)[0]
        for six_score, ten_score, expected in [(0.5, 0.4, 6), (0.4, 0.4, 6), (0.4, 0.5, 10)]:
            with self.subTest(six=six_score, ten=ten_score):
                def cv(model, train, folds, representation, feature_columns):
                    score = six_score if len(feature_columns) == 6 else ten_score
                    return pd.DataFrame({"validacion": [2021, 2022, 2023],
                                         "macro_f1": [score] * 3})

                frame = fixture()
                namespace = dict(
                    pd=pd, np=np, NUMERIC_FEATURES=NUMERIC_FEATURES,
                    train=frame, test=frame.iloc[:3], folds=(), nb_m=0.1,
                    evaluate_temporal_cv=cv, display=lambda *args: None,
                    new_discretizer=evaluation.new_discretizer,
                    MEstimateCategoricalNB=MEstimateCategoricalNB,
                    CategoricalNB=CategoricalNB,
                )
                exec(fit_source, namespace)
                self.assertEqual(len(namespace["columnas_final"]), expected)
                self.assertEqual(namespace["Xnb_train"].shape[1], expected)
                self.assertEqual(namespace["Xnb_test"].shape[1], expected)
                self.assertEqual(len(namespace["nb_final"].feature_log_prob_), expected)
                self.assertEqual(namespace["nb_ref"].n_features_in_, expected)
                np.testing.assert_allclose(
                    namespace["nb_final"].predict_proba(namespace["Xnb_test"]),
                    namespace["nb_ref"].predict_proba(namespace["Xnb_test"]),
                    atol=1e-14,
                )

    def test_root_and_delivery_keep_the_same_nb_selection_and_fitting(self):
        blocks = [next(s for s in code_cells(ROOT / path)
                       if s.startswith("columnas_ampliadas ="))
                  for path in ("notebook.ipynb", "entrega/notebook.ipynb")]
        self.assertEqual(blocks[0], blocks[1])


if __name__ == "__main__":
    unittest.main()
