"""Policy and integration checks on synthetic data; no model experiments.

Run from the repository root: python -m unittest discover -s tests -v
"""
import ast
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier

from src.baseline import BASELINE_FEATURES, TenYearWinRateClassifier
from src.evaluation import (TemporalFold, evaluate_temporal_cv, make_model_pipeline,
                            make_temporal_folds, new_discretizer, temporal_holdout)
from src.features import (_clean_matches, audit_match_records,
                          build_causal_match_features, load_clean_matches,
                          load_raw_matches, NUMERIC_FEATURES)
from src.id3 import ID3
from src.naive_bayes import MEstimateCategoricalNB

ROOT = Path(__file__).resolve().parents[1]


def matches(rows):
    return pd.DataFrame(rows, columns=['date', 'home', 'away', 'gh', 'ga', 'full_time'])


def synthetic_features():
    rows = []
    for year in range(2019, 2026):
        rows.extend([(f'{year}-01-01', 'A', 'B', 1, 0, 'F'),
                     (f'{year}-01-01', 'C', 'D', 0, 1, 'F'),
                     (f'{year}-02-01', 'A', 'C', 1, 1, 'F'),
                     (f'{year}-02-02', 'D', 'B', 0, 2, 'F')])
    return build_causal_match_features(_clean_matches(matches(rows)))


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    """Remember fit matrices to detect preprocessing learned from validation."""
    fits = []

    def fit(self, X, y):
        type(self).fits.append(np.array(X, copy=True))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.repeat(self.classes_[0], len(X))


class DataPolicyTests(unittest.TestCase):
    def test_exact_duplicate_and_all_conflicting_variants(self):
        raw = matches([
            ('2020-01-01', 'A', 'B', 1, 0, 'F'),
            ('2020-01-01', 'A', 'B', 1, 0, 'F'),
            ('2020-01-02', 'A', 'B', 1, 0, 'F'),
            ('2020-01-02', 'A', 'B', 2, 0, 'F'),
            ('2020-01-03', 'A', 'C', 1, 0, 'F'),
            ('2020-01-03', 'C', 'A', 0, 1, 'F'),
        ])
        before = raw.copy(deep=True)
        audit = audit_match_records(raw)
        self.assertEqual(audit.exact_duplicate.sum(), 1)
        self.assertEqual(audit.ambiguous_fixture.sum(), 4)
        self.assertEqual(audit.loc[audit.included, 'source_line'].tolist(), [2])
        self.assertEqual(len(_clean_matches(raw)), 1)
        pd.testing.assert_frame_equal(raw, before)
        # Policy must not select a different score when the input is reordered.
        pd.testing.assert_frame_equal(_clean_matches(raw), _clean_matches(raw.iloc[::-1]))

    def test_extra_time_penalties_excluded_without_inventing_scores(self):
        raw = matches([
            ('2020-01-01', 'A', 'B', 3, 2, 'E'),
            ('2020-01-02', 'A', 'B', 1, 1, 'P'),
            ('2020-01-03', 'A', 'B', 5, 4, 'P'),
            ('2020-01-04', 'A', 'B', 1, 1, 'F'),
        ])
        clean = _clean_matches(raw)
        self.assertEqual(clean.winner.tolist(), ['E'])
        self.assertEqual(clean[['gh','ga']].values.tolist(), [[1,1]])
        feature = build_causal_match_features(clean)
        self.assertEqual(feature.home_prior_matches.tolist(), [0])
        self.assertEqual(audit_match_records(raw).gh.tolist(), [3,1,5,1])
        raw.loc[0,'full_time'] = '?'
        with self.assertRaises(ValueError): _clean_matches(raw)

    def test_different_opponents_same_date_are_flagged_not_rewritten(self):
        raw = matches([('2020-01-01', 'A', 'B', 1, 0, 'F'),
                       ('2020-01-01', 'A', 'C', 0, 0, 'F')])
        audit = audit_match_records(raw)
        self.assertTrue(audit.team_date_warning.all())
        self.assertTrue(audit.included.all())
        featured = build_causal_match_features(_clean_matches(raw))
        self.assertTrue((featured[NUMERIC_FEATURES] == .5).all().all())

    def test_real_source_audit_counts_and_targets(self):
        source = ROOT / 'data/raw/futbol_uruguayo.zip'
        raw = load_raw_matches(source)
        audit = audit_match_records(raw)
        self.assertEqual(len(raw), 15207)
        self.assertEqual(audit.exact_duplicate.sum(), 1)
        self.assertEqual(audit.ambiguous_fixture.sum(), 16)
        self.assertEqual(audit.extra_time_or_penalties.sum(), 13)
        clean = load_clean_matches(source)
        self.assertEqual(len(clean), 15177)
        self.assertTrue(clean.full_time.eq('F').all())
        self.assertEqual(sum(clean.year.le(2023)), 14705)
        self.assertEqual(sum(clean.year.between(2024,2025)), 472)
        expected = np.where(clean.gh > clean.ga, 'L', np.where(clean.gh < clean.ga, 'V', 'E'))
        np.testing.assert_array_equal(clean.winner, expected)


class CausalHistoryTests(unittest.TestCase):
    def test_current_same_day_and_future_results_do_not_change_inputs(self):
        raw = matches([
            ('2020-01-01', 'A', 'B', 1, 0, 'F'),
            ('2021-01-01', 'A', 'C', 0, 1, 'F'),
            ('2021-01-01', 'B', 'A', 0, 1, 'F'),
            ('2021-01-02', 'A', 'D', 1, 0, 'F'),
        ])
        first = build_causal_match_features(_clean_matches(raw))
        changed = raw.copy()
        changed.loc[1:, ['gh','ga']] = [4,0]
        second = build_causal_match_features(_clean_matches(changed))
        cols = NUMERIC_FEATURES + BASELINE_FEATURES
        pd.testing.assert_frame_equal(first.iloc[:3][cols], second.iloc[:3][cols])
        self.assertFalse(first.iloc[3][cols].equals(second.iloc[3][cols]))
        prefix = build_causal_match_features(_clean_matches(raw.iloc[:3]))
        pd.testing.assert_frame_equal(first.iloc[:3].reset_index(drop=True), prefix)

    def test_online_baseline_changes_only_after_date_and_predict_is_pure(self):
        raw = matches([
            ('2023-12-31', 'A', 'B', 0, 0, 'F'),
            ('2024-01-01', 'B', 'C', 1, 0, 'F'),
            ('2024-01-01', 'A', 'B', 0, 0, 'F'),
            ('2024-01-02', 'A', 'B', 0, 0, 'F'),
        ])
        f = build_causal_match_features(_clean_matches(raw))
        train, test = temporal_holdout(f)
        model = TenYearWinRateClassifier().fit(train[BASELINE_FEATURES], train.winner)
        ab = test.loc[test.home.eq('A')]
        np.testing.assert_array_equal(model.predict(ab), ['L','V'])
        np.testing.assert_array_equal(model.predict(ab), model.predict(ab))
        np.testing.assert_array_equal(model.predict(ab.iloc[::-1]), ['V','L'])
        # A baseline is prohibited from silently falling back to a frozen history.
        with self.assertRaises(ValueError): model.predict(ab[['date','home','away']])

    def test_ten_year_window_includes_boundary_excludes_older_and_current(self):
        f = build_causal_match_features(_clean_matches(matches([
            ('2013-12-31', 'A', 'B', 1, 0, 'F'),
            ('2014-01-01', 'A', 'B', 0, 1, 'F'),
            ('2024-01-01', 'A', 'B', 1, 0, 'F'),
            ('2024-01-02', 'A', 'B', 0, 0, 'F'),
        ])))
        self.assertEqual(f.iloc[2].home_win_rate_10y, 0.)
        self.assertEqual(f.iloc[2].away_win_rate_10y, 1.)
        self.assertEqual(f.iloc[3].home_win_rate_10y, 1.)
        self.assertEqual(f.iloc[0].home_win_rate_10y, .5)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.train, self.test = temporal_holdout(synthetic_features())

    def test_common_folds_cover_complete_dates_with_shuffled_nonunique_index(self):
        frame = self.train.sample(frac=1, random_state=42)
        frame.index = [0] * len(frame)
        for fold in make_temporal_folds(frame):
            train = frame.iloc[list(fold.train_positions)]
            valid = frame.iloc[list(fold.validation_positions)]
            self.assertLess(train.date.max(), valid.date.min())
            self.assertFalse(set(train.date) & set(valid.date))
            self.assertTrue(valid.year.eq(fold.validation_year).all())
            self.assertEqual(set(fold.validation_positions),
                             set(np.flatnonzero(frame.year.eq(fold.validation_year))))
        with self.assertRaises(ValueError): make_temporal_folds(pd.concat([self.train,self.test]))
        with self.assertRaises(ValueError): make_temporal_folds(self.train[self.train.year.ne(2022)])

    def test_stale_or_partial_fold_rejected(self):
        folds = make_temporal_folds(self.train)
        bad = (TemporalFold(2021, folds[0].train_positions, folds[0].validation_positions[:1]),) + folds[1:]
        with self.assertRaises(ValueError):
            evaluate_temporal_cv(ID3(), self.train, bad, 'discrete')
        with self.assertRaises(ValueError):
            evaluate_temporal_cv(ID3(), self.train.iloc[::-1], folds, 'discrete')

    def test_preprocessing_is_fitted_on_each_training_block(self):
        f = self.train.copy()
        # Training quantiles differ strongly between successive years.
        for column in NUMERIC_FEATURES:
            f[column] = np.where(f.year < 2021, .1, np.where(f.year == 2021, .6, .9))
        folds = make_temporal_folds(f)
        RecordingClassifier.fits = []
        scores = evaluate_temporal_cv(RecordingClassifier(), f, folds, 'discrete')
        self.assertEqual(scores.validacion.tolist(), [2021,2022,2023])
        self.assertEqual(len(RecordingClassifier.fits),3)
        for actual, fold in zip(RecordingClassifier.fits, folds):
            tr = f.iloc[list(fold.train_positions)]
            expected = new_discretizer().fit_transform(tr)
            np.testing.assert_array_equal(actual, expected)
        # Frozen preprocessing does not absorb extreme/missing validation values.
        tr = f.iloc[list(folds[0].train_positions)]
        disc = new_discretizer().fit(tr)
        edges = {k:v.copy() for k,v in disc.numeric_edges_.items()}
        valid = f.iloc[list(folds[0].validation_positions)].copy()
        valid[NUMERIC_FEATURES] = np.nan
        disc.transform(valid)
        for key in edges: np.testing.assert_array_equal(edges[key],disc.numeric_edges_[key])

    def test_all_model_interfaces_share_fold_runner_on_tiny_synthetic_data(self):
        folds = make_temporal_folds(self.train)
        configurations = [
            (ID3(), 'discrete'), (MEstimateCategoricalNB(), 'discrete'),
            (DecisionTreeClassifier(random_state=42), 'discrete'),
            (DecisionTreeClassifier(random_state=42), 'continuous'),
            (RandomForestClassifier(n_estimators=2,random_state=42), 'continuous'),
            (TenYearWinRateClassifier(), 'baseline'),
            (CategoricalNB(min_categories=4), 'discrete'),
        ]
        for estimator, representation in configurations:
            with self.subTest(model=type(estimator).__name__, representation=representation):
                self.assertTrue(is_classifier(clone(estimator)))
                scores = evaluate_temporal_cv(estimator,self.train,folds,representation)
                self.assertEqual(scores.validacion.tolist(),[2021,2022,2023])
                self.assertEqual(scores.n_validacion.tolist(),[4,4,4])
                self.assertFalse(hasattr(estimator,'classes_'))
                pipeline = make_model_pipeline(estimator,representation).fit(self.train,self.train.winner)
                original = pipeline.predict(self.test)
                altered = self.test.assign(winner='INVALID', gh=999, ga=-999)
                np.testing.assert_array_equal(original, pipeline.predict(altered))

    def test_notebook_syntax_and_invalidated_outputs(self):
        notebook = json.loads((ROOT/'notebook.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']))
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'],[])
        for path in (ROOT/'src').glob('*.py'): ast.parse(path.read_text())


if __name__ == '__main__': unittest.main()
