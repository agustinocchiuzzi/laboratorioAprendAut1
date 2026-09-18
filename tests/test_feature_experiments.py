import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.features import _clean_matches, NUMERIC_FEATURES
from src.feature_experiments import (build_experiment_features, feature_variants,
    make_feature_pipeline, predict_without_draws, classification_metrics,
    run_feature_experiments, summarize_experiments)
from src.id3 import ID3
from src.naive_bayes import MEstimateCategoricalNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from test_data_evaluation import matches, synthetic_features


class FeatureExperimentTests(unittest.TestCase):
    def test_draw_rates_use_only_prior_dates_and_preserve_current_features(self):
        raw=matches([
            ('2020-01-01','A','B',1,1,'F'),
            ('2020-01-02','A','C',1,0,'F'),
            ('2020-01-02','B','D',0,0,'F'),
            ('2020-01-03','A','B',0,1,'F'),
        ])
        original=build_experiment_features(_clean_matches(raw))
        self.assertAlmostEqual(original.iloc[0].home_draw_rate_all,1/3)
        self.assertEqual(original.iloc[1].home_draw_rate_all,1.)
        self.assertEqual(original.iloc[3].home_draw_rate_all,.5)
        self.assertEqual(original.iloc[3].away_draw_rate_all,1.)
        changed=raw.copy();changed.loc[1:,['gh','ga']]=[2,0]
        altered=build_experiment_features(_clean_matches(changed))
        columns=NUMERIC_FEATURES+['home_draw_rate_all','away_draw_rate_all']
        pd.testing.assert_frame_equal(original.iloc[:3][columns],altered.iloc[:3][columns])
        self.assertNotEqual(original.iloc[3].away_draw_rate_all,altered.iloc[3].away_draw_rate_all)
        prefix=build_experiment_features(_clean_matches(raw.iloc[:3]))
        pd.testing.assert_frame_equal(original.iloc[:3].reset_index(drop=True),prefix)

    def test_same_day_shared_team_does_not_update_draw_rates(self):
        f=build_experiment_features(_clean_matches(matches([
            ('2020-01-01','A','B',1,1,'F'),('2020-01-01','A','C',0,0,'F')
        ])))
        np.testing.assert_allclose(f.home_draw_rate_all,1/3)

    def test_ablation_is_separate_and_target_columns_rejected(self):
        v=feature_variants()
        self.assertEqual(len(v),7)
        self.assertEqual(set(v['current'])-set(v['without_home_history']),{'home_win_rate_as_home_all'})
        self.assertEqual(set(v['current'])-set(v['without_home_h2h']),{'home_win_rate_h2h_as_home'})
        with self.assertRaises(ValueError):make_feature_pipeline(ID3(),'discrete',['winner'])

    def test_never_draw_preserves_three_class_truth_and_penalizes_missed_draws(self):
        truth=np.array(['E','L','V']);pred=np.array(['L','L','V'])
        result=classification_metrics(truth,pred)
        self.assertAlmostEqual(result['accuracy'],2/3)
        self.assertAlmostEqual(result['macro_f1'],(0+2/3+1)/3)
        self.assertEqual(result['true_draws'],1)
        self.assertEqual(result['predicted_draws'],0)
        self.assertEqual(result['draw_f1'],0)

    def test_diagnostic_for_all_models_without_refit(self):
        f=synthetic_features().iloc[:12].copy()
        for model,rep in [(ID3(),'discrete'),(MEstimateCategoricalNB(),'discrete'),
                          (CategoricalNB(min_categories=4),'discrete'),
                          (RandomForestClassifier(n_estimators=2,random_state=42),'continuous')]:
            with self.subTest(model=type(model).__name__):
                pipe=make_feature_pipeline(model,rep,feature_variants()['current']).fit(f,f.winner)
                with patch.object(pipe.named_steps['model'],'fit',side_effect=AssertionError('refit')):
                    p=predict_without_draws(pipe,f)
                    self.assertTrue(set(p)<=set(['L','V']))
                    np.testing.assert_array_equal(p,predict_without_draws(pipe,f.assign(winner='INVALID')))
                    self.assertEqual(len(p),len(f))

    def test_holdout_rejected_before_fit(self):
        with patch('src.feature_experiments.fixed_candidates') as fixed:
            with self.assertRaises(ValueError):run_feature_experiments(synthetic_features(),pd.DataFrame(),None)
            fixed.assert_not_called()

    def test_diagnostic_cannot_win_primary_selection(self):
        rows=[]
        for variant,rank,n_features in [('current',0,6),('without_home_history',5,5)]:
            for decision in ['three_class','never_draw']:
                for year in [2021,2022,2023]:
                    rows.append(dict(model='ID3',variant=variant,variant_rank=rank,n_features=n_features,
                        parameters='{}',decision=decision,validacion=year,accuracy=.5,
                        macro_f1=.9 if decision=='never_draw' else .4,train_accuracy=.6,train_macro_f1=.5,
                        draw_recall=0,draw_f1=0,true_draws=2,predicted_draws=0))
        _,selected=summarize_experiments(pd.DataFrame(rows))
        self.assertEqual(selected.iloc[0].decision,'three_class')
        self.assertEqual(selected.iloc[0].variant,'without_home_history')


if __name__=='__main__':unittest.main()
