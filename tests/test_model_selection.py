"""Regression tests for selection metric, tie rules and the holdout boundary."""
import json
import unittest
from unittest.mock import patch

import pandas as pd

from src.evaluation import evaluate_temporal_cv, make_temporal_folds
from src.model_selection import (model_candidates, run_model_selection,
                                 summarize_validation, REQUIRED_MODELS)
from test_data_evaluation import synthetic_features
from src.id3 import ID3


class SelectionTests(unittest.TestCase):
    def test_grid_and_fixed_comparisons(self):
        candidates = model_candidates()
        self.assertEqual(len(candidates),31)
        self.assertTrue(set(REQUIRED_MODELS).issubset({c.model for c in candidates}))
        rf = [c for c in candidates if c.model == 'Random Forest']
        self.assertEqual(len(rf),12)
        self.assertTrue(any(c.parameters == {'max_depth':None,'min_samples_leaf':1} for c in rf))
        self.assertTrue(all(c.estimator.n_estimators == 300 for c in rf))
        self.assertEqual(rf[0].parameters, {'max_depth':4,'min_samples_leaf':50})
        self.assertEqual(len([c for c in candidates if c.model.startswith('sklearn DT')]),2)

    def test_selection_uses_macro_f1_not_accuracy_or_fold_size(self):
        rows = []
        for config, values, accuracy, rank in [('a',[.9,.1,.1],.99,0),
                                               ('b',[.4,.4,.4],.50,2),
                                               ('c',[.4,.4,.4],.40,1)]:
            for year,score,size in zip([2021,2022,2023],values,[1000,10,10]):
                rows.append(dict(model='ID3',config_id=config,parameters='{}',
                                 representation='discrete',tie_rank=rank,validacion=year,
                                 n_validacion=size,train_error=.1,validation_error=1-accuracy,
                                 train_macro_f1=.8,macro_f1=score,accuracy=accuracy))
        summary,selected=summarize_validation(pd.DataFrame(rows))
        self.assertEqual(selected.iloc[0].config_id,'c')
        self.assertAlmostEqual(summary.loc[summary.config_id.eq('a'),'macro_f1_mean'].iloc[0],1.1/3)
        with self.assertRaises(ValueError): summarize_validation(pd.DataFrame(rows).iloc[1:])

    def test_test_dates_rejected_before_any_fit(self):
        with patch('src.model_selection.evaluate_temporal_cv') as evaluate:
            with self.assertRaises(ValueError): run_model_selection(synthetic_features(),progress=None)
            evaluate.assert_not_called()

    def test_train_and_validation_error_definitions(self):
        f=synthetic_features();f=f[f.year.le(2023)].reset_index(drop=True)
        result=evaluate_temporal_cv(ID3(),f,make_temporal_folds(f),'discrete')
        self.assertTrue((abs(result.train_error+result.train_accuracy-1)<1e-12).all())
        self.assertTrue((abs(result.validation_error+result.accuracy-1)<1e-12).all())
        self.assertTrue(result.train_macro_f1.between(0,1).all())
        self.assertEqual(result.validacion.tolist(),[2021,2022,2023])


if __name__ == '__main__': unittest.main()
