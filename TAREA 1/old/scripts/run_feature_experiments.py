"""Run the predeclared, validation-only feature comparisons."""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lab1-matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')

import pandas as pd
import numpy as np
import sklearn
from src.features import load_clean_matches
from src.evaluation import make_temporal_folds
from src.feature_experiments import (build_experiment_features, feature_variants,
    run_feature_experiments, fixed_candidates, plot_tradeoffs)


def main():
    if sys.version_info[:2] != (3,12) or not sklearn.__version__.startswith('1.9.'):
        raise RuntimeError('Se requiere Python 3.12 y scikit-learn 1.9.')
    source = ROOT/'data/raw/futbol_uruguayo.zip'
    prior = ROOT/'results/validation'
    manifest = json.loads((prior/'manifest.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['source_sha256']
    for relative, digest in manifest['implementation_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest, relative
    selected = pd.read_csv(prior/'selected.csv')
    matches = load_clean_matches(source, through_year=2023)
    frame = build_experiment_features(matches)
    folds = make_temporal_folds(frame)
    details, summary, chosen, predictions = run_feature_experiments(frame, selected)
    # Same features/settings must reproduce the original validation control.
    old = pd.read_csv(prior/'fold_metrics.csv')
    for model in chosen.model:
        config = selected.loc[selected.model.eq(model),'config_id'].iloc[0]
        reference = old.loc[old.config_id.eq(config)].sort_values('validacion')
        current = details.loc[details.model.eq(model) & details.variant.eq('current') &
                              details.decision.eq('three_class')].sort_values('validacion')
        np.testing.assert_allclose(current[['accuracy','macro_f1']].to_numpy(),
                                   reference[['accuracy','macro_f1']].to_numpy(), atol=1e-14, rtol=1e-14)
    output = ROOT/'results/feature_experiments'
    output.mkdir(parents=True, exist_ok=True)
    details.to_csv(output/'fold_metrics.csv',index=False)
    summary.to_csv(output/'summary.csv',index=False)
    chosen.to_csv(output/'selected.csv',index=False)
    predictions.to_csv(output/'validation_predictions.csv',index=False,date_format='%Y-%m-%d')
    plot_tradeoffs(summary,output/'accuracy_macro_f1.png')
    files = sorted((ROOT/'src').glob('*.py')) + [Path(__file__),ROOT/'requirements.txt',
        ROOT/'docs/feature_experiment_plan.md',prior/'selected.csv',prior/'manifest.json']
    record = {
        'python':platform.python_version(),'scikit_learn':sklearn.__version__,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'implementation_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'rows':len(frame),'latest_date':str(frame.date.max().date()),'seed':42,
        'variants':feature_variants(),
        'fixed_models':[{'model':c.model,'parameters':c.estimator.get_params(),
                         'representation':c.representation} for c in fixed_candidates(selected)],
        'folds':[{'year':f.validation_year,'n_train':len(f.train_positions),
                  'n_validation':len(f.validation_positions),
                  'train_end':str(frame.iloc[list(f.train_positions)].date.max().date()),
                  'validation_start':str(frame.iloc[list(f.validation_positions)].date.min().date()),
                  'validation_end':str(frame.iloc[list(f.validation_positions)].date.max().date())} for f in folds],
        'fits':len(feature_variants())*4*len(folds),
        'selection_metric':'mean annual macro-F1; labels E,L,V; equal weights; three_class only',
        'tie_policy':'fewer features, declared variant order, model name',
        'diagnostic':'restricted L/V prediction; original three-class truth and training unchanged',
        'control_reproduces_prior_selection':True,
        'test_evaluated':False,'final_refit_performed':False,
        'global_selection':chosen.iloc[0][['model','variant','parameters']].to_dict(),
    }
    (output/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print('\nSelecciones por modelo (solo tres clases):')
    print(chosen[['model','variant','accuracy_mean','macro_f1_mean','macro_f1_std']].to_string(index=False))
    print('Control actual reproduce la validación anterior. Test no evaluado.')


if __name__=='__main__':main()
