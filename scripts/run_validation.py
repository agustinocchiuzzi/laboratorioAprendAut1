"""Run the predeclared comparisons through 2023 only; never evaluate test.

python3.12 scripts/run_validation.py
"""
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

import numpy as np
import pandas as pd
import sklearn
from src.features import load_clean_matches, build_causal_match_features, NUMERIC_FEATURES
from src.evaluation import make_temporal_folds, new_discretizer
from src.model_selection import run_model_selection, plot_validation_curves, model_candidates


def main():
    if sys.version_info[:2] != (3, 12) or not sklearn.__version__.startswith('1.9.'):
        raise RuntimeError('Ejecutar con Python 3.12 y scikit-learn 1.9.')
    source = ROOT / 'data/raw/futbol_uruguayo.zip'
    matches = load_clean_matches(source)
    # Cut BEFORE feature construction: no 2024/2025 labels or inputs reach CV.
    matches = matches.loc[matches.date.dt.year.le(2023)].copy()
    frame = build_causal_match_features(matches)
    del matches
    folds = make_temporal_folds(frame)
    output = ROOT / 'results/validation'
    output.mkdir(parents=True, exist_ok=True)
    fold_info = []
    for fold in folds:
        tr = frame.iloc[list(fold.train_positions)]
        va = frame.iloc[list(fold.validation_positions)]
        disc = new_discretizer().fit(tr)
        encoded = disc.transform(tr)
        fold_info.append({
            'validation_year': fold.validation_year,
            'train_start': str(tr.date.min().date()), 'train_end': str(tr.date.max().date()),
            'validation_start': str(va.date.min().date()), 'validation_end': str(va.date.max().date()),
            'n_train': len(tr), 'n_validation': len(va),
            'training_category_cardinalities_including_zero': (encoded.max(axis=0) + 1).tolist(),
            'numeric_edges': {k:v.tolist() for k,v in disc.numeric_edges_.items()},
        })
    details, summary, selected = run_model_selection(frame)
    details.to_csv(output / 'fold_metrics.csv', index=False)
    summary.to_csv(output / 'grid_summary.csv', index=False)
    selected.to_csv(output / 'selected.csv', index=False)
    figures = plot_validation_curves(summary, output / 'figures')
    tracked_sources = sorted((ROOT/'src').glob('*.py')) + [Path(__file__), ROOT/'requirements.txt']
    manifest = {
        'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
        'numpy': np.__version__, 'pandas': pd.__version__, 'seed': 42,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'implementation_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in tracked_sources},
        'rows': len(frame), 'latest_date': str(frame.date.max().date()),
        'features': NUMERIC_FEATURES, 'folds': fold_info,
        'selection_metric': 'mean annual validation macro-F1; E,L,V; zero_division=0',
        'error_metric': '1 - accuracy', 'std': 'population SD across annual folds, ddof=0',
        'tie_policy': 'exact unrounded tie: smaller gain/m/alpha; RF shallower depth then larger leaf',
        'test_evaluated': False, 'final_refit_performed': False,
        'candidates': [{'model':c.model, 'parameters':c.parameters,
                        'estimator_parameters':c.estimator.get_params(),
                        'representation':c.representation, 'tie_rank':c.tie_rank}
                       for c in model_candidates()],
        'figures': [str(p.relative_to(ROOT)) for p in figures],
    }
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print('\nSelección solo en validación:')
    print(selected[['model','parameters','macro_f1_mean','macro_f1_std','validation_error_mean']].to_string(index=False))
    print('\nNo se realizó reajuste final ni evaluación 2024–2025.')


if __name__ == '__main__':
    main()
