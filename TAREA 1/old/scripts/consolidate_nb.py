"""Repite SOLO la búsqueda NB ya declarada: 30 + 42 ajustes, hasta 2023.

python3.12 scripts/consolidate_nb.py
Los resultados generales anteriores quedan intactos y con su propia procedencia.
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

from src.features import load_clean_matches, NUMERIC_FEATURES
from src.evaluation import make_temporal_folds
from src.feature_experiments import (build_experiment_features, feature_variants,
    make_feature_pipeline, fixed_candidates, run_feature_experiments)
from src.model_selection import (model_candidates, run_model_selection, plot_validation_curves)
from src.nb_selection import NB_MODELS, training_fingerprint


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_comparison(actual, historical, keys):
    """Compara métricas numéricas sin atribuir a redondeo cambios de selección."""
    old = historical.loc[historical.model.isin(NB_MODELS)]
    a = actual.sort_values(keys).reset_index(drop=True)
    b = old.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[keys], b[keys], check_dtype=False)
    numeric = list(a.select_dtypes(include='number').columns)
    delta = float(np.max(np.abs(a[numeric].to_numpy() - b[numeric].to_numpy())))
    same = np.allclose(a[numeric], b[numeric], atol=1e-14, rtol=1e-14)
    return {'same_within_1e_minus_14': bool(same), 'max_absolute_difference': delta}


def preprocessing_audit(frame, folds):
    """Registra por variante los cortes/medianas que cada fold puede aprender.

    Los runners ajustan su propia instancia dentro de cada pipeline. Esta
    reconstrucción audita el protocolo y no ajusta clasificadores adicionales.
    """
    audit = []
    estimator = next(c.estimator for c in model_candidates() if c.model == 'NB propio')
    for variant, columns in feature_variants().items():
        for fold in folds:
            tr = frame.iloc[list(fold.train_positions)]
            pre = make_feature_pipeline(estimator, 'discrete', columns).named_steps['preprocessing']
            X = pre.fit_transform(tr)
            cardinalities = np.maximum(X.max(axis=0) + 1, estimator.min_categories)
            if not (cardinalities == 4).all():
                raise ValueError('La equivalencia alpha=m/4 no se cumple en ' + variant)
            audit.append({
                'variant': variant, 'validation_year': fold.validation_year,
                'columns': list(columns), 'n_categories': cardinalities.tolist(),
                'observed_max_plus_one': (X.max(axis=0) + 1).tolist(),
                'numeric_edges': {c: e.tolist() for c, e in pre.numeric_edges_.items()},
                'numeric_medians': pre.numeric_medians_,
                'fitted_through': str(tr.date.max().date()),
            })
    return audit


def main():
    if sys.version_info[:2] != (3, 12) or not sklearn.__version__.startswith('1.9.'):
        raise RuntimeError('Se requiere Python 3.12 y scikit-learn 1.9.')
    previous = {}
    historical_files = []
    for stage in ('validation', 'feature_experiments'):
        directory = ROOT / 'results' / stage
        previous[stage] = json.loads((directory / 'manifest.json').read_text())
        historical_files.extend(p for p in directory.rglob('*') if p.is_file())
    historical_hashes = {str(p.relative_to(ROOT)): digest(p) for p in historical_files}
    variants = {k: list(v) for k, v in feature_variants().items()}
    if variants != previous['feature_experiments']['variants']:
        raise ValueError('No se permite ampliar o modificar las variantes declaradas.')
    old_grid = [(c['model'], c['parameters']) for c in previous['validation']['candidates']
                if c['model'] in NB_MODELS]
    candidates = [c for c in model_candidates() if c.model in NB_MODELS]
    if [(c.model, c.parameters) for c in candidates] != old_grid:
        raise ValueError('No se permite modificar la grilla NB declarada.')

    matches = load_clean_matches(ROOT / 'data/raw/futbol_uruguayo.zip', through_year=2023)
    frame = build_experiment_features(matches)
    folds = make_temporal_folds(frame)
    fold_records = []
    for fold, old, feature_old in zip(folds, previous['validation']['folds'],
                                    previous['feature_experiments']['folds']):
        tr, va = frame.iloc[list(fold.train_positions)], frame.iloc[list(fold.validation_positions)]
        current = {
            'validation_year': fold.validation_year, 'n_train': len(tr), 'n_validation': len(va),
            'train_start': str(tr.date.min().date()), 'train_end': str(tr.date.max().date()),
            'validation_start': str(va.date.min().date()), 'validation_end': str(va.date.max().date()),
        }
        if any(old[k] != v for k, v in current.items()):
            raise ValueError('Los folds no coinciden con la validación histórica.')
        if any(feature_old[k] != current[k] for k in feature_old if k != 'year') or feature_old['year'] != fold.validation_year:
            raise ValueError('Los experimentos de atributos no usaron los mismos folds.')
        current['train_positions_sha256'] = hashlib.sha256(np.asarray(fold.train_positions, dtype='<i8').tobytes()).hexdigest()
        current['validation_positions_sha256'] = hashlib.sha256(np.asarray(fold.validation_positions, dtype='<i8').tobytes()).hexdigest()
        fold_records.append(current)
    audit = preprocessing_audit(frame, folds)

    details, summary, selected = run_model_selection(frame, models=NB_MODELS)
    f_details, f_summary, f_selected, predictions = run_feature_experiments(
        frame, selected, models=NB_MODELS)
    # El control de seis atributos debe reproducir la etapa de hiperparámetros.
    for row in selected.itertuples():
        a = details.loc[details.config_id.eq(row.config_id)].sort_values('validacion')
        b = f_details.loc[f_details.model.eq(row.model) & f_details.variant.eq('current') &
                          f_details.decision.eq('three_class')].sort_values('validacion')
        np.testing.assert_allclose(a[['accuracy', 'macro_f1']], b[['accuracy', 'macro_f1']], atol=1e-14)
    comparison = {
        'validation': frame_comparison(details, pd.read_csv(ROOT/'results/validation/fold_metrics.csv'), ['model', 'config_id', 'validacion']),
        'features': frame_comparison(f_details, pd.read_csv(ROOT/'results/feature_experiments/fold_metrics.csv'), ['model', 'variant', 'decision', 'validacion']),
    }
    old_predictions = pd.read_csv(ROOT/'results/feature_experiments/validation_predictions.csv')
    old_predictions = old_predictions.loc[old_predictions.model.isin(NB_MODELS)]
    keys = ['model', 'variant', 'validacion', 'position']
    old_predictions = old_predictions.sort_values(keys).reset_index(drop=True)
    current_predictions = predictions.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(current_predictions[keys], old_predictions[keys], check_dtype=False)
    comparison['changed_validation_predictions'] = {
        c: int((current_predictions[c] != old_predictions[c]).sum()) for c in ('three_class', 'never_draw')}
    # Verificar ambos modelos por separado, incluso aunque sean equivalentes.
    paired = f_details.pivot(index=['variant', 'decision', 'validacion'], columns='model', values=['accuracy', 'macro_f1'])
    for metric in ('accuracy', 'macro_f1'):
        np.testing.assert_allclose(paired[(metric, NB_MODELS[0])], paired[(metric, NB_MODELS[1])], atol=1e-14)

    output = ROOT / 'results/naive_bayes'
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        'validation_folds.csv': details, 'validation_summary.csv': summary, 'validation_selected.csv': selected,
        'feature_folds.csv': f_details, 'feature_summary.csv': f_summary, 'feature_selected.csv': f_selected,
        'validation_predictions.csv': predictions,
    }
    for name, table in frames.items():
        table.to_csv(output/name, index=False, date_format='%Y-%m-%d')
    figures = plot_validation_curves(summary, output/'figures')
    fixed = {c.model: c for c in fixed_candidates(selected, models=NB_MODELS)}
    final = {row.model: {
        'estimator': type(fixed[row.model].estimator).__name__,
        'estimator_parameters': fixed[row.model].estimator.get_params(),
        'variant': row.variant, 'columns': variants[row.variant],
        'representation': 'discrete', 'decision': 'three_class',
        'preprocessing': {'n_bins': 3, 'fixed_cuts': {c: [0.3, 0.6] for c in NUMERIC_FEATURES[:2]},
                          'quantiles_and_medians': 'fit on training only'},
        'validation_macro_f1_mean': row.macro_f1_mean,
        'validation_accuracy_mean': row.accuracy_mean,
    } for row in f_selected.itertuples()}
    (output/'final_configurations.json').write_text(json.dumps(final, ensure_ascii=False, indent=2)+'\n')
    sources = sorted((ROOT/'src').glob('*.py')) + [Path(__file__), ROOT/'requirements.txt']
    artifacts = [output/name for name in frames] + figures + [output/'final_configurations.json']
    record = {
        'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
        'numpy': np.__version__, 'pandas': pd.__version__, 'seed': 42,
        'scope': list(NB_MODELS), 'rows': len(frame), 'latest_date': str(frame.date.max().date()),
        'training_sha256': training_fingerprint(matches),
        'source_policy': 'date-only partition pass; parse/clean outcomes only through 2023',
        'implementation_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'artifacts_sha256': {str(p.relative_to(ROOT)): digest(p) for p in artifacts},
        'historical_artifacts_sha256': historical_hashes,
        'historical_code_mismatches': {stage: [p for p, h in m['implementation_sha256'].items()
                                               if digest(ROOT/p) != h] for stage, m in previous.items()},
        'folds': fold_records, 'preprocessing_by_variant_and_fold': audit,
        'grid': [{'model': c.model, 'parameters': c.parameters, 'estimator_parameters': c.estimator.get_params()} for c in candidates],
        'variants': variants, 'fits': {'hyperparameters': len(details), 'features': len(f_details)//2},
        'selection_metric': 'unrounded mean annual validation macro-F1; E/L/V; equal year weights',
        'tie_policy': {'hyperparameters': 'smaller m/alpha', 'features': 'fewer columns, declared variant order'},
        'error_metric': '1 - accuracy', 'selection_order': 'hyperparameters on six rates, then fixed-parameter feature comparison',
        'comparison_with_historical': comparison, 'final_configurations': final,
        'test_evaluated': False, 'final_refit_performed': False,
    }
    (output/'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(comparison, indent=2))
    print(f_selected[['model', 'variant', 'parameters', 'macro_f1_mean', 'accuracy_mean']].to_string(index=False))
    print('Solo NB, 30 + 42 ajustes; sin reajuste final ni consulta de resultados 2024–2025.')


if __name__ == '__main__':
    main()
