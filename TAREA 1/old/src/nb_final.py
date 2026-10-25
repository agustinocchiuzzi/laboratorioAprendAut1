"""Final holdout for the two selected NB models and the causal ten-year baseline.

No search, cross-validation, alternate decision rule or tree fitting occurs here.
The selection manifest is immutable; final evidence lives in its own directory.
"""
from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support

try:
    from .baseline import BASELINE_FEATURES, TenYearWinRateClassifier
    from .evaluation import CLASSES, temporal_holdout
    from .features import build_causal_match_features, load_clean_matches
    from .nb_selection import NB_MODELS, final_nb_pipelines, load_nb_manifest, training_fingerprint
except ImportError:
    from baseline import BASELINE_FEATURES, TenYearWinRateClassifier
    from evaluation import CLASSES, temporal_holdout
    from features import build_causal_match_features, load_clean_matches
    from nb_selection import NB_MODELS, final_nb_pipelines, load_nb_manifest, training_fingerprint

MODEL_SLUGS = {'NB propio': 'nb_propio', 'CategoricalNB': 'categorical_nb',
               'Base 10 años': 'baseline_10y'}
BASELINE = 'Base 10 años'
IDENTITY = ['date', 'home', 'away', 'winner']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_sha256(frame):
    return hashlib.sha256(frame.to_csv(index=False, date_format='%Y-%m-%d').encode()).hexdigest()


def write_json(path, record):
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def evaluate_final_nb(frame, selection):
    """Fit each fresh NB pipeline once on <=2023; predict the common 2024–25 rows.

    frame must come from build_causal_match_features on admitted matches. Only
    the explicit input columns are passed to fit/predict; labels are separate.
    """
    train, test = temporal_holdout(frame)
    if frame.duplicated(['date', 'home', 'away']).any():
        raise ValueError('Se requiere una única fila por partido admitido.')
    configs = selection['final_configurations']
    if set(configs) != set(NB_MODELS):
        raise ValueError('La evaluación final admite exclusivamente ambos NB seleccionados.')
    columns = list(configs['NB propio']['columns'])
    if len(columns) != 10 or columns != list(configs['CategoricalNB']['columns']):
        raise ValueError('Ambos NB deben usar los mismos diez atributos seleccionados.')
    pipelines = final_nb_pipelines(selection)
    predictions = test[IDENTITY].copy()
    summaries, reports, probabilities = [], [], []
    matrices, fitted, transformed, audit, states, joint = {}, {}, {}, {}, {}, {}
    for name in NB_MODELS:
        config = configs[name]
        pipeline = pipelines[name]
        pre = pipeline.named_steps['preprocessing']
        if (config['decision'] != 'three_class' or config['representation'] != 'discrete'
                or pre.n_bins != config['preprocessing']['n_bins']
                or {c: list(v) for c, v in pre.fixed_cuts.items()}
                != config['preprocessing']['fixed_cuts']):
            raise ValueError('El pipeline no reproduce el preprocesamiento/decisión cerrados.')
        pipeline.fit(train[columns], train.winner)
        fitted[name] = pipeline
        model = pipeline.named_steps['model']
        np.testing.assert_array_equal(model.classes_, CLASSES)
        states[name] = joblib.hash(pipeline)
        predictions['pred_' + MODEL_SLUGS[name]] = pipeline.predict(test[columns])
        transformed[name] = pre.transform(test[columns])
        joint[name] = model._joint_log_likelihood(transformed[name])
        log_proba = model.predict_log_proba(transformed[name])
        probability_rows = test[IDENTITY].assign(model=name)
        for index, label in enumerate(CLASSES):
            probability_rows['joint_log_' + label] = joint[name][:, index]
            probability_rows['log_probability_' + label] = log_proba[:, index]
        probabilities.append(probability_rows)
        if states[name] != joblib.hash(pipeline):
            raise AssertionError('El ajuste cambió durante test: ' + name)
        audit[name] = {
            'columns': columns, 'n_fit_rows': len(train),
            'fitted_through': str(train.date.max().date()),
            'numeric_edges': {c: e.tolist() for c, e in pre.numeric_edges_.items()},
            'numeric_medians': pre.numeric_medians_,
            'n_categories': model.n_categories_.tolist(),
            'classes': model.classes_.tolist(), 'class_count': model.class_count_.tolist(),
            'class_log_prior': model.class_log_prior_.tolist(),
            'state_before_test': states[name], 'state_after_test': joblib.hash(pipeline),
        }

    baseline = TenYearWinRateClassifier(tie_break='home').fit(train[BASELINE_FEATURES], train.winner)
    fitted[BASELINE] = baseline
    baseline_state = joblib.hash(baseline)
    predictions['pred_baseline_10y'] = baseline.predict(test[BASELINE_FEATURES])
    if joblib.hash(baseline) != baseline_state:
        raise AssertionError('El baseline cambió durante predict.')
    for name, slug in MODEL_SLUGS.items():
        predicted = predictions['pred_' + slug]
        predictions['correct_' + slug] = predicted.eq(test.winner)
        summaries.append({'model': name, 'n_train': len(train), 'n_test': len(test),
                          'accuracy': accuracy_score(test.winner, predicted),
                          'macro_f1': f1_score(test.winner, predicted, labels=list(CLASSES),
                                               average='macro', zero_division=0)})
        precision, recall, f1, support = precision_recall_fscore_support(
            test.winner, predicted, labels=list(CLASSES), zero_division=0)
        reports.extend({'model': name, 'class': label, 'precision': float(precision[i]),
                        'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])}
                       for i, label in enumerate(CLASSES))
        matrices[name] = pd.DataFrame(confusion_matrix(test.winner, predicted, labels=list(CLASSES)),
                                     index=pd.Index(CLASSES, name='real'), columns=CLASSES)

    own, reference = (fitted[name].named_steps['model'] for name in NB_MODELS)
    predictions['nb_identical'] = predictions.pred_nb_propio.eq(predictions.pred_categorical_nb)
    equal_inputs = np.array_equal(transformed['NB propio'], transformed['CategoricalNB'])
    equal_categories = np.array_equal(own.n_categories_, reference.n_categories_)
    equal_counts = all(np.array_equal(a, b) for a, b in zip(own.category_count_, reference.category_count_))
    equivalence = (equal_inputs and equal_categories and equal_counts
                   and np.allclose(own.m / own.n_categories_, reference.alpha, rtol=0, atol=0)
                   and np.allclose(own.class_log_prior_, reference.class_log_prior_, rtol=0, atol=1e-14))
    delta = float(np.max(np.abs(joint['NB propio'] - joint['CategoricalNB'])))
    differences = predictions.loc[~predictions.nb_identical].copy()
    # Preserve both scores and top-two margins for every disagreement, without
    # changing smoothing, columns or argmax based on holdout outcomes.
    for name in NB_MODELS:
        scores = joint[name]
        slug = MODEL_SLUGS[name]
        differences['margin_' + slug] = (np.sort(scores, axis=1)[:, -1]
                                         - np.sort(scores, axis=1)[:, -2])[~predictions.nb_identical]
        for i, label in enumerate(CLASSES):
            differences[slug + '_joint_log_' + label] = scores[~predictions.nb_identical, i]
    diagnosis = {
        'n_predictions': len(test), 'n_differences': len(differences),
        'identical_predictions': bool(predictions.nb_identical.all()),
        'identical_discrete_inputs': equal_inputs, 'identical_category_counts': equal_counts,
        'identical_cardinalities': equal_categories,
        'm_over_K_equals_alpha': bool(np.all(own.m / own.n_categories_ == reference.alpha)),
        'equivalence_conditions_met': bool(equivalence),
        'max_abs_class_log_prior_difference': float(np.max(np.abs(own.class_log_prior_ - reference.class_log_prior_))),
        'max_abs_feature_log_probability_difference': float(max(
            np.max(np.abs(a-b)) for a, b in zip(own.feature_log_prob_, reference.feature_log_prob_))),
        'max_abs_joint_log_likelihood_difference': delta,
        'decision': 'argmax over classes E, L, V; exact ties choose first class',
        'explanation': ('Los conteos, entradas, dominios y priors coinciden; K=4 y alpha=m/4. '
                        'Ambas fórmulas son equivalentes salvo redondeo de punto flotante.'
                        if equivalence else
                        'Las condiciones de equivalencia no se cumplen; consultar entradas, '
                        'cardinalidades, conteos, priors y puntajes guardados.'),
    }
    if len(differences) and equivalence:
        diagnosis['disagreement_cause'] = (
            'Diferencias numéricas cerca del empate entre puntajes; se conservan ambos argmax.'
            if all((differences['margin_' + MODEL_SLUGS[n]] <= 2 * delta).all() for n in NB_MODELS)
            else 'Discrepancia de decisión no explicada por los márgenes; requiere investigar los puntajes.')
    return {'summary': pd.DataFrame(summaries), 'class_report': pd.DataFrame(reports),
            'predictions': predictions, 'probabilities': pd.concat(probabilities, ignore_index=True),
            'disagreements': differences, 'confusion_matrices': matrices,
            'diagnosis': diagnosis, 'preprocessing': audit, 'models': fitted,
            'train': train, 'test': test}


def run_final_evaluation(root):
    """Reproduce the closed evaluation and save metrics, fitted state and provenance."""
    root = Path(root).resolve()
    if sys.version_info[:2] != (3, 12) or not version('scikit-learn').startswith('1.9.'):
        raise RuntimeError('Se requiere Python 3.12 y scikit-learn 1.9.')
    # Validate the selection BEFORE loading test outcomes.
    selection = load_nb_manifest(root)
    config_file = root / 'results/naive_bayes/final_configurations.json'
    configs = json.loads(config_file.read_text())
    if configs != selection['final_configurations']:
        raise ValueError('La configuración final difiere del manifiesto de selección.')
    raw = root / 'data/raw/futbol_uruguayo.zip'
    prefix = load_clean_matches(raw, through_year=2023)
    if training_fingerprint(prefix) != selection['training_sha256']:
        raise ValueError('El entrenamiento no coincide con el prefijo de selección.')
    admitted = load_clean_matches(raw)
    pd.testing.assert_frame_equal(prefix, admitted.loc[admitted.year.le(2023)].reset_index(drop=True))
    frame = build_causal_match_features(admitted)
    train, _ = temporal_holdout(frame)
    pd.testing.assert_frame_equal(train, build_causal_match_features(prefix))
    result = evaluate_final_nb(frame, selection)
    output = root / 'results/naive_bayes/final'
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []

    def artifact(name):
        path = output / name
        artifacts.append(path)
        return path

    for key in ('summary', 'class_report', 'predictions', 'probabilities', 'disagreements'):
        result[key].to_csv(artifact(key + '.csv'), index=False, date_format='%Y-%m-%d')
    for name, matrix in result['confusion_matrices'].items():
        matrix.to_csv(artifact('confusion_' + MODEL_SLUGS[name] + '.csv'))
    columns = configs['NB propio']['columns']
    result['test'][IDENTITY + columns + BASELINE_FEATURES].to_csv(
        artifact('test_features.csv'), index=False, date_format='%Y-%m-%d')
    for name, model in result['models'].items():
        joblib.dump(model, artifact(MODEL_SLUGS[name] + '.joblib'), compress=3)
    write_json(artifact('configurations.json'), {
        'naive_bayes': configs,
        'baseline': {'estimator': 'TenYearWinRateClassifier', 'tie_break': 'home',
                     'columns': BASELINE_FEATURES, 'no_history': 0.5,
                     'window': '[date - 10 calendar years, date)',
                     'provenance': 'Reevaluated with the shared causal feature builder; historical '
                                   'notebook results predate the current policy and are not reused.'},
    })
    write_json(artifact('preprocessing.json'), result['preprocessing'])
    write_json(artifact('equivalence.json'), result['diagnosis'])
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'matplotlib',
                'threadpoolctl', 'narwhals', 'contourpy', 'cycler', 'fonttools',
                'kiwisolver', 'packaging', 'pillow', 'pyparsing', 'python-dateutil',
                'pytz', 'tzdata', 'six']
    versions = {name: version(name) for name in packages}
    artifact('requirements.lock.txt').write_text(''.join(f'{p}=={v}\n' for p, v in versions.items()))
    # Standalone confusion figure, always in E/L/V order.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    vmax = max(int(m.to_numpy().max()) for m in result['confusion_matrices'].values())
    for ax, (name, matrix) in zip(axes, result['confusion_matrices'].items()):
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=vmax)
        ax.set(xticks=range(3), xticklabels=CLASSES, yticks=range(3), yticklabels=CLASSES,
               xlabel='Predicción', ylabel='Resultado real', title=name)
        for row in range(3):
            for col in range(3):
                value = int(matrix.iloc[row, col])
                ax.text(col, row, str(value), ha='center', va='center',
                        color='white' if value > vmax / 2 else 'black')
    fig.savefig(artifact('confusion_matrices.png'), dpi=160)
    plt.close(fig)
    sources = [root / p for p in selection['implementation_sha256']]
    sources += [root / 'src/nb_final.py', root / 'scripts/evaluate_final_nb.py', root / 'docs/data_policy.md']
    selection_path = root / 'results/naive_bayes/manifest.json'
    manifest = {
        'python': platform.python_version(), 'packages': versions,
        'scope': list(MODEL_SLUGS), 'classes': list(CLASSES),
        'fit_calls': {name: 1 for name in MODEL_SLUGS},
        'test_evaluated': True, 'final_refit_performed': True, 'test_used_for_selection': False,
        'information_policy': 'Admitted F matches only; all matches on a date use strictly earlier '
                              'dates; histories update after the whole date, including earlier '
                              'test dates; preprocessing and classifiers never refit within test.',
        'metrics': {'aggregation': 'all admitted 2024–2025 matches pooled',
                    'macro_f1_labels': list(CLASSES), 'zero_division': 0,
                    'confusion_rows': 'real', 'confusion_columns': 'prediction'},
        'split': {name: {'rows': len(part), 'start': str(part.date.min().date()),
                         'end': str(part.date.max().date()),
                         'class_support': {c: int(part.winner.eq(c).sum()) for c in CLASSES},
                         'rows_by_year': {str(y): int(n) for y, n in part.groupby(part.date.dt.year).size().items()},
                         'identity_sha256': frame_sha256(part[IDENTITY]),
                         'input_sha256': frame_sha256(part[columns + BASELINE_FEATURES])}
                  for name, part in [('train', result['train']), ('test', result['test'])]},
        'training_sha256': training_fingerprint(prefix), 'admitted_sha256': frame_sha256(admitted),
        'raw_sha256': sha256(raw), 'prefix_features_unchanged_by_test': True,
        'selection_manifest_sha256': sha256(selection_path),
        'implementation_sha256': {str(p.relative_to(root)): sha256(p) for p in sources},
        'artifacts_sha256': {str(p.relative_to(root)): sha256(p) for p in artifacts},
        'reproduce': 'python3.12 scripts/evaluate_final_nb.py',
    }
    write_json(output / 'manifest.json', manifest)
    return result


def load_final_evaluation(root):
    """Read verified tables for the notebook without loading or fitting models."""
    root = Path(root).resolve()
    load_nb_manifest(root)
    output = root / 'results/naive_bayes/final'
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest['selection_manifest_sha256'] != sha256(root / 'results/naive_bayes/manifest.json'):
        raise ValueError('Cambió la selección previa a test.')
    if manifest['raw_sha256'] != sha256(root / 'data/raw/futbol_uruguayo.zip'):
        raise ValueError('Cambió el archivo original de partidos.')
    for group in ('implementation_sha256', 'artifacts_sha256'):
        for relative, expected in manifest[group].items():
            if sha256(root / relative) != expected:
                raise ValueError('Evaluación final desactualizada: ' + relative)
    result = {key: pd.read_csv(output / (key + '.csv')) for key in
              ('summary', 'class_report', 'predictions', 'probabilities', 'disagreements')}
    result['confusion_matrices'] = {name: pd.read_csv(output / ('confusion_' + slug + '.csv'), index_col=0)
                                    for name, slug in MODEL_SLUGS.items()}
    for key, name in [('diagnosis', 'equivalence'), ('preprocessing', 'preprocessing'),
                      ('configurations', 'configurations')]:
        result[key] = json.loads((output / (name + '.json')).read_text())
    result['manifest'] = manifest
    return result
