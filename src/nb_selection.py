"""Configuraciones de NB verificadas; cargar artefactos no entrena modelos."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sklearn.naive_bayes import CategoricalNB

try:
    from .feature_experiments import make_feature_pipeline
    from .naive_bayes import MEstimateCategoricalNB
except ImportError:
    from feature_experiments import make_feature_pipeline
    from naive_bayes import MEstimateCategoricalNB

NB_MODELS = ('NB propio', 'CategoricalNB')


def training_fingerprint(matches):
    """Identidad del prefijo admitido; no requiere leer resultados reservados."""
    if matches.date.dt.year.gt(2023).any():
        raise ValueError('La selección de NB solo admite datos hasta 2023.')
    payload = matches.to_csv(index=False, date_format='%Y-%m-%d').encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def load_nb_manifest(root):
    """Comprueba código, resultados y referencias históricas antes de usarlos."""
    root = Path(root)
    record = json.loads((root / 'results/naive_bayes/manifest.json').read_text())
    if record['latest_date'] > '2023-12-31' or record['test_evaluated'] or record['final_refit_performed']:
        raise ValueError('El manifiesto no corresponde a selección sin test.')
    for group in ('implementation_sha256', 'artifacts_sha256', 'historical_artifacts_sha256'):
        for relative, expected in record[group].items():
            actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError('Artefacto NB desactualizado: ' + relative)
    return record


def final_nb_pipelines(record):
    """Crea los dos pipelines finales SIN ajustarlos ni cargar datos de test."""
    constructors = {'NB propio': MEstimateCategoricalNB, 'CategoricalNB': CategoricalNB}
    return {
        name: make_feature_pipeline(constructors[name](**config['estimator_parameters']),
                                    'discrete', config['columns'])
        for name, config in record['final_configurations'].items()
    }
