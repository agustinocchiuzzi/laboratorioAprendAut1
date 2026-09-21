"""Execute the common notebook from the original ZIP with no saved results.

Run with Python 3.12 / scikit-learn 1.9:
    python scripts/verify_final_nb.py

Only notebook.ipynb, requirements.txt, src/*.py and the original ZIP are copied
to an isolated temporary workspace. All code cells execute in order; the
preserved final tree/forest cells remain disabled. Outputs are written to
results/nb_delivery, never over the historical evaluation.
"""
from contextlib import ExitStack
from collections import Counter
import atexit
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import sklearn
    if sys.version_info[:2] != (3, 12) or not sklearn.__version__.startswith('1.9.'):
        raise RuntimeError('Se requiere Python 3.12 y scikit-learn 1.9.')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import numpy as np
    import pandas as pd
    from IPython import InteractiveShell
    from IPython.utils.capture import capture_output
    from sklearn.naive_bayes import CategoricalNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    original_cwd = Path.cwd()
    originals = {
        str(p.relative_to(ROOT)): sha256(p)
        for directory in ('results/naive_bayes', 'results/validation', 'results/feature_experiments')
        for p in (ROOT / directory).rglob('*') if p.is_file()
    }
    with tempfile.TemporaryDirectory(prefix='lab1-nb-delivery-') as scratch:
        work = Path(scratch)
        (work / 'src').mkdir()
        for source in (ROOT / 'src').glob('*.py'):
            shutil.copy2(source, work / 'src' / source.name)
        for name in ('notebook.ipynb', 'requirements.txt'):
            shutil.copy2(ROOT / name, work / name)
        (work / 'data/raw').mkdir(parents=True)
        shutil.copy2(ROOT / 'data/raw/futbol_uruguayo.zip', work / 'data/raw/futbol_uruguayo.zip')
        assert not (work / 'results').exists()
        os.environ['MPLCONFIGDIR'] = str(work / 'matplotlib')
        os.environ['IPYTHONDIR'] = str(work / 'ipython')
        notebook = json.loads((work / 'notebook.ipynb').read_text())
        shell = InteractiveShell.instance()
        counts = Counter()
        executed = []
        os.chdir(work)
        try:
            with ExitStack() as stack:
                for cls in (RandomForestClassifier, DecisionTreeClassifier):
                    stack.enter_context(patch.object(
                        cls, 'fit', side_effect=AssertionError('No entrenar árboles en la entrega NB')))

                def record_fit(cls):
                    original = cls.fit

                    def fit(instance, *args, **kwargs):
                        counts[cls.__name__] += 1
                        return original(instance, *args, **kwargs)
                    return fit

                for index, cell in enumerate(notebook['cells']):
                    if cell['cell_type'] != 'code':
                        continue
                    print(f'Celda {index}: ejecutando', flush=True)
                    with capture_output() as captured:
                        result = shell.run_cell(''.join(cell['source']), store_history=False)
                    result.raise_error()
                    executed.append(index)
                    cell['execution_count'] = len(executed)
                    cell['outputs'] = []
                    for name, content in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
                        if content:
                            cell['outputs'].append({'output_type': 'stream', 'name': name, 'text': content})
                    for rich in captured.outputs:
                        cell['outputs'].append({'output_type': 'display_data',
                                                'data': rich.data, 'metadata': rich.metadata})
                    if index == 2:
                        stack.enter_context(patch.object(
                            shell.user_ns['ID3'], 'fit', side_effect=AssertionError('No entrenar ID3')))
                        for cls in (shell.user_ns['MEstimateCategoricalNB'],
                                    CategoricalNB, shell.user_ns['TenYearWinRateClassifier']):
                            stack.enter_context(patch.object(cls, 'fit', record_fit(cls)))

            ns = shell.user_ns
            assert ns['RUN_FINAL_TEST'] is False
            assert counts == {'MEstimateCategoricalNB': 37, 'CategoricalNB': 37,
                              'TenYearWinRateClassifier': 1}, counts
            assert len(ns['nb_cv_details']) == 30
            assert len(ns['nb_feature_details']) == 84
            assert ns['nb_final']['diagnosis']['n_differences'] == 0
            comparison = {}
            # Optional regression comparison, only AFTER the independent run.
            saved = ROOT / 'results/naive_bayes/final/predictions.csv'
            if saved.is_file():
                expected = pd.read_csv(saved)
                pd.testing.assert_frame_equal(ns['nb_final']['predictions'], expected)
                comparison['saved_predictions_identical'] = len(expected)
                expected_summary = pd.read_csv(saved.parent / 'summary.csv')
                pd.testing.assert_frame_equal(ns['nb_final']['summary'], expected_summary)
            saved_selection = ROOT / 'results/naive_bayes/final_configurations.json'
            if saved_selection.is_file():
                expected = json.loads(saved_selection.read_text())
                for name, actual in ns['NB_FINAL_CONFIG'].items():
                    for key in ('estimator_parameters', 'variant', 'columns', 'decision', 'preprocessing'):
                        assert actual[key] == expected[name][key]
                    for key in ('validation_macro_f1_mean', 'validation_accuracy_mean'):
                        np.testing.assert_allclose(actual[key], expected[name][key], rtol=0, atol=1e-14)
                comparison['saved_selection_identical'] = True
            for relative, expected in originals.items():
                assert sha256(ROOT / relative) == expected, relative

            output = ROOT / 'results/nb_delivery'
            output.mkdir(parents=True, exist_ok=True)
            shutil.copytree(work / 'results/nb_delivery/figures', output / 'figures', dirs_exist_ok=True)
            # Una sola columna IEEE: ejes y leyenda legibles al tamaño de impresión.
            import matplotlib.pyplot as plt
            curve = ns['nb_cv_summary'].loc[ns['nb_cv_summary'].model.eq('NB propio')].copy()
            curve['m'] = curve.parameters.map(lambda value: json.loads(value)['m'])
            curve = curve.sort_values('m')
            with plt.rc_context({'font.size': 8}):
                figure, axis = plt.subplots(figsize=(3.5, 2.2), layout='constrained')
                for key, label in [('train_error_mean', 'Entrenamiento'),
                                   ('validation_error_mean', 'Validación')]:
                    axis.plot(curve.m, curve[key], marker='o', markersize=3, label=label)
                axis.set(xscale='log', xlabel='m (sklearn: α = m/4)',
                         ylabel='Error (1 − accuracy)', ylim=(0, 1))
                axis.grid(alpha=.25)
                axis.legend(fontsize=8)
                figure.savefig(output / 'figures/nb_error_report.png', dpi=220)
                plt.close(figure)
            for name, table in {
                'summary': ns['nb_final']['summary'],
                'class_report': ns['nb_final']['class_report'],
                'predictions': ns['nb_final']['predictions'],
                'validation_summary': ns['nb_cv_summary'],
                'validation_selected': ns['nb_cv_selected'],
                'feature_summary': ns['nb_feature_summary'],
                'feature_selected': ns['nb_feature_selected'],
                'examples': ns['nb_ejemplos'],
                'scenarios': ns['nb_resumen_escenarios'],
            }.items():
                table.to_csv(output / (name + '.csv'), index=False)
            for name, matrix in ns['nb_final']['confusion_matrices'].items():
                slug = {'NB propio': 'nb_propio', 'CategoricalNB': 'categorical_nb',
                        'Base 10 años': 'baseline_10y'}[name]
                matrix.to_csv(output / ('confusion_' + slug + '.csv'))
            (output / 'final_configurations.json').write_text(
                json.dumps(ns['NB_FINAL_CONFIG'], ensure_ascii=False, indent=2) + '\n')
            executed_path = output / 'notebook.executed.ipynb'
            executed_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
            record = {
                'python': sys.version.split()[0], 'scikit_learn': sklearn.__version__,
                'clean_workspace_inputs': ['notebook.ipynb', 'requirements.txt', 'src/*.py',
                                           'data/raw/futbol_uruguayo.zip'],
                'precomputed_results_available_during_execution': False,
                'all_code_cells_executed': executed, 'classifier_fit_counts': dict(counts),
                'tree_or_forest_fits': 0, 'test_used_for_selection': False,
                'historical_artifacts_unchanged': True, **comparison,
                'notebook_sha256': sha256(ROOT / 'notebook.ipynb'),
                'executed_notebook_sha256': sha256(executed_path),
                'verification_script_sha256': sha256(Path(__file__)),
                'source_sha256': {str(p.relative_to(ROOT)): sha256(p)
                                  for p in sorted((ROOT / 'src').glob('*.py'))},
                'raw_zip_sha256': sha256(ROOT / 'data/raw/futbol_uruguayo.zip'),
                'artifact_sha256': {str(p.relative_to(output)): sha256(p)
                                    for p in output.rglob('*')
                                    if p.is_file() and p.name != 'verification.json'},
            }
            (output / 'verification.json').write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + '\n')
            print(ns['nb_final']['summary'].to_string(index=False))
            print('Verificado desde cero, sin resultados intermedios:', output)
        finally:
            # Flush IPython history before TemporaryDirectory removes its database.
            atexit.unregister(shell.atexit_operations)
            shell.atexit_operations()
            os.chdir(original_cwd)


if __name__ == '__main__':
    main()
