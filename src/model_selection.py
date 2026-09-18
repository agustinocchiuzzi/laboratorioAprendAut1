"""Predeclared grids and temporal selection, with no final-test evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier

try:
    from .baseline import TenYearWinRateClassifier
    from .evaluation import evaluate_temporal_cv, make_temporal_folds
    from .id3 import ID3
    from .naive_bayes import MEstimateCategoricalNB
except ImportError:
    from baseline import TenYearWinRateClassifier
    from evaluation import evaluate_temporal_cv, make_temporal_folds
    from id3 import ID3
    from naive_bayes import MEstimateCategoricalNB

SEED = 42
ID3_GRID = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)
M_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
# Four codes (0 reserved, plus bins 1/2/3): alpha=m/4 for comparable priors.
ALPHA_GRID = tuple(m / 4 for m in M_GRID)
RF_DEPTHS = (4, 8, None)
RF_LEAVES = (50, 20, 5, 1)
REQUIRED_MODELS = ('ID3', 'NB propio', 'CategoricalNB', 'Random Forest')


@dataclass
class Candidate:
    model: str
    estimator: object
    representation: str
    parameters: dict
    tie_rank: int


def model_candidates() -> list[Candidate]:
    """Order declares exact-tie preferences before any score is computed."""
    candidates = []
    for rank, gain in enumerate(ID3_GRID):
        candidates.append(Candidate('ID3', ID3(min_info_gain=gain), 'discrete',
                                    {'min_info_gain': gain}, rank))
    for rank, m in enumerate(M_GRID):
        candidates.append(Candidate('NB propio', MEstimateCategoricalNB(m=m),
                                    'discrete', {'m': m}, rank))
    for rank, alpha in enumerate(ALPHA_GRID):
        candidates.append(Candidate('CategoricalNB', CategoricalNB(
            alpha=alpha, min_categories=4, force_alpha=True),
            'discrete', {'alpha': alpha}, rank))
    for rank, (depth, leaf) in enumerate(product(RF_DEPTHS, RF_LEAVES)):
        candidates.append(Candidate('Random Forest', RandomForestClassifier(
            n_estimators=300, max_depth=depth, min_samples_leaf=leaf,
            class_weight='balanced_subsample', random_state=SEED, n_jobs=2),
            'continuous', {'max_depth': depth, 'min_samples_leaf': leaf}, rank))
    # Preserve useful previous fixed comparisons, including the unpruned forest
    # already present as depth=None/leaf=1 above. No tuning of these controls.
    for name, representation in [('sklearn DT (códigos)', 'discrete'),
                                 ('sklearn DT (tasas)', 'continuous')]:
        candidates.append(Candidate(name, DecisionTreeClassifier(
            criterion='entropy', random_state=SEED), representation, {}, 0))
    candidates.append(Candidate('Base 10 años', TenYearWinRateClassifier(),
                                'baseline', {}, 0))
    return candidates


def summarize_validation(details: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-weight annual means; select ONLY unrounded validation macro-F1.

    Standard deviations are descriptive population SDs across the three years,
    not confidence intervals. Training metrics are resubstitution diagnostics.
    """
    rows = []
    metrics = ('train_error', 'validation_error', 'train_macro_f1', 'macro_f1', 'accuracy')
    for config_id, group in details.groupby('config_id', sort=False):
        if sorted(group.validacion.tolist()) != [2021, 2022, 2023]:
            raise ValueError('Cada configuración debe cubrir una vez los tres años comunes.')
        first = group.iloc[0]
        row = {key: first[key] for key in
               ('model', 'config_id', 'parameters', 'representation', 'tie_rank')}
        for metric in metrics:
            if not np.isfinite(group[metric]).all():
                raise ValueError('Métricas no finitas: ' + metric)
            row[metric + '_mean'] = float(group[metric].mean())
            row[metric + '_std'] = float(group[metric].std(ddof=0))
        rows.append(row)
    summary = pd.DataFrame(rows)
    selected = summary.sort_values(
        ['model', 'macro_f1_mean', 'tie_rank'],
        ascending=[True, False, True], kind='stable',
    ).groupby('model', sort=False).head(1).reset_index(drop=True)
    return summary, selected


def run_model_selection(frame: pd.DataFrame, progress=print):
    """Fit CV candidates only. Reject any date after 2023 before the first fit."""
    folds = make_temporal_folds(frame)
    details = []
    for i, candidate in enumerate(model_candidates()):
        parameters = json.dumps(candidate.parameters, sort_keys=True)
        if progress:
            progress(f'{i+1:02d}/31 {candidate.model}: {parameters}', flush=True)
        scores = evaluate_temporal_cv(candidate.estimator, frame, folds,
                                      candidate.representation)
        details.append(scores.assign(
            model=candidate.model, config_id=f'config_{i:02d}', parameters=parameters,
            representation=candidate.representation, tie_rank=candidate.tie_rank,
        ))
    details = pd.concat(details, ignore_index=True)
    summary, selected = summarize_validation(details)
    return details, summary, selected


def plot_validation_curves(summary: pd.DataFrame, output_dir):
    """Export train/validation 1-accuracy and macro-F1 for all four grids."""
    from pathlib import Path
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    simple = [('ID3', 'min_info_gain', 'id3'), ('NB propio', 'm', 'nb_propio'),
              ('CategoricalNB', 'alpha', 'categorical_nb')]
    paths = []

    def draw(axes, part, parameter, title, logarithmic=False):
        part = part.copy()
        part['x'] = part.parameters.map(lambda p: json.loads(p)[parameter])
        part = part.sort_values('x')
        for axis, pairs, ylabel in [
            (axes[0], [('train_error', 'Entrenamiento'), ('validation_error', 'Validación')],
             'Error (1 − accuracy)'),
            (axes[1], [('train_macro_f1', 'Entrenamiento'), ('macro_f1', 'Validación')],
             'Macro-F1'),
        ]:
            for key, label in pairs:
                axis.plot(part.x, part[key + '_mean'], marker='o', label=label)
            axis.set(xlabel=parameter, ylabel=ylabel, ylim=(0, 1))
            axis.set_title(title)
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
            if logarithmic:
                axis.set_xscale('log')
        best = part.sort_values(['macro_f1_mean', 'tie_rank'], ascending=[False, True]).iloc[0]
        axes[1].scatter([best.x], [best.macro_f1_mean], marker='*', s=150,
                        facecolor='gold', edgecolor='black', zorder=5,
                        label='Mejor en este panel')
        axes[1].legend(fontsize=8)

    for model, parameter, slug in simple:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
        draw(axes, summary.loc[summary.model.eq(model)], parameter, model, parameter != 'min_info_gain')
        fig.suptitle('Media por año: validación 2021–2023; selección por macro-F1', fontsize=11)
        path = output_dir / f'{slug}.png'
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    forest = summary.loc[summary.model.eq('Random Forest')]
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), layout='constrained')
    for row, depth in enumerate(RF_DEPTHS):
        part = forest.loc[forest.parameters.map(lambda p: json.loads(p)['max_depth'] == depth)]
        draw(axes[row], part, 'min_samples_leaf',
             'Random Forest: profundidad ' + ('sin límite' if depth is None else str(depth)))
    fig.suptitle('300 árboles; medias 2021–2023; estrellas: mejor macro-F1 por panel', fontsize=11)
    path = output_dir / 'random_forest.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths
