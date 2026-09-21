"""Focused feature comparisons with fixed, previously selected hyperparameters.

No change to the original six-feature pipeline or its saved validation results.
"""
from __future__ import annotations

from collections import defaultdict
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

try:
    from .features import NUMERIC_FEATURES, build_causal_match_features
    from .evaluation import CLASSES, LAST_5_CUTS, make_temporal_folds
    from .model_selection import model_candidates, REQUIRED_MODELS
    from .preprocessing import MixedTypeDiscretizer
    from .id3 import ID3
except ImportError:
    from features import NUMERIC_FEATURES, build_causal_match_features
    from evaluation import CLASSES, LAST_5_CUTS, make_temporal_folds
    from model_selection import model_candidates, REQUIRED_MODELS
    from preprocessing import MixedTypeDiscretizer
    from id3 import ID3

POINTS = ('home_points_per_match_5', 'away_points_per_match_5')
GOALS = ('home_goal_diff_per_match_5', 'away_goal_diff_per_match_5')
DRAWS = ('home_draw_rate_all', 'away_draw_rate_all')


def feature_variants() -> dict[str, tuple[str, ...]]:
    current = tuple(NUMERIC_FEATURES)
    return {
        'current': current,
        'plus_points': current + POINTS,
        'plus_goal_difference': current + GOALS,
        'plus_points_and_goals': current + POINTS + GOALS,
        'plus_draw_rates': current + DRAWS,
        'without_home_history': tuple(c for c in current if c != 'home_win_rate_as_home_all'),
        'without_home_h2h': tuple(c for c in current if c != 'home_win_rate_h2h_as_home'),
    }


def build_experiment_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add all-history draw rates; reveal results only AFTER the entire date.

    Draw rates ignore venue and use admitted matches only. No history -> 1/3,
    a fixed neutral prior, not a learned statistic or an exclusive missing code.
    """
    frame = build_causal_match_features(matches)
    histories = defaultdict(lambda: [0, 0])  # matches, draws
    home_rates = np.empty(len(frame))
    away_rates = np.empty(len(frame))
    for _, same_day in frame.groupby('date', sort=True):
        for row in same_day.itertuples():
            for team, output in [(row.home, home_rates), (row.away, away_rates)]:
                total, draws = histories[team]
                output[row.Index] = draws / total if total else 1.0 / 3.0
        for row in same_day.itertuples():
            for team in (row.home, row.away):
                histories[team][0] += 1
                histories[team][1] += int(row.winner == 'E')
    return frame.assign(home_draw_rate_all=home_rates, away_draw_rate_all=away_rates)


def make_feature_pipeline(estimator, representation, columns):
    allowed = set(NUMERIC_FEATURES) | set(POINTS + GOALS + DRAWS)
    if not columns or len(set(columns)) != len(columns) or not set(columns) <= allowed:
        raise ValueError('Solo se admiten los atributos declarados, sin etiquetas ni goles actuales.')
    if representation == 'discrete':
        pre = MixedTypeDiscretizer([], list(columns), n_bins=3, fixed_cuts={
            c: LAST_5_CUTS for c in NUMERIC_FEATURES[:2] if c in columns})
    elif representation == 'continuous':
        pre = ColumnTransformer([('inputs', 'passthrough', list(columns))],
            remainder='drop', verbose_feature_names_out=False).set_output(transform='pandas')
    else:
        raise ValueError('Representación no prevista para estos experimentos.')
    return Pipeline([('preprocessing', pre), ('model', clone(estimator))])


def predict_without_draws(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Restrict the learned THREE-class scores to L/V; exact ties favor L.

    ID3 uses counts in the reached leaf (or deepest node on an unknown branch).
    Other classifiers use their fitted class probabilities. Training labels and
    evaluation truth are never filtered, relabeled or given to this decision rule.
    """
    encoded = pipeline.named_steps['preprocessing'].transform(frame)
    model = pipeline.named_steps['model']
    if isinstance(model, ID3):
        scores = []
        for row in np.asarray(encoded):
            node = model.raiz_
            while node.atributo is not None:
                child = node.ramas.get(int(row[node.atributo]))
                if child is None:
                    break
                node = child
            scores.append([node.conteos.get('L', 0), node.conteos.get('V', 0)])
        scores = np.asarray(scores).reshape(-1, 2)
    else:
        probabilities = model.predict_proba(encoded)
        scores = np.column_stack([
            probabilities[:, list(model.classes_).index(c)] if c in model.classes_
            else np.zeros(len(frame)) for c in ('L', 'V')
        ])
    return np.asarray(['L', 'V'])[np.argmax(scores, axis=1)]


def classification_metrics(y, prediction):
    """Keep all three true classes, including true draws when none are predicted."""
    matrix = confusion_matrix(y, prediction, labels=list(CLASSES))
    draw_count = matrix[0].sum()
    return {
        'accuracy': accuracy_score(y, prediction),
        'macro_f1': f1_score(y, prediction, labels=list(CLASSES), average='macro', zero_division=0),
        'draw_f1': f1_score(y, prediction, labels=['E'], average=None, zero_division=0)[0],
        'draw_recall': matrix[0, 0] / draw_count if draw_count else 0.0,
        'true_draws': int(draw_count), 'predicted_draws': int(matrix[:, 0].sum()),
        'confusion_matrix': json.dumps(matrix.tolist()),
    }


def fixed_candidates(selected: pd.DataFrame, models=REQUIRED_MODELS):
    """Resolve the prior selection against declared candidates, never test data."""
    result = []
    candidates = model_candidates()
    for name in models:
        rows = selected.loc[selected.model.eq(name)]
        if len(rows) != 1:
            raise ValueError('Se requiere una selección previa por modelo: ' + name)
        params = json.loads(rows.iloc[0].parameters)
        matches = [c for c in candidates if c.model == name and c.parameters == params]
        if len(matches) != 1:
            raise ValueError('Parámetros previos no reconocidos: ' + name)
        result.append(matches[0])
    return result


def summarize_experiments(details):
    rows = []
    for _, group in details.groupby(['model', 'variant', 'decision'], sort=False):
        if sorted(group.validacion.tolist()) != [2021, 2022, 2023]:
            raise ValueError('Cada comparación requiere exactamente los tres folds comunes.')
        first = group.iloc[0]
        row = {k: first[k] for k in ('model', 'variant', 'variant_rank', 'n_features', 'parameters', 'decision')}
        for metric in ('accuracy', 'macro_f1', 'train_accuracy', 'train_macro_f1', 'draw_recall', 'draw_f1'):
            row[metric + '_mean'] = float(group[metric].mean())
            row[metric + '_std'] = float(group[metric].std(ddof=0))
        row['true_draws_total'] = int(group.true_draws.sum())
        row['predicted_draws_total'] = int(group.predicted_draws.sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    # Diagnostic is explicitly excluded from model/feature selection.
    regular = summary.loc[summary.decision.eq('three_class')]
    ranked = regular.sort_values(['macro_f1_mean', 'n_features', 'variant_rank', 'model'],
                                 ascending=[False, True, True, True], kind='stable')
    selected = ranked.groupby('model', sort=False).head(1).reset_index(drop=True)
    return summary, selected


def run_feature_experiments(frame, selected, progress=print, *, models=REQUIRED_MODELS):
    folds = make_temporal_folds(frame)  # Reject test before any estimator is fit.
    details, predictions = [], []
    candidates = fixed_candidates(selected, models=models)
    for rank, (variant, columns) in enumerate(feature_variants().items()):
        for candidate in candidates:
            if progress:
                progress(f'{variant} / {candidate.model}', flush=True)
            for fold in folds:
                tr = frame.iloc[list(fold.train_positions)]
                va = frame.iloc[list(fold.validation_positions)]
                pipeline = make_feature_pipeline(candidate.estimator, candidate.representation, columns)
                pipeline.fit(tr, tr.winner)
                normal = pipeline.predict(va)
                no_draw = predict_without_draws(pipeline, va)
                train_normal = pipeline.predict(tr)
                train_no_draw = predict_without_draws(pipeline, tr)
                for decision, pred, train_pred in [('three_class', normal, train_normal),
                                                    ('never_draw', no_draw, train_no_draw)]:
                    metrics = classification_metrics(va.winner, pred)
                    details.append(dict(
                        model=candidate.model, parameters=json.dumps(candidate.parameters, sort_keys=True),
                        variant=variant, variant_rank=rank, n_features=len(columns), decision=decision,
                        validacion=fold.validation_year, n_train=len(tr), n_validacion=len(va),
                        train_accuracy=accuracy_score(tr.winner, train_pred),
                        train_macro_f1=f1_score(tr.winner, train_pred, labels=list(CLASSES),
                                                average='macro', zero_division=0), **metrics))
                predictions.append(pd.DataFrame({
                    'model': candidate.model, 'variant': variant, 'validacion': fold.validation_year,
                    'position': fold.validation_positions, 'date': va.date.to_numpy(),
                    'truth': va.winner.to_numpy(), 'three_class': normal, 'never_draw': no_draw,
                }))
    details = pd.DataFrame(details)
    summary, chosen = summarize_experiments(details)
    return details, summary, chosen, pd.concat(predictions, ignore_index=True)


def plot_tradeoffs(summary, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    colors = plt.get_cmap('tab10').colors
    for axis, model in zip(axes.flat, REQUIRED_MODELS):
        data = summary.loc[summary.model.eq(model)]
        for rank, name in enumerate(feature_variants()):
            pair = data.loc[data.variant.eq(name)].set_index('decision')
            regular, diagnostic = pair.loc['three_class'], pair.loc['never_draw']
            axis.plot([regular.accuracy_mean, diagnostic.accuracy_mean],
                      [regular.macro_f1_mean, diagnostic.macro_f1_mean], color=colors[rank], alpha=.6)
            axis.scatter(regular.accuracy_mean, regular.macro_f1_mean, color=colors[rank], marker='o')
            axis.scatter(diagnostic.accuracy_mean, diagnostic.macro_f1_mean, color=colors[rank], marker='x')
            axis.annotate(str(rank), (regular.accuracy_mean, regular.macro_f1_mean), xytext=(4, 4),
                          textcoords='offset points', fontsize=9)
        axis.set(title=model, xlabel='Accuracy medio de validación', ylabel='Macro-F1 medio (E/L/V)')
        axis.grid(alpha=.25)
    labels = ['0: actuales', '1: + puntos', '2: + diferencia de gol', '3: + ambos',
              '4: + tasas de empate', '5: sin historial local', '6: sin H2H local']
    handles = [plt.Line2D([], [], color=colors[i], marker='o', linestyle='', label=l)
               for i,l in enumerate(labels)]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, fontsize=9)
    fig.suptitle('Validación 2021–2023: círculo = tres clases; cruz = nunca predice empate', fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)
