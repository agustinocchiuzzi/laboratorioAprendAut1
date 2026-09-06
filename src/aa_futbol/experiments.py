"""End-to-end temporal model selection and final 2024-2025 evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from aa_futbol.data import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    DEFAULT_REPORT,
    prepare_dataset,
)
from aa_futbol.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    build_causal_match_features,
)
from aa_futbol.model_selection import DateBlockedTimeSeriesSplit
from aa_futbol.models import (
    CategoricalDecisionTreeClassifier,
    MEstimateCategoricalNB,
    TenYearWinRateClassifier,
)
from aa_futbol.preprocessing import MixedTypeDiscretizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results"
LABELS = ["E", "L", "V"]
RANDOM_STATE = 42


@dataclass(frozen=True)
class ExperimentSpec:
    estimator: object
    parameter_grid: dict[str, list[object]] | None
    primary_parameter: str | None


def _discrete_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            (
                "discretizer",
                MixedTypeDiscretizer(CATEGORICAL_FEATURES, NUMERIC_FEATURES, n_bins=5),
            ),
            ("model", model),
        ]
    )


def build_experiment_specs(profile: str = "quick") -> dict[str, ExperimentSpec]:
    """Create comparable estimators and their temporal-CV search spaces."""
    if profile not in {"quick", "full"}:
        raise ValueError("profile debe ser 'quick' o 'full'.")

    if profile == "quick":
        m_values = [0.1, 1.0, 10.0]
        gain_values = [0.0, 0.002, 0.01]
        alpha_values = [0.1, 1.0, 10.0]
        forest_grid = {
            "model__max_depth": [10, None],
            "model__min_samples_leaf": [1, 5],
        }
        forest_trees = 200
    else:
        m_values = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0]
        gain_values = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
        alpha_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
        forest_grid = {
            "model__max_depth": [10, 20, None],
            "model__min_samples_leaf": [1, 5, 10],
        }
        forest_trees = 500

    forest_preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )

    return {
        "baseline_10y": ExperimentSpec(TenYearWinRateClassifier(), None, None),
        "naive_bayes_custom": ExperimentSpec(
            _discrete_pipeline(MEstimateCategoricalNB()),
            {"model__m": m_values},
            "model__m",
        ),
        "decision_tree_custom": ExperimentSpec(
            _discrete_pipeline(CategoricalDecisionTreeClassifier(max_depth=8)),
            {"model__min_info_gain": gain_values},
            "model__min_info_gain",
        ),
        "naive_bayes_sklearn": ExperimentSpec(
            _discrete_pipeline(CategoricalNB()),
            {"model__alpha": alpha_values},
            "model__alpha",
        ),
        "random_forest_sklearn": ExperimentSpec(
            Pipeline(
                [
                    ("preprocessor", forest_preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=forest_trees,
                            class_weight="balanced_subsample",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            forest_grid,
            "model__max_depth",
        ),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _save_confusion_matrix(
    y_true: pd.Series,
    predictions: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> list[list[int]]:
    matrix = confusion_matrix(y_true, predictions, labels=LABELS)
    display = ConfusionMatrixDisplay(matrix, display_labels=LABELS)
    display.plot(cmap="Blues", colorbar=False)
    display.ax_.set_title(model_name.replace("_", " ").title())
    display.ax_.set_xlabel("Etiqueta predicha")
    display.ax_.set_ylabel("Etiqueta real")
    display.figure_.tight_layout()
    display.figure_.savefig(
        output_dir / f"confusion_{model_name}.png", dpi=180, bbox_inches="tight"
    )
    plt.close(display.figure_)
    return matrix.tolist()


def _save_validation_results(
    model_name: str,
    search: GridSearchCV,
    primary_parameter: str,
    output_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for params, mean_score, std_score, rank in zip(  # noqa: B905
        search.cv_results_["params"],
        search.cv_results_["mean_test_score"],
        search.cv_results_["std_test_score"],
        search.cv_results_["rank_test_score"],
    ):
        display_parameters = {
            key.replace("model__", "").replace("min_samples_leaf", "min_leaf"): value
            for key, value in params.items()
        }
        display_value = "; ".join(
            f"{key}={value}" for key, value in display_parameters.items()
        )
        rows.append(
            {
                "model": model_name,
                "parameters": json.dumps(_json_safe(params), sort_keys=True),
                "primary_parameter": primary_parameter,
                "primary_value": str(params[primary_parameter]),
                "display_value": display_value,
                "mean_macro_f1": float(mean_score),
                "mean_error": float(1.0 - mean_score),
                "std_macro_f1": float(std_score),
                "rank": int(rank),
            }
        )

    frame = pd.DataFrame(rows).sort_values("rank", kind="stable")
    frame.to_csv(output_dir / f"validation_{model_name}.csv", index=False)

    ordered = frame.sort_values(
        ["primary_value", "parameters"], kind="stable"
    ).reset_index(drop=True)
    multiple_parameters = (
        ordered["parameters"].map(lambda value: len(json.loads(value)) > 1).any()
    )
    x = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.errorbar(
        x,
        ordered["mean_error"],
        yerr=ordered["std_macro_f1"],
        marker="o",
        linestyle="none" if multiple_parameters else "-",
        capsize=3,
    )
    tick_labels = (
        ordered["display_value"].str.replace("; ", "\n", regex=False)
        if multiple_parameters
        else ordered["display_value"]
    )
    axis.set_xticks(
        x,
        tick_labels,
        rotation=0 if multiple_parameters else 30,
        ha="center" if multiple_parameters else "right",
    )
    axis.set_xlabel(
        "Combinacion de hiperparametros"
        if multiple_parameters
        else primary_parameter.replace("model__", "")
    )
    axis.set_ylabel("Error de validacion (1 - macro-F1)")
    axis.set_title(model_name.replace("_", " ").title())
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_dir / f"validation_{model_name}.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)
    return rows


def run_experiments(
    *,
    raw_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_RESULTS,
    profile: str = "quick",
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object], pd.DataFrame]:
    """Run temporal CV, refit through 2023, and evaluate on 2024-2025."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    matches, cleaning_report = prepare_dataset(raw_path, DEFAULT_OUTPUT, DEFAULT_REPORT)
    featured = build_causal_match_features(matches)
    train = featured.loc[featured["year"] <= 2023].reset_index(drop=True)
    test = featured.loc[featured["year"].between(2024, 2025)].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("La particion temporal produjo un conjunto vacio.")

    X_train = train.drop(columns="winner")
    y_train = train["winner"]
    X_test = test.drop(columns="winner")
    y_test = test["winner"]
    n_splits = 3 if profile == "quick" else 5
    cv = DateBlockedTimeSeriesSplit(n_splits=n_splits)
    scorer = make_scorer(
        f1_score,
        average="macro",
        labels=LABELS,
        pos_label=None,
        zero_division=0,
    )
    specs = build_experiment_specs(profile)

    summary_rows: list[dict[str, object]] = []
    details: dict[str, object] = {}
    fitted_models: dict[str, object] = {}
    prediction_table = test.loc[
        :, ["date", "home", "away", "winner"]
    ].copy()

    for model_name, spec in specs.items():
        if spec.parameter_grid is None:
            cv_scores = cross_val_score(
                clone(spec.estimator),
                X_train,
                y_train,
                cv=cv,
                scoring=scorer,
                n_jobs=n_jobs,
            )
            fitted = clone(spec.estimator).fit(X_train, y_train)
            best_params: dict[str, object] = {}
            cv_macro_f1 = float(np.mean(cv_scores))
            validation_rows: list[dict[str, object]] = [
                {
                    "model": model_name,
                    "fold_macro_f1": [float(score) for score in cv_scores],
                    "mean_macro_f1": cv_macro_f1,
                    "mean_error": 1.0 - cv_macro_f1,
                }
            ]
        else:
            search = GridSearchCV(
                clone(spec.estimator),
                spec.parameter_grid,
                cv=cv,
                scoring=scorer,
                refit=True,
                n_jobs=n_jobs,
                return_train_score=False,
            )
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            best_params = _json_safe(search.best_params_)
            cv_macro_f1 = float(search.best_score_)
            validation_rows = _save_validation_results(
                model_name,
                search,
                str(spec.primary_parameter),
                output_dir,
            )

        predictions = fitted.predict(X_test)
        prediction_table[f"prediction_{model_name}"] = predictions
        report = classification_report(
            y_test,
            predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
        matrix = _save_confusion_matrix(y_test, predictions, model_name, output_dir)
        summary_rows.append(
            {
                "model": model_name,
                "best_parameters": json.dumps(best_params, sort_keys=True),
                "cv_macro_f1": cv_macro_f1,
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_macro_f1": f1_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "precision_E": report["E"]["precision"],
                "recall_E": report["E"]["recall"],
                "f1_E": report["E"]["f1-score"],
                "precision_L": report["L"]["precision"],
                "recall_L": report["L"]["recall"],
                "f1_L": report["L"]["f1-score"],
                "precision_V": report["V"]["precision"],
                "recall_V": report["V"]["recall"],
                "f1_V": report["V"]["f1-score"],
            }
        )
        details[model_name] = {
            "best_parameters": best_params,
            "classification_report": _json_safe(report),
            "confusion_matrix_labels": LABELS,
            "confusion_matrix": matrix,
            "validation": validation_rows,
        }
        fitted_models[model_name] = fitted
        joblib.dump(fitted, output_dir / "models" / f"{model_name}.joblib")

    summary = pd.DataFrame(summary_rows).sort_values(
        "test_macro_f1", ascending=False, kind="stable"
    )
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    prediction_table.to_csv(output_dir / "test_predictions.csv", index=False)
    (output_dir / "experiment_details.json").write_text(
        json.dumps(_json_safe(details), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "random_state": RANDOM_STATE,
        "profile": profile,
        "cv_splits": n_splits,
        "n_jobs": n_jobs,
        "training": {
            "rows": len(train),
            "first_date": train["date"].min().date().isoformat(),
            "last_date": train["date"].max().date().isoformat(),
        },
        "evaluation": {
            "rows": len(test),
            "first_date": test["date"].min().date().isoformat(),
            "last_date": test["date"].max().date().isoformat(),
        },
        "feature_policy": (
            "Causal online: each match uses only results from strictly earlier dates."
        ),
        "model_features": MODEL_FEATURES,
        "cleaning_report": cleaning_report,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary, details, fitted_models, featured


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, _, _, _ = run_experiments(
        raw_path=args.input,
        output_dir=args.output_dir,
        profile=args.profile,
        n_jobs=args.n_jobs,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
