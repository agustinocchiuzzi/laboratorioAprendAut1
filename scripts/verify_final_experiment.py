"""Recompute the frozen final evaluation without selecting from test data.

The validation record is an input contract.  This script only fits the recorded
configurations on matches through 2023 and writes compact report artifacts.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.naive_bayes import CategoricalNB

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baseline import BASELINE_FEATURES, TenYearWinRateClassifier
from evaluation import CLASSES, new_discretizer, temporal_holdout
from features import NUMERIC_FEATURES, build_causal_match_features, load_clean_matches
from id3 import ID3
from naive_bayes import MEstimateCategoricalNB


EXTRA = [
    "home_points_per_match_5", "away_points_per_match_5",
    "home_goal_diff_per_match_5", "away_goal_diff_per_match_5",
]
EXPECTED = {
    "ID3": {"min_info_gain": 0.005, "columns": NUMERIC_FEATURES + EXTRA},
    "NB propio": {"m": 0.1, "columns": NUMERIC_FEATURES + EXTRA},
    "sklearn CategoricalNB": {"alpha": 0.025, "columns": NUMERIC_FEATURES + EXTRA},
    "Random Forest": {
        "n_estimators": 300, "random_state": 42,
        "class_weight": "balanced_subsample", "columns": NUMERIC_FEATURES,
        "max_depth": None, "min_samples_leaf": 20,
    },
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_configuration():
    selected = json.loads((ROOT / "results/validation/selected.json").read_text())
    selected.pop("test_evaluated", None)
    if selected != EXPECTED:
        raise RuntimeError("El registro de validacion no coincide con la configuracion congelada.")
    return selected


def causal_leakage_check(matches, featured, test):
    """Changing every score on one test date cannot change that date's inputs."""
    date = test["date"].min()
    altered = matches.copy()
    mask = altered["date"].eq(date)
    altered.loc[mask, ["gh", "ga"]] = altered.loc[mask, ["ga", "gh"]].to_numpy()
    altered.loc[mask, "winner"] = altered.loc[mask, "winner"].map(
        {"L": "V", "V": "L", "E": "E"}
    )
    changed = build_causal_match_features(altered)
    columns = ["home", "away"] + NUMERIC_FEATURES + EXTRA
    before = featured.loc[featured["date"].eq(date), columns].reset_index(drop=True)
    after = changed.loc[changed["date"].eq(date), columns].reset_index(drop=True)
    if not before.equals(after):
        raise AssertionError("Los resultados actuales o del mismo dia cambiaron los atributos.")
    return {"date": str(date.date()), "matches_on_date": int(len(before))}


def id3_contract_check():
    X = np.array([[1], [1], [2], [2]])
    y = np.array(["L", "L", "V", "V"])
    stopped = ID3(min_info_gain=1.0).fit(X, y)
    fallback = ID3(min_info_gain=0.0).fit(X, y)
    if stopped.get_depth() != 0 or fallback.predict([[9]])[0] != "L":
        raise AssertionError("Fallo la parada de ID3 o el fallback de categoria no vista.")
    return {"stopped_depth": stopped.get_depth(), "unseen_prediction": "L"}


def training_only_discretizer_check(discretizer, train, test, columns):
    """Its learned cuts must equal train cuts and remain fixed for test."""
    expected = {}
    for column in columns:
        if column in NUMERIC_FEATURES[:2]:
            edges = np.array([0.3, 0.6])
        else:
            values = pd.to_numeric(train[column], errors="raise").to_numpy(dtype=float)
            edges = np.unique(np.quantile(values, [1 / 3, 2 / 3]))
        if not np.allclose(discretizer.numeric_edges_[column], edges):
            raise AssertionError(f"Los cortes de {column} no provienen solo de train.")
        expected[column] = edges.tolist()
    before = {name: values.tolist() for name, values in discretizer.numeric_edges_.items()}
    discretizer.transform(test[columns])
    after = {name: values.tolist() for name, values in discretizer.numeric_edges_.items()}
    if before != after:
        raise AssertionError("Transformar test modifico el discretizador ajustado en train.")
    return expected


def metric_rows(name, y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    summary = {
        "model": name,
        "n_test": int(len(y_true)),
        "correct": int(np.sum(np.asarray(y_true) == np.asarray(y_pred))),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": precision_recall_fscore_support(
            y_true, y_pred, labels=CLASSES, average="macro", zero_division=0
        )[2],
    }
    by_class = [
        {"model": name, "class": label, "precision": float(p), "recall": float(r),
         "f1": float(score), "support": int(n)}
        for label, p, r, score, n in zip(CLASSES, precision, recall, f1, support)
    ]
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)
    confusion = [
        {"model": name, "actual": actual, "predicted": predicted, "count": int(matrix[i, j])}
        for i, actual in enumerate(CLASSES) for j, predicted in enumerate(CLASSES)
    ]
    return summary, by_class, confusion


def subgroup_rows(test, y_true, predictions):
    signal = np.select(
        [test["home_points_per_match_5"] > test["away_points_per_match_5"],
         test["home_points_per_match_5"] < test["away_points_per_match_5"]],
        ["Local > visitante", "Local < visitante"], default="Local = visitante"
    )
    rows = []
    for group in ("Local > visitante", "Local = visitante", "Local < visitante"):
        mask = signal == group
        for name, prediction in predictions.items():
            rows.append({
                "subgroup": group, "model": name, "n_test": int(mask.sum()),
                "actual_E_L_V": "/".join(str(int((y_true[mask] == label).sum())) for label in CLASSES),
                "correct": int((prediction[mask] == y_true[mask]).sum()),
                "accuracy": accuracy_score(y_true[mask], prediction[mask]),
                "macro_f1": precision_recall_fscore_support(
                    y_true[mask], prediction[mask], labels=CLASSES,
                    average="macro", zero_division=0,
                )[2],
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/futbol_uruguayo.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "results/finalized")
    args = parser.parse_args()
    frozen = frozen_configuration()
    args.output.mkdir(parents=True, exist_ok=True)

    matches = load_clean_matches(args.raw)
    featured = build_causal_match_features(matches)
    train, test = temporal_holdout(featured)
    y_train, y_test = train["winner"].to_numpy(), test["winner"].to_numpy()

    id3_columns = frozen["ID3"]["columns"]
    id3_disc = new_discretizer(id3_columns).fit(train[id3_columns])
    X_id3_train = id3_disc.transform(train[id3_columns])
    X_id3_test = id3_disc.transform(test[id3_columns])
    id3_edges = training_only_discretizer_check(id3_disc, train, test, id3_columns)
    id3 = ID3(min_info_gain=frozen["ID3"]["min_info_gain"]).fit(X_id3_train, y_train)

    nb_columns = frozen["NB propio"]["columns"]
    nb_disc = new_discretizer(nb_columns).fit(train[nb_columns])
    ref_disc = new_discretizer(nb_columns).fit(train[nb_columns])
    X_nb_train, X_nb_test = nb_disc.transform(train[nb_columns]), nb_disc.transform(test[nb_columns])
    X_ref_train, X_ref_test = ref_disc.transform(train[nb_columns]), ref_disc.transform(test[nb_columns])
    own = MEstimateCategoricalNB(m=frozen["NB propio"]["m"], min_categories=4).fit(X_nb_train, y_train)
    reference = CategoricalNB(alpha=frozen["sklearn CategoricalNB"]["alpha"], min_categories=4,
                              force_alpha=True).fit(X_ref_train, y_train)

    rf_spec = frozen["Random Forest"]
    rf = RandomForestClassifier(
        n_estimators=rf_spec["n_estimators"], random_state=rf_spec["random_state"],
        class_weight=rf_spec["class_weight"], max_depth=rf_spec["max_depth"],
        min_samples_leaf=rf_spec["min_samples_leaf"], n_jobs=2,
    ).fit(train[rf_spec["columns"]], y_train)
    baseline = TenYearWinRateClassifier().fit(train[BASELINE_FEATURES], y_train)

    predictions = {
        "ID3": id3.predict(X_id3_test),
        "NB propio": own.predict(X_nb_test),
        "CategoricalNB": reference.predict(X_ref_test),
        "Random Forest": rf.predict(test[rf_spec["columns"]]),
        "Base 10 años": baseline.predict(test[BASELINE_FEATURES]),
    }
    max_probability_difference = float(np.max(np.abs(
        own.predict_proba(X_nb_test) - reference.predict_proba(X_ref_test)
    )))
    if not np.array_equal(predictions["NB propio"], predictions["CategoricalNB"]):
        raise AssertionError("Los NB no producen las mismas predicciones finales.")

    summaries, reports, matrices = [], [], []
    for name, prediction in predictions.items():
        summary, report, matrix = metric_rows(name, y_test, prediction)
        summaries.append(summary)
        reports.extend(report)
        matrices.extend(matrix)

    examples = test[["date", "home", "away", "winner"]].copy()
    for name, prediction in predictions.items():
        key = name.lower().replace(" ", "_").replace("ñ", "n")
        examples[f"prediction_{key}"] = pd.Series(prediction, index=test.index).loc[examples.index]
        examples[f"correct_{key}"] = examples[f"prediction_{key}"].eq(examples["winner"])
    scores = matches[["date", "home", "away", "gh", "ga"]]
    examples = examples.merge(scores, on=["date", "home", "away"], validate="one_to_one")
    examples["score"] = examples["gh"].astype(str) + "-" + examples["ga"].astype(str)
    examples = (
        examples.sort_values(["date", "home", "away"])
        .groupby(["winner", "prediction_nb_propio"], sort=True, as_index=False)
        .head(1)
        .sort_values(["winner", "prediction_nb_propio"])
        .reset_index(drop=True)
    )

    checks = {
        "frozen_validation_configuration": {"passed": True, "selected": frozen},
        "current_match_and_same_day_leakage": {"passed": True, **causal_leakage_check(matches, featured, test)},
        "training_only_discretizer": {"passed": True, "edges": id3_edges},
        "id3_stopping_and_unseen_category": {"passed": True, **id3_contract_check()},
        "nb_equivalence": {"passed": True, "identical_predictions": int(len(y_test)),
                           "max_probability_difference": max_probability_difference},
        "final_fit_propagation": {
            "passed": True, "id3_features": int(X_id3_train.shape[1]),
            "id3_min_info_gain": id3.min_info_gain, "nb_features": int(X_nb_train.shape[1]),
            "nb_m": own.m, "categorical_nb_alpha": reference.alpha,
            "rf_features": int(len(rf_spec["columns"])), "rf_n_estimators": rf.n_estimators,
            "rf_max_depth": rf.max_depth, "rf_min_samples_leaf": rf.min_samples_leaf,
        },
    }
    environment = {
        "python": sys.version.split()[0], "implementation": platform.python_implementation(),
        "platform": platform.platform(), "scikit_learn": sklearn.__version__,
        "dependencies": {name: importlib.metadata.version(name) for name in [
            "numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib",
            "ipykernel", "nbclient", "nbformat",
        ]},
        "raw_zip_sha256": digest(args.raw),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in [
            ROOT / "src/features.py", ROOT / "src/preprocessing.py", ROOT / "src/id3.py",
            ROOT / "src/naive_bayes.py", ROOT / "src/baseline.py", ROOT / "src/evaluation.py",
        ]},
    }
    pd.DataFrame(summaries).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(reports).to_csv(args.output / "class_metrics.csv", index=False)
    pd.DataFrame(matrices).to_csv(args.output / "confusion_matrices.csv", index=False)
    examples.to_csv(args.output / "qualitative_examples.csv", index=False)
    pd.DataFrame(subgroup_rows(test, y_test, predictions)).to_csv(args.output / "subgroup_analysis.csv", index=False)
    (args.output / "checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False) + "\n")
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
