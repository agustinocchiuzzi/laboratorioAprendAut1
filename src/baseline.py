"""Baseline sobre las mismas tasas causales pre-partido que los modelos.

Clase Python simple, sin herencia de scikit-learn: se ajusta y se predice a
mano. fit no aprende parametros: solo recuerda las clases del train.
"""

import numpy as np

BASELINE_FEATURES = ["home_win_rate_10y", "away_win_rate_10y"]


class TenYearWinRateClassifier:
    """Compara las tasas de victoria causales de los ultimos diez anos.

    predict compara las dos tasas: gana el equipo con la tasa mayor; si son
    iguales, gana el local (a menos que se configure otro despliegue). Las
    tasas provienen de features.py, que solo usa partidos anteriores.
    """

    def __init__(self, tie_break="home"):
        self.tie_break = tie_break

    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict(self, X):
        rates = X.loc[:, BASELINE_FEATURES].to_numpy(dtype=float)
        home, away = rates[:, 0], rates[:, 1]
        return np.where(home > away, "L", np.where(away > home, "V",
                        "L" if self.tie_break == "home" else "V"))