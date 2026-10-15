"""Modèles de prévision du rendement normalisé z_{f,t} = r_{f,t+1} / sigma_{f,t}.

Marche 0 (référence nulle) : moyenne historique de z pour chaque facteur.
Marche 1 (modèle principal) : ridge en panel commun aux facteurs,

    z_{f,t} = a_f + b' x_{f,t} + e_{f,t+1},     x = (mom, vol)
    b = argmin  somme (z - a_f - b'x)² + lambda * ||b||²

Les constantes a_f (prime moyenne propre à chaque facteur) ne sont pas
pénalisées : on centre z et x facteur par facteur, on estime b sans
constante, puis a_f = moyenne(z_f) - b' moyenne(x_f). Les variables sont
standardisées sur l'échantillon d'apprentissage uniquement.

Chaque prévision faite à la fin du mois t n'utilise que les couples (x_s, z_s)
dont la cible est déjà observée, c'est-à-dire s + 1 <= t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Pénalité en unités de n : avec des variables standardisées, X'X ≈ n·I, donc
# lambda = 1 divise à peu près les coefficients des moindres carrés par deux.
RIDGE_LAMBDA = 1.0


def _fit_ridge(train: pd.DataFrame, features: list[str], lam: float) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    X = train[features].to_numpy()
    mu_x, sd_x = X.mean(axis=0), X.std(axis=0)
    sd_x[sd_x == 0] = 1.0
    Xs = (X - mu_x) / sd_x
    groups = train.index.get_level_values("factor")
    Xd = Xs.copy()
    zd = train["z"].to_numpy().copy()
    means_x, means_z = {}, {}
    for f in np.unique(groups):
        m = groups == f
        means_x[f], means_z[f] = Xs[m].mean(axis=0), zd[m].mean()
        Xd[m] -= means_x[f]
        zd[m] -= means_z[f]
    n = len(zd)
    b = np.linalg.solve(Xd.T @ Xd + lam * n * np.eye(Xd.shape[1]), Xd.T @ zd)
    intercepts = {f: means_z[f] - b @ means_x[f] for f in means_z}
    return intercepts, b, mu_x, sd_x


def walk_forward(panel: pd.DataFrame, model: str, features: list[str], start: pd.Timestamp,
                 min_train_months: int = 60, lam: float = RIDGE_LAMBDA) -> pd.Series:
    """Prévision hors échantillon de z, fin de mois par fin de mois à partir de start."""
    data = panel.dropna(subset=features + ["z"])
    dates = data.index.get_level_values("date")
    usable = panel.dropna(subset=features + ["sigma"])
    forecast_dates = sorted(d for d in usable.index.get_level_values("date").unique() if d >= start)
    preds = {}
    for t in forecast_dates:
        # cibles observées à t : les lignes s avec s < t (z_s = r_{s+1}/sigma_s, connu en s+1 <= t)
        train = data[dates < t]
        if train.index.get_level_values("date").nunique() < min_train_months:
            continue
        today = usable.xs(t, level="date", drop_level=False)
        if model == "mean":
            means = train.groupby(level="factor")["z"].mean()
            for (d, f) in today.index:
                preds[(d, f)] = means.get(f, np.nan)
        elif model == "ridge":
            a, b, mu_x, sd_x = _fit_ridge(train, features, lam)
            Xs = (today[features].to_numpy() - mu_x) / sd_x
            for k, (d, f) in enumerate(today.index):
                preds[(d, f)] = a.get(f, 0.0) + Xs[k] @ b
        else:
            raise ValueError(model)
    s = pd.Series(preds, dtype=float)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["date", "factor"])
    return s.sort_index()
