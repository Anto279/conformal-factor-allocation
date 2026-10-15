"""Volatilité GARCH(1,1) prévue pour le mois suivant, sans information future.

Modèle sur les rendements quotidiens r_d (en %) :
    r_d = mu + eps_d,   eps_d = sigma_d * z_d
    sigma²_{d+1} = omega + alpha * eps_d² + beta * sigma²_d

Prévision à h jours, avec V = omega / (1 - alpha - beta) la variance de long terme :
    E_d[sigma²_{d+h}] = V + (alpha + beta)^(h-1) * (sigma²_{d+1} - V)

Variance du mois suivant = somme des h = 1..H prévisions (H = 21 jours ouvrés).

Les paramètres sont réestimés chaque fin d'année sur toutes les données
disponibles à cette date (fenêtre croissante), puis utilisés tels quels pour
les 12 fins de mois suivantes. La récurrence est causale : sigma²_{d+1} ne
dépend que des rendements jusqu'au jour d.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model

H_DAYS = 21
MIN_FIT_DAYS = 250


def fit_garch(returns_pct: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(returns_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(
            disp="off", show_warning=False
        )
    p = res.params
    alpha, beta = float(p["alpha[1]"]), float(p["beta[1]"])
    if alpha + beta >= 0.999:  # garde-fou : persistance quasi intégrée
        beta = 0.999 - alpha
    return {"mu": float(p["mu"]), "omega": float(p["omega"]), "alpha": alpha, "beta": beta,
            "var0": float(np.var(returns_pct))}


def _filter(eps: np.ndarray, prm: dict) -> np.ndarray:
    """sigma²_{d+1} pour chaque jour d (variance prévue pour le lendemain)."""
    out = np.empty_like(eps)
    s2 = prm["var0"]
    for d, e in enumerate(eps):
        s2 = prm["omega"] + prm["alpha"] * e * e + prm["beta"] * s2
        out[d] = s2
    return out


def _monthly_variance(next_day_var: float, prm: dict, h: int = H_DAYS) -> float:
    persistence = prm["alpha"] + prm["beta"]
    v_long = prm["omega"] / (1.0 - persistence)
    k = np.arange(h)
    return float(np.sum(v_long + persistence**k * (next_day_var - v_long)))


def garch_monthly_vol(daily: pd.Series, month_ends: pd.DatetimeIndex) -> pd.Series:
    """Volatilité prévue (décimale, non annualisée) du mois t+1, connue à la fin du mois t."""
    r = daily.dropna() * 100.0
    out = pd.Series(np.nan, index=month_ends, dtype=float)
    years = sorted({d.year for d in month_ends})
    for year in years:
        fit_end = pd.Timestamp(year - 1, 12, 31)
        sample = r[r.index <= fit_end]
        if len(sample) < MIN_FIT_DAYS:
            continue
        prm = fit_garch(sample.to_numpy())
        months = month_ends[month_ends.year == year]
        upto = r[r.index <= months.max()]
        next_var = pd.Series(_filter(upto.to_numpy() - prm["mu"], prm), index=upto.index)
        for t in months:
            known = next_var[next_var.index <= t]
            if known.empty:
                continue
            out[t] = np.sqrt(_monthly_variance(known.iloc[-1], prm)) / 100.0
    return out
