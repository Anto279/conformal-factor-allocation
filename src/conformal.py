"""Fourchettes de prévision par Conformal Prediction normalisée.

Score de non-conformité (erreur rapportée à la volatilité prévue) :
    s_{f,t} = | z_{f,t} - zhat_{f,t} | = | r_{f,t+1} - muhat_{f,t} | / sigma_{f,t}

Fourchette au niveau 1 - alpha, connue à la fin du mois t :
    [ muhat_{f,t} - q_t * sigma_{f,t} ;  muhat_{f,t} + q_t * sigma_{f,t} ]
avec muhat = sigma * zhat et q_t le quantile empirique d'ordre
ceil((n+1)(1-alpha))/n des n scores déjà observés à t (toutes dates s < t,
tous facteurs confondus).

Variante adaptative (ACI, Gibbs et Candès, 2021) : le niveau visé évolue
    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
où err_t est la part des facteurs dont le rendement est sorti de la
fourchette au mois t. Après des erreurs, alpha_t baisse et la fourchette
s'élargit ; après une période calme, elle se resserre.

Garantie : la couverture moyenne sur longue période vaut 1 - alpha (à
l'échangeabilité près pour la version simple, sans hypothèse pour l'ACI).
Ce n'est pas une garantie mois par mois.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """k-ième plus petit score, k = ceil((n+1)(1-alpha)) : correction de taille finie."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(np.partition(scores, max(k, 1) - 1)[max(k, 1) - 1])


def conformal_intervals(z: pd.Series, zhat: pd.Series, sigma: pd.Series, alpha: float,
                        gamma: float = 0.0, min_scores: int = 300,
                        window: int | None = None) -> pd.DataFrame:
    """Fourchettes en ligne, date par date, sur le panel (date, factor).

    gamma = 0 donne la CP normalisée simple ; gamma > 0 donne l'ACI.
    window = nombre de mois de scores conservés (None : tout l'historique).
    """
    df = pd.DataFrame({"z": z, "zhat": zhat, "sigma": sigma}).dropna(subset=["zhat", "sigma"])
    dates = sorted(df.index.get_level_values("date").unique())
    past_scores: list[float] = []
    n_factors = df.index.get_level_values("factor").nunique()
    keep = None if window is None else window * n_factors
    alpha_t = alpha
    rows = []
    for t in dates:
        today = df.xs(t, level="date", drop_level=False)
        if len(past_scores) >= min_scores:
            q = conformal_quantile(np.asarray(past_scores), min(max(alpha_t, 1e-3), 0.999))
            muhat = today["sigma"] * today["zhat"]
            lo, hi = muhat - q * today["sigma"], muhat + q * today["sigma"]
            out = pd.DataFrame({"muhat": muhat, "lower": lo, "upper": hi, "q": q,
                                "alpha_t": alpha_t, "sigma": today["sigma"]})
            realized = today["z"] * today["sigma"]
            out["target"] = realized
            out["covered"] = ((realized >= lo) & (realized <= hi)).where(realized.notna())
            rows.append(out)
            err = 1.0 - out["covered"].dropna().mean() if out["covered"].notna().any() else np.nan
            if gamma > 0 and not np.isnan(err):
                alpha_t = alpha_t + gamma * (alpha - err)
        # les scores du mois t ne sont connus qu'à t+1 : on les ajoute après usage
        observed = today.dropna(subset=["z"])
        past_scores.extend(np.abs(observed["z"] - observed["zhat"]).tolist())
        if keep is not None and len(past_scores) > keep:
            past_scores = past_scores[-keep:]
    return pd.concat(rows).sort_index()


def decision_scores(iv: pd.DataFrame) -> pd.DataFrame:
    """Deux règles de déviation par rapport à l'ERC, pour chaque (date, facteur).

    binaire  : s = signe(muhat) si la fourchette exclut zéro, 0 sinon
    continue : s = clip( muhat / (q * sigma), -1, 1 ) = clip( centre / demi-largeur, -1, 1 )
    """
    half = (iv["upper"] - iv["lower"]) / 2
    out = pd.DataFrame(index=iv.index)
    out["binary"] = np.where(iv["lower"] > 0, 1.0, np.where(iv["upper"] < 0, -1.0, 0.0))
    out["continuous"] = (iv["muhat"] / half).clip(-1, 1)
    return out
