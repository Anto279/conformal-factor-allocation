"""Étape 5 : variables explicatives, toutes connues à la fin du mois t.

Pour chaque facteur f et chaque fin de mois t :

    rv_{f,t}     = sqrt( somme des r_{f,d}² sur les jours d du mois t )
                   volatilité réalisée du mois t (quotidien -> mensuel)
    mom_{f,t}    = somme_{k=0..11} r_{f,t-k} / sqrt( somme_{k=0..11} rv²_{f,t-k} )
                   rendement des 12 derniers mois rapporté à sa volatilité réalisée
                   (un « t-stat » de tendance, comparable d'un facteur à l'autre)
    vol_{f,t}    = log rv_{f,t} - moyenne( log rv_{f,t-60..t-1} )
                   volatilité récente relative à son propre passé
    sigma_{f,t}  = volatilité GARCH(1,1) prévue pour le mois t+1
    target_{f,t} = r_{f,t+1}                        (ce qu'on cherche à prévoir)
    z_{f,t}      = r_{f,t+1} / sigma_{f,t}          (cible normalisée)

Toute variable indexée t n'utilise que des données datées au plus tard de t.
Seules target et z regardent t+1 : ce sont les cibles, jamais des entrées.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.garch import garch_monthly_vol

MOM_WINDOW = 12
VOL_WINDOW = 60
VOL_MIN_PERIODS = 24


def realized_vol(daily: pd.DataFrame) -> pd.DataFrame:
    rv = np.sqrt((daily**2).resample("ME").sum(min_count=1))
    rv.index = rv.index.normalize()
    return rv


def build_features(monthly: pd.DataFrame, daily: pd.DataFrame, factors: list[str],
                   with_garch: bool = True) -> pd.DataFrame:
    """Panel long (date, factor) avec les variables et les cibles."""
    rv = realized_vol(daily[factors]).reindex(monthly.index)
    frames = []
    for f in factors:
        r = monthly[f]
        cum = r.rolling(MOM_WINDOW, min_periods=MOM_WINDOW).sum()
        vol12 = np.sqrt((rv[f] ** 2).rolling(MOM_WINDOW, min_periods=MOM_WINDOW).sum())
        log_rv = np.log(rv[f])
        past_mean = log_rv.shift(1).rolling(VOL_WINDOW, min_periods=VOL_MIN_PERIODS).mean()
        df = pd.DataFrame(
            {
                "ret": r,
                "rv": rv[f],
                "mom": cum / vol12,
                "vol": log_rv - past_mean,
                "target": r.shift(-1),
            },
            index=monthly.index,
        )
        if with_garch:
            df["sigma"] = garch_monthly_vol(daily[f], monthly.index)
            df["z"] = df["target"] / df["sigma"]
        df["factor"] = f
        frames.append(df)
    panel = pd.concat(frames).reset_index().rename(columns={"index": "date"})
    return panel.set_index(["date", "factor"]).sort_index()


FEATURES = ["mom", "vol"]
