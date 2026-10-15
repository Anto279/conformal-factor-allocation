"""Étape 6 : test go/no-go sur la période d'exploration (1963-1989) uniquement.

On ne regarde AUCUNE performance de portefeuille. On mesure seulement :
- la couverture réalisée des fourchettes (sont-elles bien calibrées ?) ;
- leur largeur ;
- la fréquence à laquelle chaque règle de décision se déclencherait ;
- la couverture par régime de volatilité (risque 3) ;
- le lien entre le score de décision et la volatilité (risque 4).

Critère fixé à l'avance dans docs/protocole.md : la règle binaire est retenue
seulement si elle se déclenche sur au moins 10 % des couples (mois, facteur).

Usage : python -m scripts.go_no_go
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.conformal import conformal_intervals, decision_scores
from src.data.build_factors import STYLES, load_daily, load_monthly
from src.features import FEATURES, build_features
from src.models import walk_forward
from src.plotting import COLORS, FACTOR_LABELS, save, style
from src.split import exploration

RESULTS = Path("results")
FIRST_FORECAST = pd.Timestamp("1970-12-31")
GO_THRESHOLD = 0.10
ALPHAS = {"80 %": 0.20, "50 %": 0.50}
ACI_GAMMA = 0.005


def summarize(iv: pd.DataFrame, dec: pd.DataFrame) -> dict:
    covered = iv["covered"].dropna()
    half = (iv["upper"] - iv["lower"]) / 2
    by_month = dec["binary"].abs().groupby(level="date").max()
    return {
        "couverture": covered.mean(),
        "demi_largeur_moy": half.mean(),
        "prevision_abs_moy": iv["muhat"].abs().mean(),
        "binaire_declenche": (dec["binary"] != 0).mean(),
        "mois_avec_un_declenchement": (by_month > 0).mean(),
        "continu_abs_moy": dec["continuous"].abs().mean(),
        "continu_sup_025": (dec["continuous"].abs() > 0.25).mean(),
        "n": int(len(covered)),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    monthly = exploration(load_monthly())
    daily = exploration(load_daily())
    panel = build_features(monthly, daily, STYLES)
    panel.to_parquet(RESULTS / "panel_exploration.parquet")

    forecasts = {m: walk_forward(panel, m, FEATURES, FIRST_FORECAST) for m in ("mean", "ridge")}

    rows, intervals = [], {}
    for model, zhat in forecasts.items():
        for label, alpha in ALPHAS.items():
            for method, gamma in (("CP", 0.0), ("ACI", ACI_GAMMA)):
                iv = conformal_intervals(panel["z"], zhat, panel["sigma"], alpha, gamma)
                dec = decision_scores(iv)
                intervals[(model, label, method)] = (iv, dec)
                rows.append({"modele": model, "niveau": label, "methode": method, **summarize(iv, dec)})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "go_no_go.csv", index=False)

    # Détail du cas principal : ridge, 80 %, CP normalisée simple
    iv, dec = intervals[("ridge", "80 %", "CP")]
    by_factor = pd.DataFrame({
        "couverture": iv["covered"].groupby(level="factor").mean(),
        "demi_largeur_moy": ((iv["upper"] - iv["lower"]) / 2).groupby(level="factor").mean(),
        "binaire_declenche": (dec["binary"] != 0).groupby(level="factor").mean(),
        "continu_abs_moy": dec["continuous"].abs().groupby(level="factor").mean(),
    })
    by_factor.to_csv(RESULTS / "go_no_go_par_facteur.csv")

    # Couverture par régime : terciles de la volatilité prévue, définis facteur par facteur
    regime = iv.groupby(level="factor")["sigma"].transform(
        lambda s: pd.qcut(s, 3, labels=["basse", "moyenne", "haute"]))
    cov_regime = {}
    for key in [("ridge", "80 %", "CP"), ("ridge", "80 %", "ACI")]:
        ivk = intervals[key][0]
        cov_regime[key[2]] = ivk["covered"].groupby(regime.reindex(ivk.index), observed=True).mean()
    cov_regime = pd.DataFrame(cov_regime)
    cov_regime.to_csv(RESULTS / "couverture_par_regime.csv")

    # Risque 4 : le score n'est-il que l'inverse de la volatilité ?
    corr_sigma = {
        m: float(np.corrcoef(intervals[(m, "80 %", "CP")][1]["continuous"].abs(),
                             intervals[(m, "80 %", "CP")][0]["sigma"])[0, 1])
        for m in forecasts
    }

    main_case = table.query("modele == 'ridge' and niveau == '80 %' and methode == 'CP'").iloc[0]
    verdict = {
        "regle_binaire_frequence": float(main_case["binaire_declenche"]),
        "seuil": GO_THRESHOLD,
        "regle_binaire_retenue": bool(main_case["binaire_declenche"] >= GO_THRESHOLD),
        "correlation_score_sigma": corr_sigma,
        "periode": [str(iv.index.get_level_values("date").min().date()),
                    str(iv.index.get_level_values("date").max().date())],
    }
    with open(RESULTS / "go_no_go_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print(table.round(3).to_string(index=False))
    print(by_factor.round(3))
    print(cov_regime.round(3))
    print(json.dumps(verdict, indent=2))

    figures(intervals, panel)


def figures(intervals: dict, panel: pd.DataFrame) -> None:
    style()
    iv, dec = intervals[("ridge", "80 %", "CP")]

    # 1. La fourchette dans le temps, pour un facteur
    f = "MOM"
    d = iv.xs(f, level="factor")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(d.index, d["lower"] * 100, d["upper"] * 100, color=COLORS["band"], alpha=0.6,
                    label="fourchette à 80 % (CP normalisée)")
    ax.plot(d.index, d["muhat"] * 100, color=COLORS["main"], lw=1.2, label="prévision ridge")
    ax.scatter(d.index, d["target"] * 100, s=4, color=COLORS["grey"], label="rendement réalisé")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("rendement mensuel (%)")
    ax.set_title(f"{FACTOR_LABELS[f]} : prévision et fourchette, période d'exploration")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), fontsize=8, ncol=3)
    save(fig, "gonogo_fourchette_mom")

    # 2. Distribution du score continu
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(dec["continuous"], bins=40, color=COLORS["main"], alpha=0.85)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("score de confiance  centre / demi-largeur")
    ax.set_ylabel("nombre de couples (mois, facteur)")
    share = (dec["binary"] != 0).mean()
    ax.set_title(f"Score continu (ridge, 80 %) ; la règle binaire ne se déclenche que {share:.1%} du temps")
    save(fig, "gonogo_score_continu")

    # 3. Couverture glissante sur 36 mois, CP contre ACI
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for method, color in (("CP", COLORS["main"]), ("ACI", COLORS["alt"])):
        c = intervals[("ridge", "80 %", method)][0]["covered"].groupby(level="date").mean()
        ax.plot(c.rolling(36).mean() * 100, color=color, label=method)
    ax.axhline(80, color="black", lw=0.8, ls="--", label="niveau visé")
    ax.set_ylabel("couverture sur 36 mois (%)")
    ax.set_title("Couverture réalisée des fourchettes à 80 %, 5 facteurs de style")
    ax.legend(fontsize=8)
    save(fig, "gonogo_couverture_glissante")


if __name__ == "__main__":
    main()
