"""Description de la base : statistiques, corrélations, figures du README.

Il s'agit de décrire les séries brutes de Kenneth French, pas d'évaluer une
stratégie : aucune règle d'allocation n'est appliquée ici.

Usage : python -m scripts.describe_data
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.build_factors import load_daily, load_monthly
from src.features import realized_vol
from src.garch import garch_monthly_vol
from src.plotting import FACTOR_COLORS, FACTOR_LABELS, save, style
from src.split import EXPLORATION_END, exploration

RESULTS = Path("results")
FACTORS = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]


def stats(df: pd.DataFrame) -> pd.DataFrame:
    m = df[FACTORS]
    out = pd.DataFrame({
        "moyenne mensuelle (%)": m.mean() * 100,
        "volatilité mensuelle (%)": m.std() * 100,
        "Sharpe annualisé": m.mean() / m.std() * np.sqrt(12),
        "t-stat de la moyenne": m.mean() / m.std() * np.sqrt(len(m)),
        "pire mois (%)": m.min() * 100,
        "autocorrélation d'ordre 1": [m[c].autocorr(1) for c in FACTORS],
    })
    return out.round(2)


def to_markdown(df: pd.DataFrame) -> str:
    cols = [""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]) + " |")
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    style()
    monthly, daily = load_monthly(), load_daily()

    periods = {
        "1963-1989 (exploration)": monthly[monthly.index <= EXPLORATION_END],
        "1990-fin (test)": monthly[monthly.index > EXPLORATION_END],
        "ensemble": monthly,
    }
    md = []
    for label, df in periods.items():
        md.append(f"### {label} : {df.index[0]:%m/%Y} - {df.index[-1]:%m/%Y}, {len(df)} mois\n")
        md.append(to_markdown(stats(df)) + "\n")
    corr = exploration(monthly)[FACTORS].corr().round(2)
    md.append("### Corrélations des rendements mensuels, 1963-1989\n")
    md.append(to_markdown(corr) + "\n")
    (RESULTS / "statistiques_descriptives.md").write_text("\n".join(md))
    print("\n".join(md))

    # Figure 1 : rendements cumulés (échelle log)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for f in FACTORS:
        wealth = (1 + monthly[f]).cumprod()
        ax.plot(wealth.index, wealth, color=FACTOR_COLORS[f], lw=1.3, label=FACTOR_LABELS[f])
    ax.set_yscale("log")
    ax.axvspan(monthly.index[0], EXPLORATION_END, color="#f1f0ec", zorder=0)
    ax.text(pd.Timestamp("1965-01-01"), ax.get_ylim()[1] * 0.6, "exploration", color="#52514e", fontsize=8)
    ax.text(pd.Timestamp("1991-01-01"), ax.get_ylim()[1] * 0.6, "test sous scellé", color="#52514e", fontsize=8)
    ax.set_ylabel("valeur de 1 $ investi (échelle log)")
    ax.set_title("Les six facteurs Fama-French + momentum, États-Unis, rendements cumulés")
    ax.legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.08))
    save(fig, "facteurs_cumules")

    # Figure 2 : matrice de corrélation (exploration)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FACTORS)), FACTORS)
    ax.set_yticks(range(len(FACTORS)), FACTORS)
    ax.grid(False)
    for i in range(len(FACTORS)):
        for j in range(len(FACTORS)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else "#0b0b0b")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Corrélations mensuelles, 1963-1989")
    save(fig, "correlations_exploration")

    # Le risque se prévoit-il ? GARCH contre la volatilité du mois passé (exploration)
    m_exp, d_exp = exploration(monthly), exploration(daily)
    rows = {}
    for f in FACTORS:
        rv = realized_vol(d_exp[[f]])[f].reindex(m_exp.index)
        sig = garch_monthly_vol(d_exp[f], m_exp.index)
        df = pd.DataFrame({"next": rv.shift(-1), "garch": sig, "naive": rv}).dropna()
        qlike = lambda h: np.mean(df["next"] ** 2 / h**2 - np.log(df["next"] ** 2 / h**2) - 1)
        rows[f] = {
            "corr(GARCH, vol réalisée t+1)": np.corrcoef(np.log(df["garch"]), np.log(df["next"]))[0, 1],
            "corr(vol réalisée t, t+1)": np.corrcoef(np.log(df["naive"]), np.log(df["next"]))[0, 1],
            "QLIKE GARCH": qlike(df["garch"]),
            "QLIKE naïf": qlike(df["naive"]),
        }
    vol_table = pd.DataFrame(rows).T.round(3)
    md.append("### Prévision de la volatilité du mois suivant, 1963-1989\n")
    md.append(to_markdown(vol_table) + "\n")
    (RESULTS / "statistiques_descriptives.md").write_text("\n".join(md))
    print(vol_table)

    # Figure 3 : volatilité GARCH prévue contre volatilité réalisée (exploration)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, f in zip(axes, ["HML", "MOM"]):
        rv = realized_vol(d_exp[[f]])[f].reindex(m_exp.index)
        sig = garch_monthly_vol(d_exp[f], m_exp.index)
        ax.plot(rv.index, rv * 100, color="#8a8984", lw=0.8, label="volatilité réalisée du mois t+1")
        ax.plot(sig.index + pd.offsets.MonthEnd(1), sig * 100, color=FACTOR_COLORS[f], lw=1.4,
                label="prévision GARCH faite en t")
        ax.set_title(FACTOR_LABELS[f])
        ax.set_ylabel("volatilité mensuelle (%)")
        ax.legend(fontsize=7)
    save(fig, "garch_vs_realisee")


if __name__ == "__main__":
    main()
