"""Télécharge les cours mensuels ajustés (dividendes et splits inclus) utilisés à l'étape 4.

Usage :
    pip install yfinance pandas
    python scripts/telecharger_cours.py

Produit data/raw/cours_mensuels.csv : une ligne par mois, une colonne par ticker,
cours de clôture ajusté en fin de mois.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

from univers import ACTION, ETF, PORTEFEUILLE

DEBUT, FIN = "1990-01-01", "2026-09-01"


def main():
    tickers = ACTION + ETF + [t for liste in PORTEFEUILLE.values() for t in liste]
    brut = yf.download(tickers, start=DEBUT, end=FIN, interval="1mo",
                       auto_adjust=True, progress=True, group_by="column")
    cours = brut["Close"].copy()
    cours.index = pd.to_datetime(cours.index).to_period("M")
    cours.index.name = "mois"

    manquants = [t for t in tickers if t not in cours.columns or cours[t].dropna().empty]
    if manquants:
        print("Tickers sans données :", manquants)

    sortie = Path(__file__).resolve().parents[1] / "data" / "raw" / "cours_mensuels.csv"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    cours.to_csv(sortie)
    print(f"{cours.shape[1]} séries, {cours.index.min()} -> {cours.index.max()}, écrit dans {sortie}")


if __name__ == "__main__":
    main()
