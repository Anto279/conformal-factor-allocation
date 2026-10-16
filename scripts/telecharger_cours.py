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

# Action étudiée seule
ACTION = ["AAPL"]

# Références de marché : S&P 500 pondéré par la capitalisation et équipondéré
ETF = ["SPY", "RSP"]

# Portefeuille diversifié : grandes capitalisations américaines cotées sans interruption
# depuis 2000, réparties sur les 11 secteurs GICS
PORTEFEUILLE = {
    "Technologie": ["MSFT", "ORCL", "CSCO", "INTC", "IBM", "TXN", "QCOM", "ADBE", "AMAT",
                    "ADI", "MU", "HPQ", "KLAC", "LRCX"],
    "Santé": ["JNJ", "PFE", "MRK", "ABT", "AMGN", "BMY", "LLY", "MDT", "UNH", "CVS",
              "BDX", "SYK", "CI", "TMO", "BAX"],
    "Finance": ["JPM", "BAC", "WFC", "C", "AXP", "USB", "PNC", "BK", "SCHW", "AIG",
                "MMC", "TRV", "ALL", "CB", "BEN"],
    "Consommation de base": ["KO", "PEP", "PG", "WMT", "COST", "CL", "KMB", "GIS", "SYY",
                             "HSY", "CLX", "MO"],
    "Consommation discrétionnaire": ["HD", "LOW", "MCD", "NKE", "TGT", "SBUX", "TJX", "F",
                                     "GPC", "ROST"],
    "Industrie": ["GE", "HON", "MMM", "CAT", "DE", "UNP", "LMT", "GD", "EMR", "ITW", "FDX",
                  "UPS", "NOC", "CSX", "NSC", "ETN", "PH"],
    "Énergie": ["XOM", "CVX", "COP", "SLB", "OXY", "EOG", "HAL", "DVN"],
    "Services aux collectivités": ["DUK", "SO", "NEE", "D", "AEP", "ED", "EXC"],
    "Matériaux": ["SHW", "APD", "ECL", "NUE", "NEM", "PPG"],
    "Communication": ["T", "VZ", "DIS", "CMCSA"],
    "Immobilier": ["SPG", "PSA"],
}

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
