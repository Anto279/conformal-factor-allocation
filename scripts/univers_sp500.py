"""Composition historique du S&P 500, mois par mois.

Source : github.com/fja05680/sp500 (licence MIT), fichier
« S&P 500 Historical Components & Changes (Updated).csv », copié dans
data/raw/sp500_composition_historique.csv. Chaque ligne donne la liste des
membres de l'indice à une date de changement.
"""
from pathlib import Path

import pandas as pd

FICHIER = Path(__file__).resolve().parents[1] / "data" / "raw" / "sp500_composition_historique.csv"
DEBUT_ETUDE = "2009-12"


def ticker_yahoo(ticker):
    # BRK.B -> BRK-B : convention de Yahoo Finance et de la SEC
    return ticker.replace(".", "-")


def membres_mensuels(debut=DEBUT_ETUDE, fichier=FICHIER):
    """Tableau booléen mois x ticker : True si le titre est dans l'indice en fin de mois."""
    compo = pd.read_csv(fichier, parse_dates=["date"]).set_index("date")["tickers"]
    fin_de_mois = compo.resample("ME").last().ffill()
    fin_de_mois.index = fin_de_mois.index.to_period("M")
    fin_de_mois = fin_de_mois.loc[debut:]
    listes = fin_de_mois.str.split(",").apply(lambda l: [ticker_yahoo(t.strip()) for t in l])
    tickers = sorted({t for l in listes for t in l})
    membres = pd.DataFrame(False, index=fin_de_mois.index, columns=tickers)
    for mois, l in listes.items():
        membres.loc[mois, l] = True
    return membres


def tickers_etude(debut=DEBUT_ETUDE):
    return list(membres_mensuels(debut).columns)
