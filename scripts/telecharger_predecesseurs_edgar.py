"""Complète edgar_comptes.csv.gz avec les comptes des entités prédécesseures.

La table ticker -> CIK de la SEC donne l'entité qui porte le ticker aujourd'hui. Quand
une entreprise a changé de structure juridique (nouvelle holding, fusion, redomiciliation),
ses comptes antérieurs sont publiés sous un autre CIK. Exemple : ExxonMobil publie depuis 2026
sous une nouvelle holding, ses comptes de 2009 à 2025 sont sous le CIK 34088.

    python scripts/telecharger_predecesseurs_edgar.py --nom "Prénom Nom" --email "adresse@exemple.com"

Produit data/raw/edgar_comptes_predecesseurs.csv.gz (quelques secondes).
"""
import argparse

import pandas as pd

from telecharger_donnees_sp500 import RAW, Sec, extraire_faits

# ticker actuel -> CIK de l'entité qui publiait avant la réorganisation
PREDECESSEURS = {
    "XOM": 34088,      # Exxon Mobil Corp, avant la holding de 2026
    "DIS": 1001039,    # Walt Disney Co (TWDC Enterprises 18), avant le rachat de Fox en 2019
    "BLK": 1364742,    # BlackRock Inc, avant la nouvelle holding de 2024
    "CI": 701221,      # Cigna Corp, avant le rachat d'Express Scripts en 2018
    "APA": 6769,       # Apache Corp, avant la holding APA de 2021
    "XRX": 108772,     # Xerox Corp, avant Xerox Holdings en 2019
    "DOW": 29915,      # Dow Chemical, avant la fusion DowDuPont de 2017
    "DD": 30554,       # E. I. du Pont de Nemours, avant DowDuPont
    "GOOGL": 1288776,  # Google Inc, avant Alphabet en 2015
    "MDT": 64670,      # Medtronic Inc, avant Medtronic plc en 2015
    "ETN": 31277,      # Eaton Corp, avant Eaton plc en 2012
    "ICE": 1174746,    # IntercontinentalExchange Inc, avant le rachat de NYSE en 2013
    "AVGO": 1441634,   # Avago Technologies / Broadcom Ltd, avant Broadcom Inc en 2018
    "FOXA": 1308161,   # Twenty-First Century Fox, avant Fox Corp en 2019
    "PRGO": 820096,    # Perrigo Co, avant Perrigo plc en 2013
    "DELL": 826083,    # Dell Inc, avant son retrait de la cote en 2013
    "FTI": 1135152,    # FMC Technologies, avant TechnipFMC en 2017
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nom", required=True)
    parser.add_argument("--email", required=True)
    args = parser.parse_args()

    sec = Sec(args.nom, args.email)
    faits = []
    for ticker, cik in PREDECESSEURS.items():
        donnees = sec.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
        lignes = extraire_faits(donnees, ticker, cik) if donnees else []
        print(f"{ticker} (CIK {cik}) : {len(lignes)} chiffres")
        faits += lignes
    faits = pd.DataFrame(faits)
    faits = (faits.sort_values("depose")
             .drop_duplicates(["ticker", "concept", "unite", "debut", "fin"], keep="first")
             .sort_values(["ticker", "concept", "fin"]))
    faits.to_csv(RAW / "edgar_comptes_predecesseurs.csv.gz", index=False)
    print(f"{len(faits)} chiffres écrits dans edgar_comptes_predecesseurs.csv.gz")


if __name__ == "__main__":
    main()
