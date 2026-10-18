"""Télécharge les données de l'étape 8 pour tous les titres passés par le S&P 500 depuis fin 2009.

1. Cours mensuels (Yahoo Finance) : clôture ajustée des divisions d'actions,
   clôture ajustée des divisions et des dividendes, et historique des divisions.
2. Comptes annuels et trimestriels (SEC EDGAR, API « companyfacts ») : fonds propres,
   actif total, résultat d'exploitation, charges d'intérêts, nombre d'actions,
   avec la date de dépôt de chaque chiffre. Plus le code SIC (secteur) de chaque société.

La SEC demande de s'identifier par un nom et une adresse e-mail :

    pip install yfinance pandas requests
    python scripts/telecharger_donnees_sp500.py --nom "Prénom Nom" --email "adresse@exemple.com"

Durée : 20 à 40 minutes. En cas d'interruption, relancer la même commande :
les sociétés déjà traitées sont conservées dans data/cache/ et ne sont pas retéléchargées.

Fichiers produits dans data/raw/ :
    sp500_cours_mensuels.csv        clôture ajustée des divisions (pour la capitalisation)
    sp500_cours_ajustes_mensuels.csv clôture ajustée des divisions et dividendes (pour les rendements)
    sp500_divisions.csv             divisions d'actions (date, ratio)
    edgar_societes.csv              ticker, CIK, nom, code SIC
    edgar_comptes.csv.gz            un chiffre comptable par ligne, première publication
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from univers_sp500 import tickers_etude

RACINE = Path(__file__).resolve().parents[1]
RAW = RACINE / "data" / "raw"
CACHE = RACINE / "data" / "cache" / "edgar"
DEBUT_COURS, FIN_COURS = "2004-01-01", "2026-09-01"

CONCEPTS = {
    "us-gaap": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "PreferredStockValue",
        "Assets",
        "OperatingIncomeLoss",
        "InterestExpense",
        "InterestExpenseNonoperating",
        "InterestExpenseDebt",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
    "dei": ["EntityCommonStockSharesOutstanding"],
}
FORMULAIRES_ANNUELS = {"10-K", "10-K/A", "10-KT", "20-F", "40-F"}
FORMULAIRES_TRIMESTRIELS = {"10-Q", "10-Q/A"}
# le nombre d'actions est utile chaque trimestre ; les autres chiffres seulement une fois par an
CONCEPTS_TRIMESTRIELS = {"EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding",
                         "WeightedAverageNumberOfSharesOutstandingBasic"}


# ---------------------------------------------------------------- cours
def telecharger_cours(tickers):
    cours, ajustes = [], []
    for i in range(0, len(tickers), 100):
        lot = tickers[i:i + 100]
        print(f"Cours : titres {i + 1} à {i + len(lot)} sur {len(tickers)}")
        brut = yf.download(lot, start=DEBUT_COURS, end=FIN_COURS, interval="1mo",
                           auto_adjust=False, progress=False, group_by="column", threads=True)
        cours.append(brut["Close"])
        ajustes.append(brut["Adj Close"])
    for tables, nom in [(cours, "sp500_cours_mensuels.csv"), (ajustes, "sp500_cours_ajustes_mensuels.csv")]:
        t = pd.concat(tables, axis=1)
        t = t.loc[:, ~t.columns.duplicated()]
        t.index = pd.to_datetime(t.index).to_period("M")
        t.index.name = "mois"
        t.to_csv(RAW / nom)
    vides = [c for c in t.columns if t[c].dropna().empty]
    print(f"Cours : {t.shape[1] - len(vides)} titres avec des données, {len(vides)} sans données")


def telecharger_divisions(tickers):
    lignes = []
    for i, t in enumerate(tickers, 1):
        try:
            s = yf.Ticker(t).splits
        except Exception:
            continue
        for date, ratio in s.items():
            lignes.append({"ticker": t, "date": pd.Timestamp(date).date(), "ratio": float(ratio)})
        if i % 100 == 0:
            print(f"Divisions : {i}/{len(tickers)}")
    pd.DataFrame(lignes, columns=["ticker", "date", "ratio"]).to_csv(RAW / "sp500_divisions.csv", index=False)


# ---------------------------------------------------------------- EDGAR
class Sec:
    def __init__(self, nom, email):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"{nom} {email}", "Accept-Encoding": "gzip, deflate"})
        self.dernier = 0.0

    def get(self, url):
        for essai in range(5):
            attente = 0.15 - (time.time() - self.dernier)  # moins de 10 requêtes par seconde
            if attente > 0:
                time.sleep(attente)
            self.dernier = time.time()
            r = self.session.get(url, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            time.sleep(2 ** essai)
        r.raise_for_status()


def extraire_faits(donnees, ticker, cik):
    lignes = []
    for espace, concepts in CONCEPTS.items():
        bloc = donnees.get("facts", {}).get(espace, {})
        for concept in concepts:
            for unite, faits in bloc.get(concept, {}).get("units", {}).items():
                formulaires = FORMULAIRES_ANNUELS | (FORMULAIRES_TRIMESTRIELS if concept in CONCEPTS_TRIMESTRIELS else set())
                for f in faits:
                    if f.get("form") not in formulaires:
                        continue
                    lignes.append({"ticker": ticker, "cik": cik, "concept": concept, "unite": unite,
                                   "debut": f.get("start"), "fin": f.get("end"), "valeur": f.get("val"),
                                   "exercice": f.get("fy"), "periode": f.get("fp"), "formulaire": f.get("form"),
                                   "depose": f.get("filed")})
    return lignes


def telecharger_edgar(tickers, nom, email):
    sec = Sec(nom, email)
    CACHE.mkdir(parents=True, exist_ok=True)
    table = sec.get("https://www.sec.gov/files/company_tickers.json")
    cik_de = {v["ticker"].upper(): int(v["cik_str"]) for v in table.values()}

    societes, absents = [], []
    for i, t in enumerate(tickers, 1):
        cik = cik_de.get(t.upper())
        if cik is None:
            absents.append(t)
            continue
        fichier = CACHE / f"{t}.json"
        if not fichier.exists():
            faits = sec.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
            soumissions = sec.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json") or {}
            contenu = {
                "societe": {"ticker": t, "cik": cik, "nom": soumissions.get("name"),
                            "sic": soumissions.get("sic"), "secteur_sic": soumissions.get("sicDescription")},
                "faits": extraire_faits(faits, t, cik) if faits else [],
            }
            fichier.write_text(json.dumps(contenu))
        contenu = json.loads(fichier.read_text())
        societes.append(contenu["societe"])
        if i % 50 == 0:
            print(f"EDGAR : {i}/{len(tickers)}")

    faits = [f for s in societes for f in json.loads((CACHE / f"{s['ticker']}.json").read_text())["faits"]]
    pd.DataFrame(societes).to_csv(RAW / "edgar_societes.csv", index=False)
    faits = pd.DataFrame(faits)
    # un même chiffre réapparaît dans les dépôts suivants (colonne de comparaison) :
    # on garde la première publication, la seule connue à cette date
    faits = (faits.sort_values("depose")
             .drop_duplicates(["ticker", "concept", "unite", "debut", "fin"], keep="first")
             .sort_values(["ticker", "concept", "fin"]))
    faits.to_csv(RAW / "edgar_comptes.csv.gz", index=False)
    print(f"EDGAR : {len(societes)} sociétés trouvées, {len(absents)} tickers inconnus de la SEC")
    print(f"EDGAR : {len(faits)} chiffres comptables écrits dans edgar_comptes.csv.gz")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nom", required=True, help="nom transmis à la SEC")
    parser.add_argument("--email", required=True, help="adresse e-mail transmise à la SEC")
    parser.add_argument("--sans-cours", action="store_true", help="ne pas retélécharger les cours")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    tickers = tickers_etude()
    print(f"{len(tickers)} titres passés par le S&P 500 depuis fin 2009")
    if not args.sans_cours:
        telecharger_cours(tickers)
        telecharger_divisions(tickers)
    telecharger_edgar(tickers, args.nom, args.email)


if __name__ == "__main__":
    main()
