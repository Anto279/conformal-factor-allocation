"""Univers de titres de l'étape 4 : l'action étudiée seule, les ETF de référence et le portefeuille diversifié."""

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
