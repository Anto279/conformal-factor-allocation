"""Étape 3 : construire la base propre à partir de la copie figée.

Sorties (data/processed/) :
- factors_monthly.parquet : MKT, SMB, HML, RMW, CMA, MOM, RF, mensuel, décimaux
- factors_daily.parquet   : mêmes colonnes, quotidien
- factors_<region>_monthly.parquet pour Developed, Europe, Japan
- build_info.json : période couverte, nombre d'observations, empreintes sources

Usage :
    python -m src.data.build_factors
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.french import has_missing_codes, read_french_zip
from src.data.freeze_raw import latest_raw_dir

OUT_DIR = Path("data/processed")
COLUMNS = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]
STYLES = ["SMB", "HML", "RMW", "CMA", "MOM"]
RENAME = {"Mkt-RF": "MKT", "Mom": "MOM", "WML": "MOM"}

US_FILES = {
    "M": ("F-F_Research_Data_5_Factors_2x3_CSV.zip", "F-F_Momentum_Factor_CSV.zip"),
    "D": ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "F-F_Momentum_Factor_daily_CSV.zip"),
}
REGIONS = {
    "developed": ("Developed_5_Factors_CSV.zip", "Developed_Mom_Factor_CSV.zip"),
    "europe": ("Europe_5_Factors_CSV.zip", "Europe_Mom_Factor_CSV.zip"),
    "japan": ("Japan_5_Factors_CSV.zip", "Japan_Mom_Factor_CSV.zip"),
}


def combine(five_path: Path, mom_path: Path, freq: str) -> pd.DataFrame:
    """Fusionne les 5 facteurs et le momentum, sur les dates communes."""
    five = read_french_zip(five_path, freq).rename(columns=RENAME)
    mom = read_french_zip(mom_path, freq).rename(columns=RENAME)
    for path, df in ((five_path, five), (mom_path, mom)):
        if has_missing_codes(df):
            raise ValueError(f"{path.name} : valeurs manquantes")
    df = five.join(mom[["MOM"]], how="inner")
    df = df[COLUMNS] / 100.0  # pourcentages -> décimaux
    return df.sort_index()


def build(raw_dir: Path | None = None) -> dict:
    raw_dir = raw_dir or latest_raw_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info: dict = {"raw_dir": str(raw_dir), "tables": {}}

    outputs = {}
    for freq, (five, mom) in US_FILES.items():
        name = "factors_monthly" if freq == "M" else "factors_daily"
        outputs[name] = combine(raw_dir / five, raw_dir / mom, freq)
    for region, (five, mom) in REGIONS.items():
        if (raw_dir / five).exists() and (raw_dir / mom).exists():
            outputs[f"factors_{region}_monthly"] = combine(raw_dir / five, raw_dir / mom, "M")

    for name, df in outputs.items():
        df.to_parquet(OUT_DIR / f"{name}.parquet")
        info["tables"][name] = {
            "start": str(df.index[0].date()),
            "end": str(df.index[-1].date()),
            "rows": int(len(df)),
            "columns": list(df.columns),
        }
        print(f"{name:28s} {df.index[0].date()} -> {df.index[-1].date()}  {len(df):6d} lignes")

    manifest = json.loads((raw_dir / "manifest.json").read_text())
    info["download_date"] = manifest["download_date"]
    with open(OUT_DIR / "build_info.json", "w") as f:
        json.dump(info, f, indent=2)
    return info


def load_monthly(region: str = "us") -> pd.DataFrame:
    name = "factors_monthly" if region == "us" else f"factors_{region}_monthly"
    return pd.read_parquet(OUT_DIR / f"{name}.parquet")


def load_daily() -> pd.DataFrame:
    return pd.read_parquet(OUT_DIR / "factors_daily.parquet")


if __name__ == "__main__":
    build()
