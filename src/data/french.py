"""Lecture des fichiers CSV zippés de la Kenneth French Data Library.

Les fichiers French ont une structure particulière :
- quelques lignes de texte libre en en-tête ;
- un ou plusieurs tableaux, chacun précédé d'une ligne de titre éventuelle
  et d'une ligne d'en-tête qui commence par une virgule (",Mkt-RF,SMB,...") ;
- des dates au format AAAAMM (mensuel), AAAA (annuel) ou AAAAMMJJ (quotidien) ;
- des rendements en pourcentage, et -99.99 / -999 pour les valeurs manquantes.

On ne dépend pas de pandas_datareader : on lit directement le CSV contenu
dans le zip, ce qui permet de travailler sur une copie figée des fichiers.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MISSING_CODES = (-99.99, -999.0)
_DATE_RE = re.compile(r"^\s*(\d{4}|\d{6}|\d{8})\s*$")


@dataclass
class FrenchTable:
    title: str
    frequency: str  # "D", "M" ou "A"
    data: pd.DataFrame


def _read_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
        if len(names) != 1:
            raise ValueError(f"{path.name} : un seul CSV attendu, trouvé {names}")
        raw = zf.read(names[0])
    return raw.decode("latin-1").replace("\r\n", "\n").replace("\r", "\n")


def _frequency(token: str) -> str:
    return {4: "A", 6: "M", 8: "D"}[len(token)]


def _to_index(tokens: list[str], freq: str) -> pd.DatetimeIndex:
    if freq == "D":
        return pd.DatetimeIndex(pd.to_datetime(tokens, format="%Y%m%d"))
    if freq == "M":
        return pd.PeriodIndex(tokens, freq="M").to_timestamp(how="end").normalize()
    return pd.PeriodIndex(tokens, freq="Y").to_timestamp(how="end").normalize()


def parse_french_csv(text: str) -> list[FrenchTable]:
    """Découpe le texte d'un fichier French en tableaux."""
    lines = text.split("\n")
    tables: list[FrenchTable] = []
    last_title = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        cells = [c.strip() for c in line.split(",")]
        is_header = len(cells) > 1 and cells[0] == "" and any(cells[1:])
        if not is_header:
            if line.strip():
                last_title = line.strip()
            i += 1
            continue

        columns = [c for c in cells[1:] if c]
        dates, values = [], []
        j = i + 1
        while j < len(lines):
            row = [c.strip() for c in lines[j].split(",")]
            if not row or not _DATE_RE.match(row[0]):
                break
            dates.append(row[0])
            values.append([float(v) for v in row[1 : len(columns) + 1]])
            j += 1

        if dates:
            freq = _frequency(dates[0])
            df = pd.DataFrame(values, index=_to_index(dates, freq), columns=columns)
            df.index.name = "date"
            tables.append(FrenchTable(title=last_title, frequency=freq, data=df))
        last_title = ""
        i = j
    return tables


def read_french_zip(path: str | Path, frequency: str | None = None) -> pd.DataFrame:
    """Renvoie le premier tableau du fichier (à la fréquence demandée).

    Les rendements restent en pourcentage ; la conversion en décimaux est faite
    dans build_factors pour que ce module reste une lecture fidèle du fichier.
    """
    tables = parse_french_csv(_read_text(Path(path)))
    if frequency is not None:
        tables = [t for t in tables if t.frequency == frequency]
    if not tables:
        raise ValueError(f"{Path(path).name} : aucun tableau trouvé")
    return tables[0].data.copy()


def read_french_tables(path: str | Path) -> list[FrenchTable]:
    """Tous les tableaux d'un fichier (utile pour les fichiers de portefeuilles)."""
    return parse_french_csv(_read_text(Path(path)))


def has_missing_codes(df: pd.DataFrame) -> bool:
    values = df.to_numpy(dtype=float)
    return bool(np.isin(np.round(values, 2), MISSING_CODES).any() or np.isnan(values).any())
