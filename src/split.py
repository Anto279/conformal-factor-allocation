"""Découpage des données, fixé avant toute analyse.

07/1963 - 12/1989 : exploration et calibration (tout est permis)
01/1990 - fin     : test sous scellé, un seul passage après le tag protocole-v1
Developed, Europe, Japon : réplication, sans aucun réglage supplémentaire
"""

from __future__ import annotations

import pandas as pd

EXPLORATION_END = pd.Timestamp("1989-12-31")
TEST_START = pd.Timestamp("1990-01-01")


def exploration(df: pd.DataFrame) -> pd.DataFrame:
    """Coupe les données au 31/12/1989, avant tout calcul.

    Couper avant de calculer (et non après) garantit qu'aucune estimation
    faite en exploration n'a vu un seul jour de la période de test.
    """
    level = df.index.get_level_values("date") if isinstance(df.index, pd.MultiIndex) else df.index
    return df[level <= EXPLORATION_END].copy()
