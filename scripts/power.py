"""Calcul de puissance : quel écart de performance la période de test peut-elle détecter ?

Pour un écart de rendement mensuel d_t entre deux stratégies, avec un ratio
d'information annualisé IR = E[d] / sd(d) * sqrt(12), le t-stat attendu sur
T années vaut environ IR * sqrt(T). Il faut donc

    IR_min = (z_{1-a/2} + z_{puissance}) / sqrt(T)

pour détecter l'écart avec un risque de première espèce a et la puissance voulue.

Usage : python -m scripts.power
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import norm

from src.data.build_factors import load_monthly
from src.split import TEST_START

RESULTS = Path("results")


def ir_min(years: float, alpha: float = 0.05, power: float = 0.5) -> float:
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) / np.sqrt(years)


def main() -> None:
    monthly = load_monthly()
    test_months = int((monthly.index >= TEST_START).sum())
    years = test_months / 12
    lines = [f"Période de test : {test_months} mois, soit {years:.1f} ans.\n",
             "| puissance visée | IR minimal (a = 5 %, bilatéral) |", "|---|---|"]
    for p in (0.5, 0.8):
        lines.append(f"| {p:.0%} | {ir_min(years, power=p):.2f} |")
    lines.append("")
    lines.append("Avec les régions en réplication (Developed, Europe, Japon, à partir de 1990-1991), "
                 "si les écarts étaient indépendants entre régions, la durée effective serait "
                 f"proche de {4 * years:.0f} ans et l'IR minimal à 50 % de puissance de "
                 f"{ir_min(4 * years):.2f}. En pratique les régions sont corrélées : c'est un plancher optimiste.")
    text = "\n".join(lines)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "puissance.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
