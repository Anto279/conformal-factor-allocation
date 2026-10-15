"""Style commun des figures (docs/figures/)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path("docs/figures")

FACTOR_COLORS = {
    "MKT": "#2a78d6", "SMB": "#eb6834", "HML": "#1baf7a",
    "RMW": "#eda100", "CMA": "#e87ba4", "MOM": "#6250d6",
}
COLORS = {"main": "#2a78d6", "alt": "#eb6834", "band": "#a9c8ee", "grey": "#8a8984"}
FACTOR_LABELS = {
    "MKT": "Marché (Mkt-RF)", "SMB": "Taille (SMB)", "HML": "Value (HML)",
    "RMW": "Rentabilité (RMW)", "CMA": "Investissement (CMA)", "MOM": "Momentum (Mom)",
}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b2a", "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False,
        "lines.linewidth": 1.6, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
