"""Étape 2 : figer une copie datée des fichiers bruts.

Kenneth French révise ses séries (changement de base CRSP, révisions des
valeurs comptables). La seule garantie de reproductibilité est de garder les
zips tels qu'ils ont été téléchargés, datés, avec leur empreinte SHA-256.

Usage :
    python -m src.data.freeze_raw <dossier_des_zips> --date AAAA-MM-JJ
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

EXPECTED = {
    # cœur de la base (États-Unis)
    "F-F_Research_Data_5_Factors_2x3_CSV.zip": "US, 5 facteurs, mensuel",
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip": "US, 5 facteurs, quotidien",
    "F-F_Momentum_Factor_CSV.zip": "US, momentum, mensuel",
    "F-F_Momentum_Factor_daily_CSV.zip": "US, momentum, quotidien",
    # réplication
    "Developed_5_Factors_CSV.zip": "Developed, 5 facteurs, mensuel",
    "Developed_Mom_Factor_CSV.zip": "Developed, momentum, mensuel",
    "Europe_5_Factors_CSV.zip": "Europe, 5 facteurs, mensuel",
    "Europe_Mom_Factor_CSV.zip": "Europe, momentum, mensuel",
    "Japan_5_Factors_CSV.zip": "Japon, 5 facteurs, mensuel",
    "Japan_Mom_Factor_CSV.zip": "Japon, momentum, mensuel",
    "6_Portfolios_2x3_CSV.zip": "US, 6 portefeuilles taille x B/M (jambe longue)",
}

RAW_DIR = Path("data/raw")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(src: Path, date: str) -> Path:
    dest = RAW_DIR / date
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"download_date": date, "source": BASE_URL, "files": {}}
    for name, label in EXPECTED.items():
        path = src / name
        if not path.exists():
            print(f"absent : {name}")
            continue
        shutil.copy2(path, dest / name)
        manifest["files"][name] = {"description": label, "sha256": sha256(dest / name)}

    with open(dest / "SHA256SUMS", "w") as f:
        for name, meta in sorted(manifest["files"].items()):
            f.write(f"{meta['sha256']}  {name}\n")
    with open(dest / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"{len(manifest['files'])}/{len(EXPECTED)} fichiers figés dans {dest}")
    return dest


def latest_raw_dir() -> Path:
    dirs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    if not dirs:
        raise FileNotFoundError("aucune copie figée dans data/raw/")
    return dirs[-1]


def verify(folder: Path) -> bool:
    """Vérifie que les zips n'ont pas été modifiés depuis le gel."""
    ok = True
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        if sha256(folder / name) != digest:
            print(f"empreinte modifiée : {name}")
            ok = False
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    freeze(args.src, args.date)
