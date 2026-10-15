import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"


@pytest.fixture(scope="session")
def real_data():
    import os
    if not (PROCESSED / "factors_monthly.parquet").exists():
        pytest.skip("base non construite : lancer python -m src.data.build_factors")
    import pandas as pd
    os.chdir(ROOT)
    return (pd.read_parquet(PROCESSED / "factors_monthly.parquet"),
            pd.read_parquet(PROCESSED / "factors_daily.parquet"))
