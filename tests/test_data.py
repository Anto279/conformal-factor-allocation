"""Étape 4 : tests de cohérence de la base construite.

Ces tests tournent sur les vraies données (data/processed/) ; ils sont
ignorés tant que la base n'a pas été construite.
"""

import numpy as np
import pandas as pd

from src.data.french import MISSING_CODES
from src.data.freeze_raw import latest_raw_dir, verify

FACTORS = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]


def test_raw_files_unchanged(real_data):
    assert verify(latest_raw_dir())


def test_no_missing_values(real_data):
    for df in real_data:
        assert not df.isna().any().any()
        assert not np.isin(np.round(df.to_numpy() * 100, 2), MISSING_CODES).any()


def test_monthly_index_is_continuous(real_data):
    monthly, _ = real_data
    idx = monthly.index
    assert idx.is_unique and idx.is_monotonic_increasing
    expected = pd.date_range(idx[0], idx[-1], freq="ME")
    assert len(idx) == len(expected) and (idx == expected).all()
    assert idx[0] == pd.Timestamp("1963-07-31")


def test_daily_index_is_clean(real_data):
    _, daily = real_data
    assert daily.index.is_unique and daily.index.is_monotonic_increasing
    assert (daily.index.dayofweek < 5).all()  # jours ouvrés uniquement
    per_month = daily.groupby(daily.index.to_period("M")).size()
    assert per_month.between(15, 23).all()


def test_returns_are_decimals(real_data):
    monthly, daily = real_data
    assert monthly[FACTORS].abs().max().max() < 0.5
    assert daily[FACTORS].abs().max().max() < 0.25
    assert monthly["RF"].between(-0.001, 0.02).all()


def test_daily_compounds_to_monthly(real_data):
    """Le cumul des jours d'un mois doit retrouver le rendement mensuel publié.

    L'égalité n'est pas exacte : un facteur long-short mensuel est la différence
    de deux rendements mensuels, pas le produit des différences quotidiennes.
    Le momentum est un cas à part : la version quotidienne de French reconstitue
    ses portefeuilles chaque jour, la version mensuelle chaque mois. Les deux
    séries sont proches (corrélation ~0,97) mais ce ne sont pas les mêmes.
    """
    monthly, daily = real_data
    comp = (1 + daily[FACTORS]).resample("ME").prod() - 1
    comp.index = comp.index.normalize()
    common = monthly.index.intersection(comp.index)
    for f in FACTORS:
        a, b = monthly.loc[common, f], comp.loc[common, f]
        corr_min, gap_max = (0.95, 0.006) if f == "MOM" else (0.99, 0.001)
        assert np.corrcoef(a, b)[0, 1] > corr_min, f
        assert (a - b).abs().median() < gap_max, f


def test_orders_of_magnitude(real_data):
    monthly, _ = real_data
    m = monthly[FACTORS]
    assert 0.004 < m["MKT"].mean() < 0.008        # prime de marché ~0,5-0,7 %/mois
    assert 0.02 < m["HML"].std() < 0.035           # vol de HML ~3 %/mois
    assert 0.035 < m["MKT"].std() < 0.055
    assert m["MOM"].min() < -0.25                  # krach du momentum de 2009
    assert monthly.loc["1987-10-31", "MKT"] < -0.2  # krach d'octobre 1987
