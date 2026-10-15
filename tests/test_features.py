"""Étape 5 : aucune variable n'utilise d'information postérieure à sa date."""

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES, build_features


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("1963-07-01", "1972-12-31")
    vol = 0.006 * np.exp(0.3 * np.sin(np.arange(len(days)) / 200))
    daily = pd.DataFrame({"A": rng.normal(0.0002, vol), "B": rng.normal(0.0001, vol * 1.5)}, index=days)
    monthly = (1 + daily).resample("ME").prod() - 1
    monthly.index = monthly.index.normalize()
    return monthly, daily


def _assert_truncation_invariance(monthly, daily, factors, cut, with_garch=True):
    full = build_features(monthly, daily, factors, with_garch=with_garch)
    trunc = build_features(monthly[monthly.index <= cut], daily[daily.index <= cut], factors,
                           with_garch=with_garch)
    cols = ["mom", "vol", "rv"] + (["sigma"] if with_garch else [])
    a = full.loc[full.index.get_level_values("date") <= cut, cols]
    pd.testing.assert_frame_equal(a, trunc[cols], check_exact=False, rtol=1e-10)


def test_features_do_not_peek_synthetic():
    monthly, daily = _synthetic()
    _assert_truncation_invariance(monthly, daily, ["A", "B"], pd.Timestamp("1969-06-30"))


def test_target_is_next_month():
    monthly, daily = _synthetic()
    p = build_features(monthly, daily, ["A"], with_garch=False).xs("A", level="factor")
    assert np.allclose(p["target"].iloc[:-1].to_numpy(), monthly["A"].iloc[1:].to_numpy())
    assert np.isnan(p["target"].iloc[-1])


def test_features_do_not_peek_real(real_data):
    monthly, daily = real_data
    _assert_truncation_invariance(monthly[monthly.index <= "1975-12-31"],
                                  daily[daily.index <= "1975-12-31"],
                                  ["HML", "MOM"], pd.Timestamp("1972-03-31"))


def _r2(panel, target):
    d = panel.dropna(subset=FEATURES + [target])
    X = np.column_stack([np.ones(len(d)), d[FEATURES].to_numpy()])
    y = d[target].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)


def test_leak_detector(real_data):
    """Avec la vraie cible (t+1), le R² est proche de zéro. Si on « oublie » le
    décalage et qu'on prend le rendement du mois t, le R² grimpe nettement :
    c'est la signature d'une fuite, que le test doit être capable de voir."""
    monthly, daily = real_data
    m, d = monthly[monthly.index <= "1989-12-31"], daily[daily.index <= "1989-12-31"]
    p = build_features(m, d, ["SMB", "HML", "RMW", "CMA", "MOM"], with_garch=False)
    honest = _r2(p, "target")
    leaked = _r2(p, "ret")
    assert honest < 0.03
    assert leaked > 5 * max(honest, 0.005)
