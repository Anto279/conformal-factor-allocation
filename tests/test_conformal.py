"""La CP normalisée tient sa couverture quand les erreurs sont échangeables."""

import numpy as np
import pandas as pd

from src.conformal import conformal_intervals, conformal_quantile, decision_scores


def _panel(n_months=600, factors=("A", "B", "C"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1950-01-31", periods=n_months, freq="ME")
    idx = pd.MultiIndex.from_product([dates, factors], names=["date", "factor"])
    sigma = pd.Series(rng.uniform(0.01, 0.05, len(idx)), idx)
    z = pd.Series(rng.standard_t(5, len(idx)), idx)  # queues épaisses
    zhat = pd.Series(0.0, idx)
    return z, zhat, sigma


def test_quantile_is_finite_sample_corrected():
    scores = np.arange(1, 101, dtype=float)
    assert conformal_quantile(scores, 0.2) == 81.0  # ceil(101 * 0.8) = 81e valeur


def test_coverage_close_to_nominal():
    z, zhat, sigma = _panel()
    for alpha in (0.2, 0.5):
        iv = conformal_intervals(z, zhat, sigma, alpha)
        cov = iv["covered"].mean()
        assert abs(cov - (1 - alpha)) < 0.03


def test_aci_also_covers_and_no_future_scores_used():
    z, zhat, sigma = _panel(seed=1)
    iv = conformal_intervals(z, zhat, sigma, 0.2, gamma=0.005)
    assert abs(iv["covered"].mean() - 0.8) < 0.03
    # modifier le dernier mois ne change aucune fourchette passée
    z2 = z.copy()
    last = z.index.get_level_values("date").max()
    z2.loc[last] = 100.0
    iv2 = conformal_intervals(z2, zhat, sigma, 0.2, gamma=0.005)
    pd.testing.assert_series_equal(iv["upper"], iv2["upper"])


def test_decision_rules():
    iv = pd.DataFrame({"muhat": [0.02, -0.03, 0.01], "lower": [0.01, -0.05, -0.02],
                       "upper": [0.03, -0.01, 0.04]})
    d = decision_scores(iv)
    assert list(d["binary"]) == [1.0, -1.0, 0.0]
    assert np.allclose(d["continuous"], [1.0, -1.0, 1 / 3])
