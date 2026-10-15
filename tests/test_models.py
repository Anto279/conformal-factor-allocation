"""Le walk-forward n'utilise jamais une cible non encore observée."""

import numpy as np
import pandas as pd

from src.models import walk_forward


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1960-01-31", periods=200, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "factor"])
    df = pd.DataFrame({"mom": rng.normal(size=len(idx)), "vol": rng.normal(size=len(idx)),
                       "sigma": 0.03}, index=idx)
    df["z"] = 0.3 * df["mom"] + rng.normal(size=len(idx))
    return df


def test_future_targets_do_not_change_past_forecasts():
    p = _panel()
    start = pd.Timestamp("1966-01-31")
    cut = pd.Timestamp("1970-06-30")
    base = walk_forward(p, "ridge", ["mom", "vol"], start)
    p2 = p.copy()
    future = p2.index.get_level_values("date") >= cut
    p2.loc[future, "z"] = 50.0
    alt = walk_forward(p2, "ridge", ["mom", "vol"], start)
    past = base.index.get_level_values("date") <= cut  # la cible de cut n'est connue qu'après cut
    pd.testing.assert_series_equal(base[past], alt[past])
    assert not np.allclose(base[~past], alt[~past])


def test_ridge_recovers_signal_sign():
    p = _panel(1)
    f = walk_forward(p, "ridge", ["mom", "vol"], pd.Timestamp("1975-01-31"))
    mom = p["mom"].reindex(f.index)
    assert np.corrcoef(f, mom)[0, 1] > 0.9
