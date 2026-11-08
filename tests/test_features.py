import numpy as np
import pandas as pd
import pytest
from src.config import DEMO
from src.features import *
from src.forecast import recursive_forecast, seasonal_naive

pytestmark = pytest.mark.skipif(not (DEMO / "prepared_sales.parquet").exists(), reason="run data preparation first")


def _frame(df):
    return make_training_frame(df, calendar_features(load_calendar()), product_table(df))


def test_lags_and_rolling_exclude_current_day():
    df = load_prepared()
    fr = _frame(df)
    row = fr.iloc[500]
    s = df[df.product_id == row.product_id].set_index("date")["units_sold"]
    t = row.date
    assert row["lag_1"] == s[t - pd.Timedelta(days=1)]
    assert row["lag_7"] == s[t - pd.Timedelta(days=7)]
    prev7 = s[t - pd.Timedelta(days=7): t - pd.Timedelta(days=1)]
    assert np.isclose(row["roll_mean_7"], prev7.mean())


def test_changing_future_sales_does_not_change_past_features():
    df = load_prepared()
    cutoff = df.date.max() - pd.Timedelta(days=30)
    df2 = df.copy()
    df2.loc[df2.date > cutoff, "units_sold"] += 1000
    a, b = _frame(df), _frame(df2)
    cols = [c for c in FEATURES_ALL if c in a.columns]
    pa = a[a.date <= cutoff + pd.Timedelta(days=1)].reset_index(drop=True)
    pb = b[b.date <= cutoff + pd.Timedelta(days=1)].reset_index(drop=True)
    pd.testing.assert_frame_equal(pa[cols], pb[cols])


FEATURES_ALL = ["lag_1", "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_14",
                "roll_mean_28", "roll_std_7", "roll_std_28"]


class EchoLag1:
    """Dummy model: predicts yesterday's value, so recursion is visible in the output."""
    def predict(self, X):
        return X["lag_1"].to_numpy()


def test_recursive_forecast_feeds_on_predictions_and_shape():
    df = load_prepared()
    meta = product_table(df).iloc[:3]
    calf = calendar_features(load_calendar())
    rng = np.random.default_rng(0)
    hist = rng.integers(0, 10, size=(3, 60)).astype(float)
    dates, p = recursive_forecast(EchoLag1(), hist, meta, calf, "2016-05-23")
    assert p.shape == (3, 28) and len(dates) == 28
    assert dates[0] == pd.Timestamp("2016-05-23") and dates[-1] == pd.Timestamp("2016-06-19")
    assert np.allclose(p, hist[:, -1][:, None])  # every day echoes the last value: only predictions fed back


def test_negative_predictions_clipped():
    class Neg:
        def predict(self, X):
            return -np.ones(len(X))
    df = load_prepared()
    meta = product_table(df).iloc[:2]
    _, p = recursive_forecast(Neg(), np.ones((2, 60)), meta, calendar_features(load_calendar()), "2016-05-23")
    assert (p == 0).all()


def test_seasonal_naive():
    hist = np.arange(14, dtype=float)[None, :]
    out = seasonal_naive(hist)
    assert out.shape == (1, 28)
    assert np.array_equal(out[0, :7], hist[0, -7:]) and np.array_equal(out[0, 7:14], hist[0, -7:])
