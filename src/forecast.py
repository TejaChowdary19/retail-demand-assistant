import numpy as np
import pandas as pd
from src.config import *
from src.features import features_from_history


def sales_wide(df, meta):
    return df.pivot(index="product_id", columns="date", values="units_sold").reindex(meta["product_id"])


def seasonal_naive(hist, horizon=HORIZON):
    """Repeat the most recent 7 days across the horizon (day 1 = same weekday last week)."""
    return np.tile(hist[:, -7:], (1, int(np.ceil(horizon / 7))))[:, :horizon]


def recursive_forecast(model, hist, meta, calf, origin, horizon=HORIZON):
    """Predict day by day, appending each PREDICTION (never actuals) to a working copy of history.
    Negative forecasts are clipped to 0. Forecasts stay continuous (no rounding)."""
    origin = pd.Timestamp(origin)
    dates = pd.date_range(origin, periods=horizon)
    assert dates[-1] <= calf.index.max(), "calendar does not cover the forecast window"
    hist = np.asarray(hist, dtype=float).copy()
    preds = np.zeros((hist.shape[0], horizon))
    for i, d in enumerate(dates):
        X = features_from_history(hist, calf.loc[d], meta)
        p = np.clip(model.predict(X), 0, None)
        preds[:, i] = p
        hist = np.concatenate([hist, p[:, None]], axis=1)
    return dates, preds
