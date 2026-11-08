import numpy as np
import pandas as pd


def metrics(actual, pred):
    """bias sign convention: positive = OVERforecast (pred - actual)."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    err, denom = pred - actual, actual.sum()
    return {"MAE": np.abs(err).mean(),
            "WAPE": np.abs(err).sum() / denom if denom > 0 else np.nan,  # undefined if actual total is 0
            "bias_units_per_day": err.mean(),
            "bias_pct": err.sum() / denom if denom > 0 else np.nan}


def summarize(pred_df, by):
    rows = []
    for key, g in pred_df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        for m in ["lgbm", "baseline"]:
            rows.append({**dict(zip(by, key)), "model": m, **metrics(g["actual"], g[m])})
    return pd.DataFrame(rows)
