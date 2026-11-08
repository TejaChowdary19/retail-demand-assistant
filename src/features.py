"""Leakage-safe features. Every feature for date t uses only sales strictly before t."""
import numpy as np
import pandas as pd
from src.config import *


def load_prepared():
    df = pd.read_parquet(DEMO / "prepared_sales.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


def ensure_calendar():
    p = DEMO / "calendar.parquet"
    if not p.exists():
        cal = pd.read_csv(ROOT / "data" / "raw" / "calendar.csv", parse_dates=["date"])
        cols = ["date", "wday", "month", "year", "event_name_1", "event_type_1", "snap_CA"]
        cal[cols].to_parquet(p, index=False)


def load_calendar():
    ensure_calendar()
    cal = pd.read_parquet(DEMO / "calendar.parquet")
    cal["date"] = pd.to_datetime(cal["date"])
    return cal


def calendar_features(cal):
    # Event and SNAP flags are published in advance, so they are legitimately known for future dates.
    return pd.DataFrame({
        "date": cal["date"], "wday": cal["wday"], "month": cal["month"],
        "is_weekend": cal["wday"].isin([1, 2]).astype(int),  # M5: wday 1=Sat, 2=Sun
        "has_event": cal["event_name_1"].notna().astype(int),
        "snap_CA": cal["snap_CA"],
    }).set_index("date", drop=False)


def product_table(df):
    meta = (df[["product_id", "department_id", "category_id"]].drop_duplicates()
            .sort_values("product_id").reset_index(drop=True))
    meta["product_code"] = range(len(meta))
    meta["dept_code"] = meta["department_id"].astype("category").cat.codes
    meta["cat_code"] = meta["category_id"].astype("category").cat.codes
    return meta


def make_training_frame(df, calf, meta):
    d = (df[["product_id", "date", "units_sold"]].merge(meta, on="product_id")
         .merge(calf.reset_index(drop=True), on="date")
         .sort_values(["product_id", "date"]).reset_index(drop=True))
    g = d.groupby("product_id")["units_sold"]
    for k in LAGS:
        d[f"lag_{k}"] = g.shift(k)
    for w in ROLL_MEANS:  # shift(1) first: window covers days t-w .. t-1 only
        d[f"roll_mean_{w}"] = g.transform(lambda s: s.shift(1).rolling(w).mean())
    for w in ROLL_STDS:
        d[f"roll_std_{w}"] = g.transform(lambda s: s.shift(1).rolling(w).std())
    return d.dropna(subset=["lag_28", "roll_mean_28", "roll_std_28"]).reset_index(drop=True)


def features_from_history(hist, cal_row, meta):
    """Features for ONE future day for all products. hist: (products x days) sales BEFORE that day."""
    cols = {}
    for k in LAGS:
        cols[f"lag_{k}"] = hist[:, -k]
    for w in ROLL_MEANS:
        cols[f"roll_mean_{w}"] = hist[:, -w:].mean(axis=1)
    for w in ROLL_STDS:
        cols[f"roll_std_{w}"] = hist[:, -w:].std(axis=1, ddof=1)
    X = pd.DataFrame(cols)
    for c in STATIC:
        X[c] = meta[c].to_numpy()
    for c in CALENDAR:
        X[c] = int(cal_row[c])
    return X[FEATURES]
