"""Backtest, select, evaluate on untouched holdout, then train the final model. Run: python -m src.train"""
import json
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.config import *
from src.features import *
from src.forecast import *
from src.evaluate import summarize


def fit(train, cfg):
    m = lgb.LGBMRegressor(**BASE_PARAMS, **cfg)
    m.fit(train[FEATURES], train["units_sold"], categorical_feature=CATEGORICAL)
    return m


def to_long(split, origin, dates, meta, actual, lgbm, base):
    return pd.DataFrame({
        "split": split, "origin": pd.Timestamp(origin),
        "product_id": np.repeat(meta["product_id"].to_numpy(), len(dates)),
        "date": np.tile(dates.to_numpy(), len(meta)),
        "actual": actual.ravel(), "lgbm": lgbm.ravel(), "baseline": base.ravel()})


def run_origin(split, origin, cfg, frame, wide, meta, calf):
    origin = pd.Timestamp(origin)
    model = fit(frame[frame["date"] < origin], cfg)           # train strictly before origin
    hist = wide.loc[:, wide.columns < origin].to_numpy(float)  # history strictly before origin
    dates, p = recursive_forecast(model, hist, meta, calf, origin)
    actual = wide[dates].to_numpy(float)                       # used ONLY for scoring
    return to_long(split, origin, dates, meta, actual, p, seasonal_naive(hist))


def main():
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    df = load_prepared(); calf = calendar_features(load_calendar()); meta = product_table(df)
    frame = make_training_frame(df, calf, meta); wide = sales_wide(df, meta)
    last = df["date"].max()
    holdout_origin = last - pd.Timedelta(days=HORIZON - 1)
    val_origins = [holdout_origin - pd.Timedelta(days=HORIZON * k) for k in (4, 3, 2, 1)]
    print("validation origins:", [str(o.date()) for o in val_origins], "| holdout origin:", holdout_origin.date())

    grid_rows, val_by_cfg = [], []
    for i, cfg in enumerate(GRID):
        v = pd.concat([run_origin("validation", o, cfg, frame, wide, meta, calf) for o in val_origins])
        per_origin = summarize(v, ["origin"]).query("model == 'lgbm'")
        grid_rows.append({"config_id": i, **cfg, "mean_val_WAPE": per_origin["WAPE"].mean(),
                          "mean_val_MAE": per_origin["MAE"].mean()})
        val_by_cfg.append(v)
        print(f"config {i} {cfg} -> mean val WAPE {grid_rows[-1]['mean_val_WAPE']:.4f}")
    grid = pd.DataFrame(grid_rows); grid.to_csv(REPORTS / "validation_grid.csv", index=False)
    best = int(grid["mean_val_WAPE"].idxmin()); best_cfg = GRID[best]
    print("chosen config:", best, best_cfg)

    holdout = run_origin("holdout", holdout_origin, best_cfg, frame, wide, meta, calf)
    preds = pd.concat([val_by_cfg[best], holdout], ignore_index=True)
    preds.to_parquet(REPORTS / "backtest_predictions.parquet", index=False)
    summarize(preds, ["split"]).to_csv(REPORTS / "metrics_summary.csv", index=False)
    summarize(preds, ["split", "origin"]).to_csv(REPORTS / "metrics_by_origin.csv", index=False)
    summarize(preds, ["split", "product_id"]).to_csv(REPORTS / "metrics_by_product.csv", index=False)
    print("\n=== Aggregate metrics (bias>0 = overforecast) ===")
    print(summarize(preds, ["split"]).round(4).to_string(index=False))

    # Final model: all data through the last observed day, then a true post-cutoff forecast
    final = fit(frame, best_cfg)
    final.booster_.save_model(str(MODELS / "lgbm_model.txt"))
    origin = last + pd.Timedelta(days=1)
    dates, p = recursive_forecast(final, wide.to_numpy(float), meta, calf, origin)
    base = seasonal_naive(wide.to_numpy(float))
    fc = pd.DataFrame({"product_id": np.repeat(meta["product_id"].to_numpy(), HORIZON),
                       "date": np.tile(dates.to_numpy(), len(meta)),
                       "lgbm": p.ravel(), "baseline": base.ravel()})
    fc.to_parquet(DEMO / "forecast_next28.parquet", index=False)
    meta.to_csv(DEMO / "product_meta.csv", index=False)
    (MODELS / "artifacts.json").write_text(json.dumps({
        "model_version": "lgbm-" + datetime.now().strftime("%Y%m%d-%H%M"),
        "training_cutoff": str(last.date()), "forecast_start": str(origin.date()),
        "forecast_end": str(dates[-1].date()), "horizon": HORIZON,
        "features": FEATURES, "categorical": CATEGORICAL,
        "dtypes": frame[FEATURES].dtypes.astype(str).to_dict(),
        "chosen_config": best_cfg, "base_params": BASE_PARAMS, "grid": GRID,
        "validation_origins": [str(o.date()) for o in val_origins],
        "holdout_origin": str(holdout_origin.date()),
        "notes": "negative forecasts clipped to 0; recursive multi-step; bias = pred - actual",
        "products": meta.to_dict(orient="records")}, indent=2, default=str))
    print("\nSaved model, artifacts, reports, forecast_next28.parquet")


if __name__ == "__main__":
    main()
