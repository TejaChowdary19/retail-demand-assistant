"""Single source of truth for every number. Dashboard pages AND LLM tools call these functions."""
import json
import math
from functools import lru_cache
import pandas as pd
from src.config import DEMO, MODELS, REPORTS, HORIZON
from src.inventory import plan_inventory


@lru_cache(maxsize=1)
def _load():
    meta = pd.read_csv(DEMO / "product_meta.csv")
    hist = pd.read_parquet(DEMO / "prepared_sales.parquet")
    hist["date"] = pd.to_datetime(hist["date"])
    fc = pd.read_parquet(DEMO / "forecast_next28.parquet")
    fc["date"] = pd.to_datetime(fc["date"])
    art = json.loads((MODELS / "artifacts.json").read_text())
    return meta, hist, fc, art


def artifacts():
    return _load()[3]


def products():
    return _load()[0]["product_id"].tolist()


def history_all():
    return _load()[1].copy()


def check_product(product_id):
    pid = str(product_id or "").strip().upper()
    if pid not in set(products()):
        raise ValueError(f"Unknown product_id '{product_id}'. Valid ids: {', '.join(products())}")
    return pid


def check_horizon(h, maximum=HORIZON, name="horizon_days"):
    try:
        v = float(h)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number between 1 and {maximum}")
    if v != int(v) or not 1 <= v <= maximum:
        raise ValueError(f"{name} must be a whole number between 1 and {maximum}")
    return int(v)


def forecast_frame(product_id, horizon_days=HORIZON):
    pid, h = check_product(product_id), check_horizon(horizon_days)
    fc = _load()[2]
    d = fc[fc.product_id == pid].sort_values("date").head(h).reset_index(drop=True)
    return d.rename(columns={"lgbm": "lightgbm_forecast", "baseline": "seasonal_naive_forecast"})


def history_frame(product_id):
    pid = check_product(product_id)
    h = _load()[1]
    return h[h.product_id == pid].sort_values("date")[["date", "units_sold"]].reset_index(drop=True)


def get_product_forecast(product_id, horizon_days=HORIZON):
    d = forecast_frame(product_id, horizon_days)
    a = artifacts()
    return {
        "product_id": d.product_id.iloc[0], "data_type": "forecast (after dataset cutoff, no actuals exist)",
        "training_cutoff": a["training_cutoff"], "horizon_days": len(d),
        "forecast_start": str(d.date.min().date()), "forecast_end": str(d.date.max().date()),
        "lightgbm_total_units": round(float(d.lightgbm_forecast.sum()), 3),
        "seasonal_naive_total_units": round(float(d.seasonal_naive_forecast.sum()), 3),
        "daily": [{"date": str(r.date.date()), "lightgbm": round(float(r.lightgbm_forecast), 3),
                   "seasonal_naive": round(float(r.seasonal_naive_forecast), 3)} for r in d.itertuples()],
        "note": "Historical dataset: dates are in the past, so this is a demo forecast, not a current prediction.",
    }


def default_inputs(product_id):
    """DEMO stock values (history has no real stock levels). Cover varies by product so the demo is not uniform."""
    pid = check_product(product_id)
    avg = float(forecast_frame(pid)["lightgbm_forecast"].mean())
    cover_days = 3 + (products().index(pid) * 5) % 12
    return {"on_hand": math.ceil(avg * cover_days), "on_order": 0, "lead_time_days": 7,
            "review_period_days": 7, "safety_stock": math.ceil(avg * 2)}


def _plan(pid, p, source):
    f = forecast_frame(pid)["lightgbm_forecast"].to_numpy()
    r = plan_inventory(f, p["on_hand"], p["on_order"], p["lead_time_days"], p["review_period_days"],
                       p["safety_stock"], p.get("on_order_arrival_days"), data_mode="historical_demo")
    r.update(product_id=pid, input_source=source, review_period_days=int(p["review_period_days"]),
             lead_time_days=int(p["lead_time_days"]))
    return r


def get_inventory_recommendation(product_id, inputs=None, source=None):
    pid = check_product(product_id)
    p = {**default_inputs(pid), **(inputs or {})}
    return _plan(pid, p, source or ("user_entered" if inputs else "demo_default"))


def simulate_inventory(product_id, on_hand, on_order, lead_time_days, review_period_days, safety_stock):
    """What-if. Pure function: never touches saved/UI inputs."""
    pid = check_product(product_id)
    p = dict(on_hand=on_hand, on_order=on_order, lead_time_days=lead_time_days,
             review_period_days=review_period_days, safety_stock=safety_stock)
    return _plan(pid, p, "scenario_simulation")


COLS = ["product_id", "current_stock_units", "stock_on_order_units", "safety_stock_units",
        "planning_horizon_days", "forecast_demand_units", "target_stock_units", "suggested_order_units",
        "first_stockout_day", "possible_stockout_before_replenishment", "input_source", "data_mode"]


def inventory_table(inputs_by_product=None):
    rows = []
    for pid in products():
        inp = (inputs_by_product or {}).get(pid)
        r = get_inventory_recommendation(pid, inp)
        rows.append({k: r[k] for k in COLS})
    return pd.DataFrame(rows)


def get_products_at_risk(horizon_days=7, inputs_by_product=None):
    h = check_horizon(horizon_days)
    t = inventory_table(inputs_by_product)
    risk = t[t.first_stockout_day.notna() & (t.first_stockout_day <= h)].sort_values("first_stockout_day")
    return {"horizon_days": h, "data_mode": "historical_demo", "n_at_risk": len(risk),
            "at_risk": risk.astype(object).where(risk.notna(), None).to_dict("records"),
            "note": "Uses demo stock values unless inputs are supplied. Stock on order is excluded from the timeline."}


def reports():
    return {"summary": pd.read_csv(REPORTS / "metrics_summary.csv"),
            "by_origin": pd.read_csv(REPORTS / "metrics_by_origin.csv"),
            "by_product": pd.read_csv(REPORTS / "metrics_by_product.csv"),
            "grid": pd.read_csv(REPORTS / "validation_grid.csv")}


def backtest_frame():
    d = pd.read_parquet(REPORTS / "backtest_predictions.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d["origin"] = pd.to_datetime(d["origin"])
    return d


def get_model_metrics(product_id=None):
    r = reports()
    if product_id is None:
        df, scope = r["summary"], "aggregate_all_products"
    else:
        pid = check_product(product_id)
        df, scope = r["by_product"][r["by_product"].product_id == pid], f"product:{pid}"
    df = df.round(4).astype(object)
    a = artifacts()
    return {"scope": scope, "metrics": df.where(df.notna(), None).to_dict("records"),
            "bias_sign_convention": "positive = overforecast (prediction minus actual)",
            "wape_note": "WAPE = total absolute error / total actual units; null means undefined (zero actual sales).",
            "validation_origins": a["validation_origins"], "holdout_origin": a["holdout_origin"],
            "model_version": a["model_version"]}
