"""Transparent replenishment logic. Stock inputs are user-entered or demo values, NOT real stock data."""
import math
import numpy as np


def _units(name, v):
    if v is None:
        raise ValueError(f"{name} is required")
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return v


def _days(name, v, minimum=0):
    v = _units(name, v)
    if v != int(v):
        raise ValueError(f"{name} must be a whole number of days")
    if int(v) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(v)


def plan_inventory(forecast, on_hand, on_order, lead_time_days, review_period_days,
                   safety_stock, on_order_arrival_days=None, data_mode="historical_demo"):
    """forecast: daily forecast demand, day 1 first. Returns a dict of structured results.

    Inventory position  = on_hand + on_order
    Planning horizon    = lead_time_days + review_period_days   (must fit in the forecast)
    Target stock        = forecast demand over horizon + safety_stock
    Suggested order     = max(0, ceil(target - position))

    Stockout timeline: a day is a stockout when cumulative forecast demand exceeds available stock.
    A new order arrives after lead_time_days (usable from day lead_time_days + 1).
    Existing on-order stock is counted only if on_order_arrival_days is given; otherwise it is
    EXCLUDED from the timeline (conservative) and the output says so.
    """
    f = np.asarray(forecast, dtype=float)
    if f.ndim != 1 or len(f) == 0 or not np.all(np.isfinite(f)) or (f < 0).any():
        raise ValueError("forecast must be a non-empty 1-D array of non-negative numbers")
    on_hand = _units("on_hand", on_hand)
    on_order = _units("on_order", on_order)
    safety_stock = _units("safety_stock", safety_stock)
    lead = _days("lead_time_days", lead_time_days)
    review = _days("review_period_days", review_period_days)
    horizon = lead + review
    if horizon < 1:
        raise ValueError("planning horizon (lead time + review period) must be at least 1 day")
    if horizon > len(f):
        raise ValueError(f"planning horizon of {horizon} days exceeds the {len(f)}-day forecast")
    arrival = None if on_order_arrival_days is None else _days("on_order_arrival_days", on_order_arrival_days)

    demand = round(float(f[:horizon].sum()), 6)
    position = on_hand + on_order
    target = demand + safety_stock
    order = max(0, math.ceil(round(target - position, 6)))

    cum = np.cumsum(f)
    days = np.arange(1, len(f) + 1)
    supply = on_hand + (on_order * (days > arrival) if arrival is not None else 0.0)
    hits = np.where(cum > supply)[0]
    first_stockout = int(days[hits[0]]) if len(hits) else None
    before_arrival = first_stockout is not None and first_stockout <= lead

    notes = ["Suggested replenishment quantity, not a guaranteed outcome.",
             "Stock inputs are user-entered or demo values; historical sales contain no stock levels.",
             "Not optimized: supplier minimums, costs, shelf life, delivery uncertainty, capacity, lost demand."]
    if arrival is None and on_order > 0:
        notes.append("Stock on order is excluded from the stockout timeline (no arrival date given).")
    if before_arrival:
        notes.append(f"Possible stockout on forecast day {first_stockout}, before a new order "
                     f"placed today would arrive (day {lead + 1}). Ordering more does not prevent this.")

    return {
        "planning_horizon_days": horizon,
        "forecast_demand_units": demand,
        "current_stock_units": on_hand,
        "stock_on_order_units": on_order,
        "inventory_position_units": position,
        "safety_stock_units": safety_stock,
        "target_stock_units": round(target, 6),
        "suggested_order_units": int(order),
        "first_stockout_day": first_stockout,
        "possible_stockout_before_replenishment": bool(before_arrival),
        "data_mode": data_mode,
        "notes": notes,
    }
