import pytest
from src.inventory import plan_inventory

# 14-day planning horizon summing to exactly 75 units, as in the brief's example
F = [5] * 13 + [10] + [5] * 14


def test_brief_example():
    r = plan_inventory(F, on_hand=50, on_order=0, lead_time_days=7, review_period_days=7, safety_stock=10)
    assert r["forecast_demand_units"] == 75
    assert r["target_stock_units"] == 85
    assert r["suggested_order_units"] == 35


def test_order_never_negative():
    r = plan_inventory(F, 500, 0, 7, 7, 10)
    assert r["suggested_order_units"] == 0


def test_on_order_reduces_suggestion():
    r = plan_inventory(F, 50, 20, 7, 7, 10)
    assert r["suggested_order_units"] == 15


def test_ceiling_applied():
    r = plan_inventory([2.5] * 28, 0, 0, 2, 2, 0)  # demand 10.0
    assert r["suggested_order_units"] == 10
    r = plan_inventory([2.6] * 28, 0, 0, 2, 2, 0)  # demand 10.4 -> 11
    assert r["suggested_order_units"] == 11


def test_stockout_before_replenishment():
    r = plan_inventory([5] * 28, on_hand=20, on_order=0, lead_time_days=7, review_period_days=7, safety_stock=0)
    assert r["first_stockout_day"] == 5
    assert r["possible_stockout_before_replenishment"] is True


def test_on_order_arrival_counts_in_timeline():
    r = plan_inventory([5] * 28, 20, 100, 7, 7, 0, on_order_arrival_days=3)
    assert r["first_stockout_day"] == 25
    assert r["possible_stockout_before_replenishment"] is False


@pytest.mark.parametrize("kwargs", [
    dict(on_hand=-1), dict(on_order=-5), dict(safety_stock=-1),
    dict(lead_time_days=-1), dict(lead_time_days=20, review_period_days=20),  # exceeds 28-day forecast
    dict(on_hand=None), dict(lead_time_days=2.5), dict(on_hand="abc"),
    dict(lead_time_days=0, review_period_days=0),
])
def test_validation(kwargs):
    base = dict(on_hand=50, on_order=0, lead_time_days=7, review_period_days=7, safety_stock=10)
    base.update(kwargs)
    with pytest.raises(ValueError):
        plan_inventory([5] * 28, **base)
