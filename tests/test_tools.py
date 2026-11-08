import copy
import pytest
from src.config import MODELS
from src import service as sv, llm_tools as lt

pytestmark = pytest.mark.skipif(not (MODELS / "artifacts.json").exists(), reason="run training first")
PID = sv.products()[0]
SAVED = {PID: dict(on_hand=5, on_order=0, lead_time_days=7, review_period_days=7, safety_stock=3)}


def test_forecast_tool_matches_service():
    assert lt.run_tool("get_product_forecast", {"product_id": PID, "horizon_days": 14}) == sv.get_product_forecast(PID, 14)


def test_inventory_tool_uses_saved_planner_inputs():
    a = lt.run_tool("get_inventory_recommendation", {"product_id": PID}, SAVED)
    assert a == sv.get_inventory_recommendation(PID, SAVED[PID])
    assert a["input_source"] == "user_entered"


def test_simulation_does_not_overwrite_saved_inputs():
    before = copy.deepcopy(SAVED)
    sim = lt.run_tool("simulate_inventory", {"product_id": PID, "on_hand": 5, "on_order": 0, "lead_time_days": 10,
                                             "review_period_days": 7, "safety_stock": 3}, SAVED)
    assert sim["input_source"] == "scenario_simulation" and sim["planning_horizon_days"] == 17
    assert SAVED == before


def test_at_risk_top_and_metrics_match_service():
    assert lt.run_tool("get_products_at_risk", {"horizon_days": 7}) == sv.get_products_at_risk(7)
    assert lt.run_tool("get_model_metrics", {}) == sv.get_model_metrics(None)
    top = lt.run_tool("get_top_replenishment", {"top_n": 3})
    orders = [p["suggested_order_units"] for p in top["products"]]
    assert len(orders) == 3 and orders == sorted(orders, reverse=True)


@pytest.mark.parametrize("name,args", [
    ("get_product_forecast", {"product_id": "NOPE"}),
    ("get_product_forecast", {"product_id": PID, "horizon_days": 99}),
    ("get_model_metrics", {"product_id": "NOPE"}),
    ("simulate_inventory", {"product_id": PID, "on_hand": -1, "on_order": 0, "lead_time_days": 7, "review_period_days": 7, "safety_stock": 0}),
    ("simulate_inventory", {"product_id": PID, "on_hand": 5}),
    ("simulate_inventory", {"product_id": PID, "on_hand": 5, "on_order": 0, "lead_time_days": 20, "review_period_days": 20, "safety_stock": 0}),
    ("not_a_tool", {}),
])
def test_bad_requests_return_errors_not_exceptions(name, args):
    assert "error" in lt.run_tool(name, args)


def test_core_works_without_llm():
    with pytest.raises(RuntimeError):
        lt.chat([{"role": "user", "content": "hi"}], None)
    assert sv.get_inventory_recommendation(PID)["suggested_order_units"] >= 0
