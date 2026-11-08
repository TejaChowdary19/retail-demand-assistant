"""LLM tool layer: thin, validated wrappers over src.service, so tool numbers equal dashboard numbers."""
import json
import os
from src import service as sv

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
PID = {"type": "string", "description": "Product id, e.g. HOBBIES_1_394"}
HZ = {"type": "integer", "description": "Number of forecast days, 1 to 28"}
NUM = lambda d: {"type": "number", "description": d}


def _schema(props, required):
    return {"type": "object", "properties": props, "required": required}


TOOLS = [
    {"name": "get_product_forecast", "description": "28-day-max forecast (LightGBM and seasonal naive baseline) for one product after the data cutoff.",
     "input_schema": _schema({"product_id": PID, "horizon_days": HZ}, ["product_id"])},
    {"name": "get_inventory_recommendation", "description": "Suggested replenishment for one product, using the user's current planner inputs if they set any, otherwise labeled demo defaults. Also returns the stock inputs used.",
     "input_schema": _schema({"product_id": PID}, ["product_id"])},
    {"name": "simulate_inventory", "description": "What-if replenishment scenario. Does NOT change the user's saved inputs. All six inputs are required.",
     "input_schema": _schema({"product_id": PID, "on_hand": NUM("units in stock"), "on_order": NUM("units already on order"),
                              "lead_time_days": {"type": "integer"}, "review_period_days": {"type": "integer"},
                              "safety_stock": NUM("buffer units")},
                             ["product_id", "on_hand", "on_order", "lead_time_days", "review_period_days", "safety_stock"])},
    {"name": "get_products_at_risk", "description": "Products whose forecast demand exceeds available stock within horizon_days.",
     "input_schema": _schema({"horizon_days": HZ}, [])},
    {"name": "get_top_replenishment", "description": "Products ranked by suggested order quantity (largest first).",
     "input_schema": _schema({"top_n": {"type": "integer", "description": "1 to 20, default 5"}}, [])},
    {"name": "get_model_metrics", "description": "Measured backtest metrics (MAE, WAPE, bias) for LightGBM vs seasonal naive; aggregate, or per product if product_id is given.",
     "input_schema": _schema({"product_id": PID}, [])},
]


def run_tool(name, args, saved_inputs=None):
    args, saved = dict(args or {}), saved_inputs or {}
    try:
        if name == "get_product_forecast":
            return sv.get_product_forecast(args["product_id"], args.get("horizon_days", 28))
        if name == "get_inventory_recommendation":
            pid = sv.check_product(args["product_id"])
            return sv.get_inventory_recommendation(pid, saved.get(pid))
        if name == "simulate_inventory":
            keys = ["product_id", "on_hand", "on_order", "lead_time_days", "review_period_days", "safety_stock"]
            missing = [k for k in keys if k not in args]
            if missing:
                return {"error": f"Missing required inputs: {missing}. Ask the user for them; do not guess."}
            return sv.simulate_inventory(*[args[k] for k in keys])
        if name == "get_products_at_risk":
            return sv.get_products_at_risk(args.get("horizon_days", 7), saved)
        if name == "get_top_replenishment":
            n = sv.check_horizon(args.get("top_n", 5), maximum=20, name="top_n")
            t = sv.inventory_table(saved).sort_values("suggested_order_units", ascending=False).head(n)
            return {"top_n": n, "data_mode": "historical_demo",
                    "products": t.astype(object).where(t.notna(), None).to_dict("records"),
                    "note": "Uses the user's planner inputs where set, otherwise demo stock values."}
        if name == "get_model_metrics":
            return sv.get_model_metrics(args.get("product_id"))
        return {"error": f"Unknown tool '{name}'"}
    except KeyError as e:
        return {"error": f"Missing required argument {e}"}
    except ValueError as e:
        return {"error": str(e)}


SYSTEM = """You are the assistant inside a retail demand forecasting demo (historical M5 data, one store, 20 products).
Rules:
- Get every number from tools. Never compute forecasts, order quantities or metrics yourself, and never invent them.
- If a tool returns an error, tell the user what was wrong. If required inputs are missing, ask for them.
- The data is historical and ends at the training cutoff. Forecasts are demonstrations on old data, not current predictions. Say 'forecast' for post-cutoff numbers and 'backtest' or 'actual' for historical ones.
- Stock levels are user-entered or demo values (input_source says which). Say so when you quote stock-based results.
- For what-if questions ('what if delivery takes 10 days'), call get_inventory_recommendation first to learn the current inputs, then simulate_inventory with only the changed value. A simulation never changes saved inputs; say so.
- Do not invent reasons why sales or recommendations changed. You may explain the arithmetic (inventory position, target stock) and compare scenario numbers. Do not present model behavior as causal proof.
- Never claim an order guarantees no stockout. If possible_stockout_before_replenishment is true, say ordering more will not fix a stockout that happens before delivery.
- Report errors with the sign convention: positive bias = overforecast. WAPE null means undefined. Do not turn metrics into an 'accuracy percentage'.
- If the user's saved inputs differ from an earlier answer in this chat, say the inputs changed; never call the earlier answer wrong.
- Be concise and plain-English."""


def build_context(pid, scenario, art):
    sc = json.dumps(scenario) if scenario else "none set (demo defaults apply)"
    return (f"\n\nApp context: selected product = {pid}. User's saved planner inputs for it: {sc}. "
            f"Training cutoff {art['training_cutoff']}; forecast {art['forecast_start']} to {art['forecast_end']}. "
            f"Valid product ids: {', '.join(sv.products())}.")


def chat(messages, api_key, context="", saved_inputs=None, max_steps=6):
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msgs, trace = list(messages), []
    for _ in range(max_steps):
        resp = client.messages.create(model=MODEL, max_tokens=1000, system=SYSTEM + context,
                                      tools=TOOLS, messages=msgs)
        msgs.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text"), trace
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                out = run_tool(b.name, b.input, saved_inputs)
                trace.append({"tool": b.name, "input": dict(b.input), "output": out})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out, default=str)})
        msgs.append({"role": "user", "content": results})
    return "I couldn't finish that within the tool-call limit. Try a more specific question.", trace
