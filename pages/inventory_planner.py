import plotly.graph_objects as go
import streamlit as st
from src import service as sv
from src.ui import product_picker

st.title("Inventory Planner")
st.warning("Stock inputs are user-entered or demo values. The historical sales data contains no real stock levels.")
pid = product_picker()
d = sv.default_inputs(pid)
k = lambda n: f"inv_{pid}_{n}"   # per-session, per-product keys keep each user's scenario separate


def reset():
    for n in ["on_hand", "on_order", "lead", "review", "safety", "use_arrival", "arrival"]:
        st.session_state.pop(k(n), None)


c = st.columns(5)
on_hand = c[0].number_input("Current stock (units)", min_value=0, value=int(d["on_hand"]), step=1, key=k("on_hand"))
on_order = c[1].number_input("Stock on order (units)", min_value=0, value=int(d["on_order"]), step=1, key=k("on_order"))
lead = c[2].number_input("Lead time (days)", min_value=0, max_value=28, value=int(d["lead_time_days"]), step=1, key=k("lead"))
review = c[3].number_input("Review period (days)", min_value=0, max_value=28, value=int(d["review_period_days"]), step=1, key=k("review"))
safety = c[4].number_input("Safety stock (units)", min_value=0, value=int(d["safety_stock"]), step=1, key=k("safety"))
arrival = None
if on_order > 0 and st.checkbox("I know when the on-order stock arrives", key=k("use_arrival")):
    arrival = st.number_input("On-order arrival (forecast day)", min_value=0, max_value=28, value=3, step=1, key=k("arrival"))
st.button("Reset to demo values", on_click=reset)

inputs = dict(on_hand=on_hand, on_order=on_order, lead_time_days=lead, review_period_days=review,
              safety_stock=safety, on_order_arrival_days=arrival)
try:
    r = sv.get_inventory_recommendation(pid, inputs)
except ValueError as e:
    st.error(str(e))
    st.stop()
r["input_source"] = "demo_default" if all(inputs[x] == d[x] for x in d) and arrival is None else "user_entered"
st.caption(f"Input source: **{r['input_source']}** | data mode: **{r['data_mode']}**")

m = st.columns(4)
m[0].metric("Planning horizon", f"{r['planning_horizon_days']} days")
m[1].metric("Forecast demand over horizon", f"{r['forecast_demand_units']:.1f} units")
m[2].metric("Target stock", f"{r['target_stock_units']:.1f} units")
m[3].metric("Suggested order quantity", f"{r['suggested_order_units']} units")
if r["possible_stockout_before_replenishment"]:
    st.error(f"Possible stockout on forecast day {r['first_stockout_day']}, before a new order placed today "
             f"would arrive (day {r['lead_time_days'] + 1}). Ordering more does not prevent this.")
elif r["first_stockout_day"]:
    st.warning(f"Forecast demand exceeds available stock on day {r['first_stockout_day']}, after a new order would arrive.")
else:
    st.success("No stockout in the 28-day forecast under these inputs.")

fc = sv.forecast_frame(pid)
cum = fc.lightgbm_forecast.cumsum()
fig = go.Figure()
fig.add_scatter(x=list(range(1, len(cum) + 1)), y=cum, name="Cumulative forecast demand")
fig.add_hline(y=on_hand, line_dash="dash", annotation_text="Current stock")
fig.add_vline(x=lead + 0.5, line_dash="dot", annotation_text="new order arrives")
fig.update_layout(xaxis_title="Forecast day", yaxis_title="units")
st.plotly_chart(fig, width="stretch")

with st.expander("Calculation and assumptions", expanded=True):
    st.markdown(f"""
- Inventory position = stock {on_hand} + on order {on_order} = **{on_hand + on_order}**
- Planning horizon = lead time {lead} + review {review} = **{r['planning_horizon_days']} days**
- Target stock = forecast demand {r['forecast_demand_units']:.1f} + safety stock {safety} = **{r['target_stock_units']:.1f}**
- Suggested order = max(0, ceil(target - position)) = **{r['suggested_order_units']}**
""")
    for n in r["notes"]:
        st.markdown(f"- {n}")

row = {k_: v for k_, v in r.items() if k_ != "notes"}
row["notes"] = " | ".join(r["notes"])
import pandas as pd
st.download_button("Download recommendation (CSV)", pd.DataFrame([row]).to_csv(index=False),
                   f"recommendation_{pid}.csv", "text/csv")

with st.expander("All products (demo stock values)"):
    t = sv.inventory_table()
    st.dataframe(t.round(2), width="stretch", hide_index=True)
    st.download_button("Download all-product table (CSV)", t.to_csv(index=False), "recommendations_all_demo.csv", "text/csv")

st.session_state.setdefault("_saved_inputs", {})[pid] = {n_: v_ for n_, v_ in inputs.items() if v_ is not None}
