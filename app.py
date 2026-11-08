import streamlit as st

st.set_page_config(page_title="Retail Demand Assistant", page_icon="📦", layout="wide")

import plotly.express as px
from src import service as sv

try:
    art = sv.artifacts()
except Exception as e:
    st.error(f"Model artifacts could not be loaded: {e}. Run `python -m src.train` first.")
    st.stop()


def overview():
    st.title("Retail Demand Forecasting & Inventory Planning")
    st.caption("Decision-support prototype built on the historical M5 dataset. Not a live company system.")
    st.markdown(
        "**Objective:** forecast the next 28 days of unit sales for ~20 products in one store, turn the forecast "
        "into suggested replenishment quantities, and explain results with an LLM assistant that reads numbers "
        "from tools.")
    if st.button("Load / reset demo data"):
        for k in [k for k in st.session_state if str(k).startswith("inv_")]:
            del st.session_state[k]
        st.session_state.pop("_saved_inputs", None)
        st.success("Demo data loaded. Inventory scenarios reset to demo values.")

    hist = sv.history_all()
    c = st.columns(4)
    c[0].metric("Store / products", f"CA_1 / {hist.product_id.nunique()}")
    c[1].metric("History", f"{hist.date.min().date()} to {hist.date.max().date()}")
    c[2].metric("Forecast cutoff", art["training_cutoff"])
    c[3].metric("Forecast dates", f"{art['forecast_start']} to {art['forecast_end']}")
    st.info("The dataset is historical (Walmart M5, ends May 2016). The 28-day forecast is a demonstration "
            "on old data, not a current market prediction. Stock levels used later are demo or user-entered values.")

    total = hist.groupby("date").units_sold.sum().to_frame("units").tail(365)
    total["7-day average"] = total["units"].rolling(7).mean()
    st.plotly_chart(px.line(total, y=["units", "7-day average"], title="Total daily units, last 365 days"),
                    width="stretch")
    a, b = st.columns(2)
    wd = hist.groupby(hist.date.dt.day_name()).units_sold.mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    a.plotly_chart(px.bar(wd, title="Average units per product by weekday", labels={"value": "units", "date": ""}),
                   width="stretch")
    zero = hist.groupby("product_id").units_sold.apply(lambda s: (s == 0).mean() * 100).sort_values()
    b.plotly_chart(px.bar(zero, title="Zero-sales days by product (%)", labels={"value": "% of days"}),
                   width="stretch")
    mean = hist.groupby("product_id").units_sold.mean().sort_values()
    st.plotly_chart(px.bar(mean, title="Average daily units by product (demand level)", labels={"value": "units/day"}),
                    width="stretch")
    st.caption("Zero-sales days are kept as genuine zeros. Low sales may reflect low demand or no stock; the data cannot tell which.")


pg = st.navigation([
    st.Page(overview, title="Overview", default=True),
    st.Page("pages/forecast_explorer.py", title="Forecast Explorer"),
    st.Page("pages/inventory_planner.py", title="Inventory Planner"),
    st.Page("pages/ai_assistant.py", title="AI Assistant"),
    st.Page("pages/model_evaluation.py", title="Model Evaluation"),
])
pg.run()
