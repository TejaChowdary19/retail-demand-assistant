import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src import service as sv
from src.ui import product_picker

art = sv.artifacts()
st.title("Forecast Explorer")
pid = product_picker()
tab_fc, tab_bt = st.tabs(["Forecast (after data cutoff)", "Historical backtest (actuals known)"])

with tab_fc:
    st.info(f"Sales data ends {art['training_cutoff']}. The forecast covers {art['forecast_start']} to "
            f"{art['forecast_end']}. No actual sales exist for these dates, and the dataset is historical, "
            "so this is a demo forecast, not a current prediction.")
    h = st.slider("Forecast horizon (days)", 7, 28, 28)
    n_hist = st.slider("History to show (days)", 28, 365, 120)
    fc, hist = sv.forecast_frame(pid, h), sv.history_frame(pid).tail(n_hist)
    fig = go.Figure()
    fig.add_scatter(x=hist.date, y=hist.units_sold, name="Historical sales (actual)", line=dict(color="gray"))
    fig.add_scatter(x=fc.date, y=fc.lightgbm_forecast, name="LightGBM forecast")
    fig.add_scatter(x=fc.date, y=fc.seasonal_naive_forecast, name="Seasonal naive baseline", line=dict(dash="dot"))
    cut = art["training_cutoff"]
    fig.add_shape(type="line", x0=cut, x1=cut, y0=0, y1=1, yref="paper", line=dict(dash="dash", color="red"))
    fig.add_annotation(x=cut, y=1, yref="paper", text="data cutoff", showarrow=False, xanchor="right")
    fig.update_layout(yaxis_title="units/day", legend_orientation="h")
    st.plotly_chart(fig, width="stretch")
    c = st.columns(2)
    c[0].metric(f"LightGBM total, {h} days", f"{fc.lightgbm_forecast.sum():.1f} units")
    c[1].metric(f"Seasonal naive total, {h} days", f"{fc.seasonal_naive_forecast.sum():.1f} units")
    out = fc[["product_id", "date", "lightgbm_forecast", "seasonal_naive_forecast"]].copy()
    out["date"] = out.date.dt.date
    st.dataframe(out.round(3), width="stretch", hide_index=True)
    st.download_button("Download forecast (CSV)", out.to_csv(index=False), f"forecast_{pid}.csv", "text/csv")
    st.caption("Forecasts are continuous expected values; negatives are clipped to 0. No uncertainty bands are shown "
               "because none were validated.")

with tab_bt:
    bt = sv.backtest_frame()
    bt = bt[bt.product_id == pid]
    split = st.radio("Split", ["holdout", "validation"], horizontal=True)
    d = bt[bt.split == split]
    if split == "validation":
        origins = sorted(d.origin.unique())
        o = st.selectbox("Forecast origin", origins, format_func=lambda x: str(pd.Timestamp(x).date()))
        d = d[d.origin == o]
    st.info("Backtest: actual sales are known for these 28 days and were used only for scoring, never as model input.")
    fig = go.Figure()
    fig.add_scatter(x=d.date, y=d.actual, name="Actual")
    fig.add_scatter(x=d.date, y=d.lgbm, name="LightGBM (recursive)")
    fig.add_scatter(x=d.date, y=d.baseline, name="Seasonal naive", line=dict(dash="dot"))
    st.plotly_chart(fig, width="stretch")
    st.json(sv.get_model_metrics(pid), expanded=False)
