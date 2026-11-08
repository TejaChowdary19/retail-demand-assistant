import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src import service as sv
from src.ui import product_picker

art, R = sv.artifacts(), sv.reports()
st.title("Model Evaluation")
st.markdown(f"""
**Method.** One global LightGBM model across 20 products, forecasting 28 days recursively (each prediction is fed back as
history; actuals are never used inside the forecast window). Compared with a seasonal naive baseline (repeat the last 7 days) on
identical windows. Model version `{art['model_version']}`, training cutoff {art['training_cutoff']}.
Validation: 4 rolling 28-day origins ({', '.join(art['validation_origins'])}) used to choose among
{len(art['grid'])} configurations. The final choice was scored once on an untouched holdout starting {art['holdout_origin']}.
""")
st.caption("Bias sign convention: positive = overforecast (prediction minus actual). WAPE = total absolute error / total actual units; "
           "blank = undefined (zero actual sales). No 'accuracy %' is reported.")

st.subheader("Aggregate metrics")
s = R["summary"].round(4)
st.dataframe(s, width="stretch", hide_index=True)

st.subheader("Hyperparameter search (validation, mean WAPE across origins)")
st.dataframe(R["grid"].round(4), width="stretch", hide_index=True)

st.subheader("Performance across backtest periods")
bo = R["by_origin"]
bo["origin"] = bo["origin"].astype(str).str[:10]
st.plotly_chart(px.bar(bo, x="origin", y="WAPE", color="model", barmode="group", facet_col="split",
                       title="WAPE by forecast origin").update_xaxes(type="category", matches=None), width="stretch")

st.subheader("Error by product")
bp = R["by_product"]
split = st.radio("Split", ["holdout", "validation"], horizontal=True)
p = bp[bp.split == split]
st.plotly_chart(px.bar(p, x="product_id", y="MAE", color="model", barmode="group", title=f"MAE by product ({split})"),
                width="stretch")
mae = p.pivot(index="product_id", columns="model", values="MAE")
worse = mae[mae["baseline"] < mae["lgbm"]].round(3)
st.markdown(f"**Products where the baseline has lower MAE ({split}):** {len(worse)} of {len(mae)}")
if len(worse):
    st.dataframe(worse, width="stretch")
st.dataframe(p.round(4), width="stretch", hide_index=True)

st.subheader("Actual vs predicted")
pid = product_picker()
bt = sv.backtest_frame()
d = bt[(bt.product_id == pid) & (bt.split == "holdout")]
fig = go.Figure()
fig.add_scatter(x=d.date, y=d.actual, name="Actual")
fig.add_scatter(x=d.date, y=d.lgbm, name="LightGBM")
fig.add_scatter(x=d.date, y=d.baseline, name="Seasonal naive", line=dict(dash="dot"))
fig.update_layout(title=f"Holdout window, {pid}")
st.plotly_chart(fig, width="stretch")

st.subheader("Known limitations")
st.markdown("""
- Retail unit sales are intermittent, so WAPE is high; per-product errors vary widely.
- Differences between the tuned configurations are small; tuning was deliberately limited.
- Recursive forecasting lets errors compound over the 28 days.
- Low historical sales may reflect stockouts rather than low demand; the data has no stock records.
- The holdout shows overforecasting on average (see bias); one 28-day window is a small sample.
- Event/SNAP flags are assumed known in advance. Prices are not used. No validated prediction intervals.
- The dataset is historical (to May 2016); forecasts are demonstrations, not current predictions.
""")
