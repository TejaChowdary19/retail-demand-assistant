import numpy as np
import pandas as pd
import pytest
from src import service as sv
from src.config import DEMO, GRID, MODELS
from src.features import *
from src.forecast import sales_wide
from src.train import run_origin

pytestmark = pytest.mark.skipif(not (DEMO / "prepared_sales.parquet").exists(), reason="run data preparation first")


def _setup(df):
    calf = calendar_features(load_calendar())
    meta = product_table(df)
    return make_training_frame(df, calf, meta), sales_wide(df, meta), meta, calf


def test_backtest_ignores_actual_sales_in_forecast_window():
    df = load_prepared()
    origin = pd.Timestamp("2016-04-25")
    df2 = df.copy()
    df2.loc[df2.date >= origin, "units_sold"] += 1000   # corrupt every actual in the window
    a = run_origin("t", origin, GRID[0], *_setup(df))
    b = run_origin("t", origin, GRID[0], *_setup(df2))
    assert np.allclose(a["lgbm"], b["lgbm"], atol=1e-6)         # forecasts unchanged
    assert np.allclose(a["baseline"], b["baseline"], atol=1e-6)
    assert not np.allclose(a["actual"], b["actual"])             # only the scoring side moved


@pytest.mark.skipif(not (MODELS / "artifacts.json").exists(), reason="run training first")
def test_downloaded_forecast_matches_displayed_and_tool_numbers():
    import io
    pid = sv.products()[0]
    d = sv.forecast_frame(pid)
    csv = d[["product_id", "date", "lightgbm_forecast", "seasonal_naive_forecast"]].to_csv(index=False)
    back = pd.read_csv(io.StringIO(csv))
    tool = sv.get_product_forecast(pid)
    assert len(back) == 28
    assert abs(back["lightgbm_forecast"].sum() - tool["lightgbm_total_units"]) < 0.01
