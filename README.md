# Retail Demand Forecasting & AI Inventory Planning Assistant

Decision-support prototype on the historical M5 dataset (store CA_1, 20 products).
Forecasts 28 days with LightGBM (recursive) vs a seasonal naive baseline, converts forecasts to suggested
replenishment quantities, and uses an LLM assistant with structured tool calls to explain results.

**Live app:** https://retail-demand-teja.streamlit.app

## Run locally
    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...   # optional; app works without it except the AI Assistant
    streamlit run app.py

## Reproduce
    # put sales_train_evaluation.csv and calendar.csv from Kaggle M5 into data/raw/
    python src/data_preparation.py
    python -m src.train
    python -m pytest -q

## Method
- Products chosen from the 365 days before the final 28-day holdout (zero fraction <= 0.5; 7 high / 7 medium / 6 low volume tertiles).
- Features use only information before the forecast date (lags 1/7/14/28, shifted rolling mean/std, calendar, event and SNAP flags).
- Validation: 4 rolling 28-day origins, 4 LightGBM configs, chosen by mean WAPE. Final choice scored once on an untouched 28-day holdout.
- Recursive forecasting feeds predictions (never actuals) back as history. Negative forecasts are clipped to 0.
- Bias = prediction minus actual (positive = overforecast). WAPE is undefined when total actual sales are zero.

## Limitations
Dataset is historical (ends 2016-05-22), so forecasts are demonstrations, not current predictions.
Stock levels are demo or user-entered. Sales may understate demand when items were out of stock.
No prices, no validated prediction intervals, no supplier minimums, costs, shelf life or delivery uncertainty.
LightGBM does not beat the baseline on every product; see the Model Evaluation page.

## Contributors
Divya Teja. Built November 8, 2026.
