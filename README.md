# Retail Demand Forecasting & AI Inventory Planning Assistant

A decision-support prototype built on the historical M5 retail dataset. It forecasts the next 28 days of unit sales
for 20 products in one store, turns forecasts into suggested replenishment quantities, and includes an LLM assistant
that explains results by calling structured tools.

**Live app:** https://retail-demand-teja.streamlit.app
(Free hosting: after inactivity the app may show a "wake up" button. Click it and wait a few seconds.)
**Demo recording:** (add link)

> The dataset is historical (sales end 2016-05-22). Forecasts cover 2016-05-23 to 2016-06-19 and are a demonstration
> on old data, not current market predictions. Stock levels are user-entered or labeled demo values.

## What the app does

| Page | Purpose |
|---|---|
| Overview | Objective, dataset, date ranges, "Load / reset demo data", summary charts |
| Forecast Explorer | Per-product history, 28-day LightGBM forecast vs seasonal-naive baseline, historical backtest view, CSV download |
| Inventory Planner | Editable stock, lead time, review period and safety stock; target stock, suggested order, stockout warning, visible calculation, CSV download |
| AI Assistant | Chat with tool calls into the same functions as the dashboard; tool results can be inspected under each answer |
| Model Evaluation | Method, baseline comparison, aggregate and per-product metrics, backtest periods, products where the baseline wins, limitations |

## Results (measured, 28-day recursive forecasts, 20 products)

| Split | Model | MAE | WAPE | Bias, units/day (+ = overforecast) |
|---|---|---|---|---|
| Holdout (2016-04-25 to 2016-05-22) | LightGBM | 1.975 | 0.715 | +0.54 |
| Holdout | Seasonal naive | 2.179 | 0.788 | +0.49 |
| Validation (4 origins, pooled) | LightGBM | 1.674 | 0.612 | -0.13 |
| Validation (4 origins, pooled) | Seasonal naive | 2.042 | 0.747 | -0.20 |

LightGBM beats the baseline on aggregate, but the margin is modest and it does not win on every product (3 of 20 on
the holdout). WAPE is high because many items sell only a few units a day. No "accuracy percentage" is reported.
Full tables: Model Evaluation page and `reports/`.

## Method

- **Data:** M5 store CA_1. Products were chosen using only the 365 days before the final 28-day holdout (zero-sales
  fraction <= 0.5; 7 high, 7 medium and 6 low volume tertiles; seed 42). See `data/demo/selection.json`. Zero-sales days
  are kept as genuine observations; no rows were excluded.
- **Features:** lags 1/7/14/28, rolling mean (7/14/28) and std (7/28) computed on shifted sales, calendar fields, event
  and SNAP flags (known in advance), product/department/category codes. Prices are not used.
- **Models:** seasonal-naive baseline (repeat the last 7 days) and one global LightGBM model.
- **Forecasting:** recursive. Each day's prediction is appended to a working copy of history and features are recomputed.
  Actual sales are never used inside a forecast window. Negative forecasts are clipped to 0 and kept continuous.
- **Evaluation:** four rolling 28-day validation origins, four LightGBM configurations chosen by mean WAPE, then one
  scoring on an untouched 28-day holdout. Baseline and LightGBM use identical origins. Bias = prediction minus actual.
  WAPE is undefined when total actual sales are zero.
- **Inventory:** position = stock + on order; horizon = lead time + review period (must fit in the 28-day forecast);
  target = forecast demand over horizon + safety stock; suggested order = max(0, ceil(target - position)). The stockout
  timeline distinguishes a stockout before a new order could arrive (ordering more cannot fix that) from the quantity
  needed to reach target stock. On-order stock is excluded from the timeline unless an arrival day is given.
- **LLM assistant:** Anthropic API with tool calling. The model must fetch every number from tools, ask for missing
  inputs, reject unknown products, label simulated inputs, and never claim an order guarantees no stockout. What-if
  scenarios never overwrite saved inputs. Sessions are capped at 15 questions. If the LLM is unavailable or no key is
  configured, the other pages keep working.

## Repository structure

    app.py                  Streamlit entry point and Overview page
    pages/                  Forecast Explorer, Inventory Planner, AI Assistant, Model Evaluation
    src/
      config.py             paths, features, hyperparameter grid
      data_preparation.py   M5 -> one store, 20 products, validated parquet
      features.py           leakage-safe lag, rolling and calendar features
      forecast.py           seasonal naive and recursive LightGBM forecasting
      evaluate.py           MAE, WAPE, bias
      train.py              backtesting, model selection, holdout, final model, artifacts
      inventory.py          replenishment and stockout logic
      service.py            single source of every number, used by pages and LLM tools
      llm_tools.py          tool schemas, validation, tool-calling loop
      ui.py                 shared Streamlit helpers
    notebooks/              exploratory_analysis.ipynb
    data/demo/              prepared data and forecasts shipped with the app
    models/                 saved LightGBM model and artifacts.json (config, features, cutoff, version)
    reports/                backtest predictions and metrics CSVs
    tests/                  leakage, recursion, backtest, inventory, tool and download tests

## Run locally

    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...    # optional; only the AI Assistant needs it
    streamlit run app.py

## Reproduce the pipeline

    # download sales_train_evaluation.csv and calendar.csv from Kaggle "M5 Forecasting - Accuracy" into data/raw/
    python src/data_preparation.py
    python -m src.train
    python -m pytest -q

Training runs outside the app. The deployed app only loads the saved artifacts.

## Deployment

Streamlit Community Cloud, deployed from this repository (`main`, `app.py`, Python 3.12). The API key is stored only in the
host's Secrets as `ANTHROPIC_API_KEY`; it is not in the code or the repository. The assistant model can be changed with the
`ANTHROPIC_MODEL` environment variable.

## Limitations

- Historical dataset; forecasts are demonstrations, not current predictions.
- Low observed sales may reflect stockouts rather than low demand; the data has no stock records.
- Recursive forecasting lets errors compound across the 28 days; intermittent demand keeps WAPE high.
- The holdout shows overforecasting on average; one 28-day window is a small sample.
- Differences between tuned configurations are small, and tuning was deliberately limited.
- Not optimized: supplier minimums, costs, shelf life, delivery uncertainty, warehouse capacity, lost demand.
- Order suggestions are suggested replenishment quantities, not guarantees.

## Not included in this version

- Price features and validated prediction intervals (listed as optional in the brief)
- Authentication and live company data integration (out of scope for the first version)

## Data attribution

Source data: M5 Forecasting - Accuracy (University of Nicosia / Kaggle), subject to the competition rules. Raw files are not
redistributed here; `data/demo/` contains a small derived subset for demonstration. Check the rules before reusing it.

## Contributors and dates

Divya Teja: design, implementation, evaluation and deployment. Built and deployed November 8, 2026.
