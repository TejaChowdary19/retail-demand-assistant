"""Prepare M5 data: one store, ~20 products, selected using the training period only."""
import json
from pathlib import Path
import pandas as pd

RAW, OUT = Path("data/raw"), Path("data/demo")
STORE, N_PRODUCTS, HOLDOUT_DAYS, SEED = "CA_1", 20, 28, 42


def select_products(sales: pd.DataFrame, day_cols: list[str]) -> list[str]:
    # Selection window: the 365 days BEFORE the final holdout (no peeking at test period)
    window = sales[day_cols[-(365 + HOLDOUT_DAYS):-HOLDOUT_DAYS]]
    stats = pd.DataFrame({
        "item_id": sales["item_id"].values,
        "mean_sales": window.mean(axis=1).values,
        "zero_frac": (window == 0).mean(axis=1).values,
    })
    stats = stats[(stats.zero_frac <= 0.5) & (stats.mean_sales > 0)]
    stats["tier"] = pd.qcut(stats.mean_sales, 3, labels=["low", "medium", "high"])
    picks = []
    for tier, n in zip(["high", "medium", "low"], [7, 7, 6]):
        picks += stats[stats.tier == tier].sample(n, random_state=SEED).item_id.tolist()
    return picks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    sales = sales[sales.store_id == STORE].reset_index(drop=True)
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    picks = select_products(sales, day_cols)
    sales = sales[sales.item_id.isin(picks)]

    long = sales.melt(
        id_vars=["item_id", "dept_id", "cat_id", "store_id"],
        value_vars=day_cols, var_name="d", value_name="units_sold",
    )
    cal_cols = ["d", "date", "wday", "month", "year", "event_name_1", "event_type_1", "snap_CA"]
    df = long.merge(cal[cal_cols], on="d", how="left")
    df = df.rename(columns={"item_id": "product_id", "dept_id": "department_id", "cat_id": "category_id"})
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    # Validation (zeros are genuine sales records, not missing)
    assert not df.duplicated(["product_id", "store_id", "date"]).any(), "duplicate keys"
    assert df.units_sold.notna().all() and (df.units_sold >= 0).all()
    assert df.date.notna().all()
    expected = df.date.nunique() * len(picks)
    assert len(df) == expected, "missing product-date rows"

    df.drop(columns="d").to_parquet(OUT / "prepared_sales.parquet", index=False)
    meta = {
        "store": STORE, "products": picks, "seed": SEED,
        "date_min": str(df.date.min().date()), "date_max": str(df.date.max().date()),
        "selection": "365 days before final 28-day holdout; zero_frac<=0.5; tertiles of mean sales (7 high/7 medium/6 low)",
    }
    (OUT / "selection.json").write_text(json.dumps(meta, indent=2))
    print(meta, len(df), "rows")


if __name__ == "__main__":
    main()
