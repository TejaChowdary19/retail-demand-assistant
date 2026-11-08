from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO, MODELS, REPORTS = ROOT / "data" / "demo", ROOT / "models", ROOT / "reports"
HORIZON = 28
LAGS = [1, 7, 14, 28]
ROLL_MEANS = [7, 14, 28]
ROLL_STDS = [7, 28]

STATIC = ["product_code", "dept_code", "cat_code"]
CALENDAR = ["wday", "month", "is_weekend", "has_event", "snap_CA"]
FEATURES = (STATIC + CALENDAR + [f"lag_{k}" for k in LAGS]
            + [f"roll_mean_{w}" for w in ROLL_MEANS] + [f"roll_std_{w}" for w in ROLL_STDS])
CATEGORICAL = STATIC

BASE_PARAMS = dict(n_estimators=300, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, random_state=42, verbose=-1)
GRID = [  # small, documented search
    dict(objective="regression", num_leaves=15, min_child_samples=50),
    dict(objective="regression", num_leaves=31, min_child_samples=20),
    dict(objective="poisson", num_leaves=15, min_child_samples=50),
    dict(objective="poisson", num_leaves=31, min_child_samples=20),
]
