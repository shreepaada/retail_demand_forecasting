import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.metrics import mean_squared_error


# -----------------------------
# Load processed data
# -----------------------------

train_data = pd.read_csv("data/processed_train.csv")
validation_data = pd.read_csv("data/processed_validation.csv")

print("Training rows:", len(train_data))
print("Validation rows:", len(validation_data))


# -----------------------------
# Features
# -----------------------------

categorical_cols = ["family"]

numeric_cols = [
    "store_nbr",
    "onpromotion",
    "year",
    "month",
    "day",
    "day_of_week",
    "quarter",
    "week_of_year",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_lag_365",
    "sales_rolling_mean_7",
    "sales_rolling_mean_28",
    "sales_rolling_std_7"
]


# -----------------------------
# Encode family
# -----------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)

X_train_cat = encoder.fit_transform(
    train_data[categorical_cols]
)

X_val_cat = encoder.transform(
    validation_data[categorical_cols]
)


# -----------------------------
# Numeric features
# -----------------------------

X_train_num = train_data[numeric_cols].values
X_val_num = validation_data[numeric_cols].values


# -----------------------------
# Combine features
# -----------------------------

X_train = hstack([
    X_train_num,
    X_train_cat
])

X_val = hstack([
    X_val_num,
    X_val_cat
])


# -----------------------------
# Target
# -----------------------------

y_train = np.log1p(
    train_data["sales"].values
)

y_val = validation_data["sales"].values


# -----------------------------
# Parameter combinations
# -----------------------------

param_grid = [
    {
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20
    },
    {
        "num_leaves": 64,
        "max_depth": -1,
        "min_child_samples": 20
    },
    {
        "num_leaves": 128,
        "max_depth": -1,
        "min_child_samples": 20
    },
    {
        "num_leaves": 64,
        "max_depth": 12,
        "min_child_samples": 20
    },
    {
        "num_leaves": 64,
        "max_depth": -1,
        "min_child_samples": 50
    },
    {
        "num_leaves": 128,
        "max_depth": -1,
        "min_child_samples": 50
    }
]


# -----------------------------
# RMSLE function
# -----------------------------

def calculate_rmsle(actual, predicted):

    predicted = np.maximum(
        predicted,
        0
    )

    return np.sqrt(
        np.mean(
            (
                np.log1p(actual)
                - np.log1p(predicted)
            ) ** 2
        )
    )


# -----------------------------
# Run experiments
# -----------------------------

results = []

for i, params in enumerate(param_grid, start=1):

    print("\n" + "=" * 50)
    print(f"Experiment {i}/{len(param_grid)}")
    print(params)
    print("=" * 50)

    model = LGBMRegressor(
        objective="regression",

        n_estimators=500,
        learning_rate=0.05,

        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        min_child_samples=params["min_child_samples"],

        subsample=0.8,
        colsample_bytree=0.8,

        reg_alpha=0.1,
        reg_lambda=1.0,

        n_jobs=-1,
        random_state=42,

        verbosity=-1
    )

    print("Training...")

    model.fit(
        X_train,
        y_train
    )

    print("Predicting...")

    predictions_log = model.predict(
        X_val
    )

    predictions = np.expm1(
        predictions_log
    )

    predictions = np.maximum(
        predictions,
        0
    )

    rmsle = calculate_rmsle(
        y_val,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            predictions
        )
    )

    print(f"RMSLE: {rmsle:.4f}")
    print(f"RMSE:  {rmse:.4f}")

    results.append({
        "experiment": i,
        "num_leaves": params["num_leaves"],
        "max_depth": params["max_depth"],
        "min_child_samples": params["min_child_samples"],
        "RMSLE": rmsle,
        "RMSE": rmse
    })


# -----------------------------
# Results
# -----------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSLE"
)

print("\n\nFinal Results")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nBest Parameters")
print("=" * 70)

best = results_df.iloc[0]

print(f"num_leaves:        {int(best['num_leaves'])}")
print(f"max_depth:         {int(best['max_depth'])}")
print(f"min_child_samples: {int(best['min_child_samples'])}")
print(f"RMSLE:             {best['RMSLE']:.4f}")