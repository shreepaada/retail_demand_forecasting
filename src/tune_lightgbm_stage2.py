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


print("\nFeature matrix:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)


# -----------------------------
# Stage 2 parameter combinations
# -----------------------------

param_grid = [
    {
        "learning_rate": 0.03,
        "n_estimators": 800,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0
    },
    {
        "learning_rate": 0.05,
        "n_estimators": 800,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0
    },
    {
        "learning_rate": 0.08,
        "n_estimators": 500,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0
    },
    {
        "learning_rate": 0.03,
        "n_estimators": 800,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0
    },
    {
        "learning_rate": 0.03,
        "n_estimators": 800,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0
    },
    {
        "learning_rate": 0.05,
        "n_estimators": 800,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0
    }
]


# -----------------------------
# RMSLE
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

    print("\n" + "=" * 55)
    print(f"Experiment {i}/{len(param_grid)}")
    print(params)
    print("=" * 55)

    model = LGBMRegressor(
        objective="regression",

        # Best settings from Stage 1
        num_leaves=128,
        max_depth=-1,
        min_child_samples=20,

        # Stage 2 parameters
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],

        subsample=0.8,
        colsample_bytree=0.8,

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
        "learning_rate": params["learning_rate"],
        "n_estimators": params["n_estimators"],
        "reg_alpha": params["reg_alpha"],
        "reg_lambda": params["reg_lambda"],
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
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)


# -----------------------------
# Best configuration
# -----------------------------

best = results_df.iloc[0]

print("\nBest Configuration")
print("=" * 80)

print(f"num_leaves:        128")
print(f"max_depth:         -1")
print(f"min_child_samples: 20")
print(f"learning_rate:     {best['learning_rate']}")
print(f"n_estimators:      {int(best['n_estimators'])}")
print(f"reg_alpha:         {best['reg_alpha']}")
print(f"reg_lambda:        {best['reg_lambda']}")
print(f"RMSLE:             {best['RMSLE']:.4f}")