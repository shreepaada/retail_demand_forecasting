import pandas as pd
import numpy as np

# -----------------------------
# Load data
# -----------------------------

train = pd.read_csv("data/train.csv")

train["date"] = pd.to_datetime(train["date"])

# Sort each store-family time series
train = train.sort_values(
    ["store_nbr", "family", "date"]
)

# -----------------------------
# Create baseline prediction
# -----------------------------

train["baseline_prediction"] = (
    train.groupby(["store_nbr", "family"])["sales"]
    .shift(7)
)

# -----------------------------
# Validation split
# -----------------------------

validation_start = train["date"].max() - pd.Timedelta(days=90)

validation = train[
    train["date"] >= validation_start
].copy()

# Remove rows where prediction is unavailable
validation = validation.dropna(
    subset=["baseline_prediction"]
)

y_true = validation["sales"].values
y_pred = validation["baseline_prediction"].values

# -----------------------------
# Metrics
# -----------------------------

mae = np.mean(np.abs(y_true - y_pred))

rmse = np.sqrt(
    np.mean((y_true - y_pred) ** 2)
)

# RMSLE
y_pred = np.maximum(y_pred, 0)

rmsle = np.sqrt(
    np.mean(
        (
            np.log1p(y_true)
            - np.log1p(y_pred)
        ) ** 2
    )
)

print("Seasonal Naive Baseline")
print("-----------------------")

print("Validation period:")
print(
    validation["date"].min(),
    "to",
    validation["date"].max()
)

print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"RMSLE: {rmsle:.4f}")