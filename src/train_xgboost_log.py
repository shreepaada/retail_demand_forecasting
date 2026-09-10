import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# Load processed data
# -----------------------------

train_data = pd.read_csv(
    "data/processed_train.csv"
)

validation_data = pd.read_csv(
    "data/processed_validation.csv"
)

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

y_train = np.log1p(train_data["sales"].values)
y_val = validation_data["sales"].values


print("\nFeature matrix:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)


# -----------------------------
# Train XGBoost
# -----------------------------

print("\nTraining XGBoost...")

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    verbose=False
)


# -----------------------------
# Predictions
# -----------------------------

print("Making predictions...")

predictions_log = model.predict(X_val)

# Convert predictions back to original sales scale
predictions = np.expm1(predictions_log)

# Sales cannot be negative
predictions = np.maximum(
    predictions,
    0
)


# -----------------------------
# Metrics
# -----------------------------

mae = mean_absolute_error(
    y_val,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_val,
        predictions
    )
)

rmsle = np.sqrt(
    np.mean(
        (
            np.log1p(y_val)
            - np.log1p(predictions)
        ) ** 2
    )
)


# -----------------------------
# Results
# -----------------------------

print("\nXGBoost Results")
print("----------------")

print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"RMSLE: {rmsle:.4f}")

print("\nModel Comparison")
print("----------------")

print("Seasonal Naive RMSLE:  0.5340")
print("Linear Regression:     1.2039")
print("Random Forest:         0.4197")
print(f"XGBoost:               {rmsle:.4f}")