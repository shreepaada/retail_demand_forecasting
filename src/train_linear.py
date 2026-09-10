import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# Load and prepare data
# -----------------------------

train = pd.read_csv("data/train.csv")

train["date"] = pd.to_datetime(train["date"])

train = train.sort_values(
    ["store_nbr", "family", "date"]
)

group_cols = ["store_nbr", "family"]


# -----------------------------
# Calendar features
# -----------------------------

train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["day_of_week"] = train["date"].dt.day_of_week
train["quarter"] = train["date"].dt.quarter
train["week_of_year"] = (
    train["date"].dt.isocalendar().week.astype(int)
)


# -----------------------------
# Lag features
# -----------------------------

for lag in [1, 7, 14, 28, 365]:
    train[f"sales_lag_{lag}"] = (
        train.groupby(group_cols)["sales"]
        .shift(lag)
    )


# -----------------------------
# Rolling features
# -----------------------------

train["sales_rolling_mean_7"] = (
    train.groupby(group_cols)["sales"]
    .transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)

train["sales_rolling_mean_28"] = (
    train.groupby(group_cols)["sales"]
    .transform(
        lambda x: x.shift(1).rolling(28).mean()
    )
)

train["sales_rolling_std_7"] = (
    train.groupby(group_cols)["sales"]
    .transform(
        lambda x: x.shift(1).rolling(7).std()
    )
)


# -----------------------------
# Remove missing values
# -----------------------------

feature_cols = [
    "store_nbr",
    "family",
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

train = train.dropna(
    subset=feature_cols + ["sales"]
)


# -----------------------------
# Train / validation split
# -----------------------------

validation_start = (
    train["date"].max()
    - pd.Timedelta(days=90)
)

train_data = train[
    train["date"] < validation_start
]

validation_data = train[
    train["date"] >= validation_start
]


# -----------------------------
# Separate X and y
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


# Combine categorical + numeric
from scipy.sparse import hstack

X_train = hstack([
    X_train_num,
    X_train_cat
])

X_val = hstack([
    X_val_num,
    X_val_cat
])

y_train = train_data["sales"].values
y_val = validation_data["sales"].values


# -----------------------------
# Train Linear Regression
# -----------------------------

print("Training Linear Regression...")

model = LinearRegression()

model.fit(X_train, y_train)


# -----------------------------
# Predictions
# -----------------------------

predictions = model.predict(X_val)

# Sales cannot be negative
predictions = np.maximum(predictions, 0)


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

print("\nLinear Regression Results")
print("-------------------------")

print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"RMSLE: {rmsle:.4f}")

print("\nBaseline RMSLE: 0.5340")
print(f"Linear Regression RMSLE: {rmsle:.4f}")