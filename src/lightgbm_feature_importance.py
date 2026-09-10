import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack


# -----------------------------
# Load processed data
# -----------------------------

train_data = pd.read_csv(
    "data/processed_train.csv"
)

validation_data = pd.read_csv(
    "data/processed_validation.csv"
)


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

X_train_num = train_data[numeric_cols].values

X_train = hstack([
    X_train_num,
    X_train_cat
])


# -----------------------------
# Target
# -----------------------------

y_train = np.log1p(
    train_data["sales"].values
)


# -----------------------------
# Train LightGBM
# -----------------------------

print("Training LightGBM...")

model = LGBMRegressor(
    objective="regression",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    verbosity=-1
)

model.fit(
    X_train,
    y_train
)


# -----------------------------
# Feature names
# -----------------------------

categorical_feature_names = encoder.get_feature_names_out(
    categorical_cols
)

feature_names = list(numeric_cols) + list(
    categorical_feature_names
)


# -----------------------------
# Feature importance
# -----------------------------

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)


# -----------------------------
# Display
# -----------------------------

print("\nTop 20 Features")
print("----------------")

print(
    importance_df.head(20).to_string(index=False)
)