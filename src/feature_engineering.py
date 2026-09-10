import pandas as pd

# load data

train = pd.read_csv("data/train.csv")

#covert date 
train["date"] = pd.to_datetime(train["date"])

# sort by each store product series

train = train.sort_values(
    ["store_nbr","family","date"]
)

# calendar feature

train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["day_of_week"] = train["date"].dt.day_of_week
train["quarter"] = train["date"].dt.quarter #wchih week of the year does this date belong to
train["week_of_year"] = train["date"].dt.isocalendar().week.astype(int)


print(train.head())
print("\nColumns:")
print(train.columns.tolist())
# -----------------------------
# Lag features
# -----------------------------

group_cols = ["store_nbr", "family"]

train["sales_lag_1"] = (
    train.groupby(group_cols)["sales"]
    .shift(1)
)

train["sales_lag_7"] = (
    train.groupby(group_cols)["sales"]
    .shift(7)
)

train["sales_lag_14"] = (
    train.groupby(group_cols)["sales"]
    .shift(14)
)

train["sales_lag_28"] = (
    train.groupby(group_cols)["sales"]
    .shift(28)
)
print("\nLag features:")
print(
    train[
        [
            "date",
            "store_nbr",
            "family",
            "sales",
            "sales_lag_1",
            "sales_lag_7",
            "sales_lag_14",
            "sales_lag_28"
        ]
    ].head(35)
)
# -----------------------------
# Rolling features
# -----------------------------

train["sales_rolling_mean_7"] = (
    train.groupby(group_cols)["sales"]
    .transform(lambda x: x.shift(1).rolling(7).mean())
)

train["sales_rolling_mean_28"] = (
    train.groupby(group_cols)["sales"]
    .transform(lambda x: x.shift(1).rolling(28).mean())
)

# -----------------------------
# Yearly lag
# -----------------------------

train["sales_lag_365"] = (
    train.groupby(group_cols)["sales"]
    .shift(365)
)

# -----------------------------
# Rolling volatility
# -----------------------------

train["sales_rolling_std_7"] = (
    train.groupby(group_cols)["sales"]
    .transform(lambda x: x.shift(1).rolling(7).std())
)

# -----------------------------
# Remove rows with missing features
# -----------------------------

feature_cols = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_mean_28",
    "sales_lag_365",
    "sales_rolling_std_7"
]

train = train.dropna(subset=feature_cols)

print("\nShape after removing NaNs:")
print(train.shape)

# -----------------------------
# Train / Validation Split
# -----------------------------

validation_start = train["date"].max() - pd.Timedelta(days=90)

train_data = train[train["date"] < validation_start]
validation_data = train[train["date"] >= validation_start]

print("\nTrain period:")
print(train_data["date"].min(), "to", train_data["date"].max())

print("\nValidation period:")
print(validation_data["date"].min(), "to", validation_data["date"].max())

print("\nTrain shape:", train_data.shape)
print("Validation shape:", validation_data.shape)


# -----------------------------
# Save processed datasets
# -----------------------------

train_data.to_csv(
    "data/processed_train.csv",
    index=False
)

validation_data.to_csv(
    "data/processed_validation.csv",
    index=False
)

print("\nProcessed datasets saved.")
print("Training file: data/processed_train.csv")
print("Validation file: data/processed_validation.csv")