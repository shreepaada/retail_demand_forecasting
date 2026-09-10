import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib


# =========================
# 1. Load model and encoder
# =========================

MODEL_PATH = "models/lightgbm_final.txt"
ENCODER_PATH = "models/family_encoder.pkl"
FEATURES_PATH = "models/feature_names.pkl"

model = lgb.Booster(model_file=MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURES_PATH)


# =========================
# 2. Load historical data
# =========================

train = pd.read_csv("data/train.csv")

train["date"] = pd.to_datetime(train["date"])

train = train.sort_values(
    ["store_nbr", "family", "date"]
).reset_index(drop=True)


# =========================
# 3. Feature creation
# =========================

def create_features(df):

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    # Calendar features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = (
        df["date"].dt.isocalendar().week.astype(int)
    )

    group_cols = ["store_nbr", "family"]

    # Lag features
    for lag in [1, 7, 14, 28, 365]:
        df[f"sales_lag_{lag}"] = (
            df.groupby(group_cols)["sales"]
            .shift(lag)
        )

    # Rolling features
    # shift(1) ensures today's sales are NOT used
    # to calculate today's prediction features.

    shifted_sales = (
        df.groupby(group_cols)["sales"]
        .shift(1)
    )

    df["sales_rolling_mean_7"] = (
        shifted_sales
        .groupby(
            [df["store_nbr"], df["family"]]
        )
        .transform(
            lambda x: x.rolling(7).mean()
        )
    )

    df["sales_rolling_mean_28"] = (
        shifted_sales
        .groupby(
            [df["store_nbr"], df["family"]]
        )
        .transform(
            lambda x: x.rolling(28).mean()
        )
    )

    df["sales_rolling_std_7"] = (
        shifted_sales
        .groupby(
            [df["store_nbr"], df["family"]]
        )
        .transform(
            lambda x: x.rolling(7).std()
        )
    )

    return df

# =========================
# 4. Predict next day
# =========================

def predict_next_day(store_nbr, family, onpromotion):

    # Last date available in training data
    last_date = train["date"].max()

    prediction_date = last_date + pd.Timedelta(days=1)

    # Historical data for the requested store/family
    history = train[
        (train["store_nbr"] == store_nbr) &
        (train["family"] == family)
    ].copy()

    if history.empty:
        raise ValueError(
            "No historical data found for this store/family combination."
        )

    # Add the future row
    future_row = pd.DataFrame({
        "id": [0],
        "date": [prediction_date],
        "store_nbr": [store_nbr],
        "family": [family],
        "sales": [np.nan],
        "onpromotion": [onpromotion]
    })

    combined = pd.concat(
        [history, future_row],
        ignore_index=True
    )

    combined = combined.sort_values("date").reset_index(drop=True)

    # Create features
    combined = create_features(combined)

    # Select the future row
    future = combined[
        combined["date"] == prediction_date
    ].copy()

    # Remove target
    X = future.drop(columns=["sales", "date"])

    # Encode family
    family_encoded = encoder.transform(
        X[["family"]]
    )

    family_names = encoder.get_feature_names_out(
        ["family"]
    )

    # Sanitize feature names exactly like training
    family_names = [
        name.replace(":", "_")
            .replace("{", "_")
            .replace("}", "_")
            .replace("[", "_")
            .replace("]", "_")
            .replace('"', "_")
            .replace("'", "_")
            .replace(",", "_")
        for name in family_names
    ]

    family_encoded_df = pd.DataFrame.sparse.from_spmatrix(
        family_encoded,
        columns=family_names,
        index=X.index
    )

    X = X.drop(columns=["family"])

    X = pd.concat(
        [X, family_encoded_df],
        axis=1
    )

    # Make sure feature order is identical to training
    X = X.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Predict log(sales + 1)
    prediction_log = model.predict(X)

    # Convert back to original sales scale
    prediction = np.expm1(prediction_log)

    prediction = max(0, float(prediction[0]))

    return prediction


# =========================
# 5. Test prediction
# =========================

if __name__ == "__main__":

    prediction = predict_next_day(
        store_nbr=1,
        family="GROCERY I",
        onpromotion=10
    )

    print("\nPrediction:")
    print(f"Next-day sales: {prediction:.2f}")