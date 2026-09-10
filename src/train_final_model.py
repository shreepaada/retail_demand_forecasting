import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder


# =========================
# 1. Load processed data
# =========================

train_path = "data/processed_train.csv"
validation_path = "data/processed_validation.csv"

train_df = pd.read_csv(train_path)
validation_df = pd.read_csv(validation_path)

print("Train shape:", train_df.shape)
print("Validation shape:", validation_df.shape)


# =========================
# 2. Combine train + validation
# =========================

df = pd.concat(
    [train_df, validation_df],
    ignore_index=True
)

print("Combined shape:", df.shape)


# =========================
# 3. Define target
# =========================

target = "sales"

X = df.drop(columns=[target, "date"])
y = df[target]

# =========================
# 4. Handle categorical column
# =========================

categorical_cols = ["family"]

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)

# Fit encoder AND transform the family column
family_encoded = encoder.fit_transform(
    X[categorical_cols]
)

# Get feature names AFTER fitting the encoder
family_names = encoder.get_feature_names_out(
    categorical_cols
)

# Remove characters that LightGBM does not allow
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

# Remove original family column
X = X.drop(columns=categorical_cols)

# Add encoded family columns
X = pd.concat(
    [X, family_encoded_df],
    axis=1
)

# =========================
# 5. Prepare target
# =========================

# Log transformation because RMSLE was our evaluation metric
y_log = pd.Series(
    __import__("numpy").log1p(y),
    index=y.index
)


# =========================
# 6. Train final LightGBM
# =========================

model = lgb.LGBMRegressor(
    objective="regression",

    num_leaves=128,
    max_depth=-1,
    min_child_samples=20,

    learning_rate=0.04,
    n_estimators=1500,

    reg_alpha=0.1,
    reg_lambda=1.0,

    subsample=0.8,
    colsample_bytree=0.8,

    n_jobs=-1,
    random_state=42,
    verbosity=-1
)

print("\nTraining final LightGBM model...")

model.fit(X, y_log)

print("Training completed.")


# =========================
# 7. Create models directory
# =========================

os.makedirs("models", exist_ok=True)


# =========================
# 8. Save model
# =========================

model_path = "models/lightgbm_final.txt"

model.booster_.save_model(model_path)

print(f"Model saved to: {model_path}")


# =========================
# 9. Save encoder
# =========================

encoder_path = "models/family_encoder.pkl"

joblib.dump(
    encoder,
    encoder_path
)

print(f"Encoder saved to: {encoder_path}")


# =========================
# 10. Save feature names
# =========================

feature_names_path = "models/feature_names.pkl"

joblib.dump(
    list(X.columns),
    feature_names_path
)

print(f"Feature names saved to: {feature_names_path}")


print("\nFinal model setup complete!")