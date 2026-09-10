from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import sys
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_next_day


app = Flask(__name__)
CORS(app)


# ==========================================
# Load historical data once when API starts
# ==========================================

TRAIN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "train.csv"
)

print("Loading historical sales data...")

train = pd.read_csv(TRAIN_PATH)

train["date"] = pd.to_datetime(train["date"])

train = train.sort_values("date")

print(f"Historical data loaded: {train.shape}")


# ==========================================
# Root
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Retail Sales Forecasting API",
        "status": "running"
    })


# ==========================================
# Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        store_nbr = int(data["store_nbr"])
        family = data["family"]
        onpromotion = int(data["onpromotion"])

        prediction = predict_next_day(
            store_nbr,
            family,
            onpromotion
        )

        return jsonify({
            "store_nbr": store_nbr,
            "family": family,
            "onpromotion": onpromotion,
            "predicted_sales": round(float(prediction), 2)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# ==========================================
# Historical Sales
# ==========================================

@app.route("/history", methods=["GET"])
def history():

    try:

        store_nbr = int(request.args.get("store_nbr"))
        family = request.args.get("family")

        filtered = train[
            (train["store_nbr"] == store_nbr)
            &
            (train["family"] == family)
        ]

        if filtered.empty:

            return jsonify({
                "error": "No historical data found."
            }), 404

        # Last 30 days
        filtered = filtered.tail(30)

        result = [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "sales": round(float(row["sales"]), 2)
            }
            for _, row in filtered.iterrows()
        ]

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# ==========================================
# Run API
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )