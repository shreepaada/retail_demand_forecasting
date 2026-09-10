import { useEffect, useState } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import "./App.css";

const families = [
  "AUTOMOTIVE",
  "BABY CARE",
  "BEAUTY",
  "BEVERAGES",
  "BOOKS",
  "BREAD/BAKERY",
  "CELEBRATION",
  "CLEANING",
  "DAIRY",
  "DELI",
  "EGGS",
  "FROZEN FOODS",
  "GROCERY I",
  "GROCERY II",
  "HARDWARE",
  "HOME AND KITCHEN I",
  "HOME AND KITCHEN II",
  "HOME APPLIANCES",
  "HOME CARE",
  "LADIESWEAR",
  "LAWN AND GARDEN",
  "LINGERIE",
  "LIQUOR,WINE,BEER",
  "MAGAZINES",
  "MEATS",
  "PERSONAL CARE",
  "PET SUPPLIES",
  "PLAYERS AND ELECTRONICS",
  "POULTRY",
  "PREPARED FOODS",
  "PRODUCE",
  "SCHOOL AND OFFICE SUPPLIES",
  "SEAFOOD",
];

interface HistoryPoint {
  date: string;
  sales: number;
}

function App() {
  const [store, setStore] = useState<number>(1);
  const [family, setFamily] = useState<string>("GROCERY I");
  const [onPromotion, setOnPromotion] = useState<number>(10);

  const [prediction, setPrediction] = useState<number | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);

  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchHistory();
  }, [store, family]);

  const fetchHistory = async () => {
    try {
      setHistoryLoading(true);
      setError("");

      const response = await axios.get("http://127.0.0.1:5000/history", {
        params: {
          store_nbr: store,
          family: family,
        },
      });

      setHistory(response.data);
    } catch (err) {
      console.error(err);
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  const predictSales = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        {
          store_nbr: store,
          family: family,
          onpromotion: onPromotion,
        }
      );

      setPrediction(Number(response.data.predicted_sales));
    } catch (err) {
      console.error(err);
      setError("Unable to generate prediction. Make sure the Flask API is running.");
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const recentAverage =
    history.length > 0
      ? history.reduce((sum, item) => sum + item.sales, 0) / history.length
      : 0;

  const last7Days = history.slice(-7);

  const last7Average =
    last7Days.length > 0
      ? last7Days.reduce((sum, item) => sum + item.sales, 0) /
        last7Days.length
      : 0;

  const changeVsRecent =
    prediction !== null && last7Average > 0
      ? ((prediction - last7Average) / last7Average) * 100
      : null;

  const formattedPrediction =
    prediction !== null
      ? Math.round(prediction).toLocaleString()
      : "—";

  return (
    <div className="app">

      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div>
            <div className="eyebrow">MACHINE LEARNING • RETAIL ANALYTICS</div>

            <h1>Retail Sales Intelligence</h1>

            <p>
              Forecast next-day sales for a selected store and product family
              using historical sales patterns, promotions, and time-based
              features.
            </p>
          </div>
        </div>
      </header>

      <main className="dashboard">

        {/* Prediction Controls */}
        <section className="card prediction-controls">
          <div className="section-heading">
            <div>
              <span className="section-number">01</span>
              <h2>Sales Forecast</h2>
            </div>

            <span className="live-badge">MODEL READY</span>
          </div>

          <p className="description">
            Select a store and product family to estimate the number of units
            expected to be sold on the next day.
          </p>

          <div className="form-grid">

            <div className="field">
              <label htmlFor="store">Store Number</label>

              <input
                id="store"
                type="number"
                min="1"
                max="54"
                value={store}
                onChange={(e) => setStore(Number(e.target.value))}
              />

              <span className="field-help">
                Select a store from 1 to 54
              </span>
            </div>

            <div className="field">
              <label htmlFor="family">Product Family</label>

              <select
                id="family"
                value={family}
                onChange={(e) => setFamily(e.target.value)}
              >
                {families.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>

              <span className="field-help">
                Product category sold by the store
              </span>
            </div>

            <div className="field">
              <label htmlFor="promotion">Products on Promotion</label>

              <input
                id="promotion"
                type="number"
                min="0"
                value={onPromotion}
                onChange={(e) => setOnPromotion(Number(e.target.value))}
              />

              <span className="field-help">
                Number of products currently on promotion
              </span>
            </div>

          </div>

          <button
            className="predict-button"
            onClick={predictSales}
            disabled={loading}
          >
            {loading ? "Generating Forecast..." : "Predict Next-Day Sales"}
          </button>

          {error && <div className="error">{error}</div>}
        </section>


        {/* Prediction Result */}
        <section className="result-card">

          <div className="result-top">
            <div>
              <span className="section-number">02</span>
              <h2>Prediction Result</h2>
            </div>

            <div className="prediction-status">
              {prediction !== null ? "FORECAST GENERATED" : "AWAITING INPUT"}
            </div>
          </div>

          {prediction !== null ? (
            <>
              <div className="result-label">
                Expected Sales Tomorrow
              </div>

              <div className="prediction">
                {formattedPrediction}
              </div>

              <div className="unit">
                units expected to be sold
              </div>

              <div className="prediction-meta">
                <div className="meta-item">
                  <span>Store</span>
                  <strong>{store}</strong>
                </div>

                <div className="meta-item">
                  <span>Product Family</span>
                  <strong>{family}</strong>
                </div>

                <div className="meta-item">
                  <span>Promotion</span>
                  <strong>{onPromotion} products</strong>
                </div>
              </div>

              <div className="comparison">
                <div>
                  <span>Last 7-Day Average</span>
                  <strong>
                    {Math.round(last7Average).toLocaleString()} units
                  </strong>
                </div>

                {changeVsRecent !== null && (
                  <div className="comparison-change">
                    <span>vs. Recent Average</span>

                    <strong
                      className={
                        changeVsRecent >= 0 ? "positive" : "negative"
                      }
                    >
                      {changeVsRecent >= 0 ? "+" : ""}
                      {changeVsRecent.toFixed(1)}%
                    </strong>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="empty-result">
              <div className="empty-icon">↗</div>

              <h3>Ready to forecast</h3>

              <p>
                Choose a store, product family, and promotion level, then
                generate a next-day sales prediction.
              </p>
            </div>
          )}

        </section>


        {/* Historical Chart */}
        <section className="chart-card">

          <div className="chart-header">
            <div>
              <span className="section-number">03</span>

              <h2>Historical Sales</h2>

              <p>
                Last 30 days of recorded sales for Store {store} · {family}
              </p>
            </div>

            {history.length > 0 && (
              <div className="chart-stat">
                <span>30-Day Average</span>
                <strong>
                  {Math.round(recentAverage).toLocaleString()}
                </strong>
              </div>
            )}
          </div>

          <div className="chart-container">
            {historyLoading ? (
              <div className="chart-empty">
                Loading historical sales...
              </div>
            ) : history.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={history}
                  margin={{
                    top: 10,
                    right: 20,
                    left: 0,
                    bottom: 10,
                  }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#e5e7eb"
                  />

                  <XAxis
                    dataKey="date"
                    tickFormatter={(value) => {
                      const date = new Date(String(value));

                      return date.toLocaleDateString("en-US", {
                        month: "short",
                        day: "numeric",
                      });
                    }}
                    tick={{ fill: "#000000", fontSize: 12 }}
                  />

                  <YAxis
                    tick={{ fill: "#000000", fontSize: 12 }}
                    tickFormatter={(value) =>
                      Number(value).toLocaleString()
                    }
                  />

                  <Tooltip
                    labelFormatter={(value) => {
                      const date = new Date(String(value));

                      return date.toLocaleDateString("en-US", {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                      });
                    }}
                    formatter={(value) => [
                      `${Number(value as number).toLocaleString()} units`,
                      "Sales",
                    ]}
                  />

                  <Line
                    type="monotone"
                    dataKey="sales"
                    stroke="#172033"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="chart-empty">
                No historical data available for this selection.
              </div>
            )}
          </div>
        </section>


        {/* Model Information */}
        <section className="model-card">

          <div className="model-heading">
            <div>
              <span className="section-number">04</span>
              <h2>Model Information</h2>
            </div>

            <span className="model-name">LightGBM</span>
          </div>

          <div className="model-grid">

            <div className="model-item">
              <span>Validation RMSLE</span>
              <strong>0.3763</strong>
            </div>

            <div className="model-item">
              <span>Validation RMSE</span>
              <strong>219.57</strong>
            </div>

            <div className="model-item">
              <span>Estimators</span>
              <strong>1,500</strong>
            </div>

            <div className="model-item">
              <span>Number of Leaves</span>
              <strong>128</strong>
            </div>

          </div>

          <div className="model-description">
            <strong>Features used</strong>

            <p>
              Calendar features, store and product family information,
              promotion counts, historical lag values, and rolling sales
              statistics.
            </p>
          </div>

        </section>

      </main>

      <footer className="footer">
        Retail Sales Intelligence · LightGBM Forecasting System
      </footer>

    </div>
  );
}

export default App;