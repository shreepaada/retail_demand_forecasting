# Retail Sales Intelligence

An end-to-end machine learning application for forecasting next-day sales across retail stores and product families.

The system uses historical sales patterns, promotions, calendar information, lag features, and rolling statistics to predict the number of units expected to be sold on the following day. A LightGBM regression model serves predictions through a Flask REST API, while a React + TypeScript dashboard provides an interactive interface for forecasts and historical sales analysis.

---

## Overview

Retail businesses need accurate sales forecasts to plan inventory, understand demand patterns, and make better operational decisions.

This project builds a machine learning pipeline that predicts next-day sales for a specific:

- Store
- Product family
- Promotion level

The project covers the complete ML workflow:

**Data → EDA → Feature Engineering → Model Training → Evaluation → Model Selection → API → Interactive Dashboard**

---

## Key Features

- Next-day sales prediction for individual stores and product families
- Historical sales visualization
- Promotion-based forecasting
- Time-series feature engineering
- Comparison of multiple machine learning models
- LightGBM hyperparameter tuning
- Flask REST API for model inference
- React + TypeScript frontend
- Interactive sales dashboard
- Model performance information

---

## Dataset

The project is based on the **Store Sales - Time Series Forecasting** dataset from Kaggle.

The dataset contains daily sales information across multiple stores and product families.

### Main data

| Feature | Description |
|---|---|
| `date` | Date of the observation |
| `store_nbr` | Store identifier |
| `family` | Product family |
| `sales` | Number of units sold |
| `onpromotion` | Number of products on promotion |

Additional datasets provide information about stores, holidays, oil prices, and transactions.

---

## Exploratory Data Analysis

The dataset was analyzed to identify important sales patterns, including:

- Long-term sales trends
- Weekly seasonality
- Monthly sales patterns
- Store-level variation
- Product-family sales distribution
- Day-of-week effects
- Relationship between promotions and sales

The analysis showed strong temporal patterns and significant differences between stores and product families.

---

## Feature Engineering

Time-series features were created separately for each **store-product-family** combination.

### Calendar Features

- Year
- Month
- Day
- Day of week
- Quarter
- Week of year

### Lag Features

Historical sales were used to capture previous sales behavior:

- 1-day lag
- 7-day lag
- 14-day lag
- 28-day lag
- 365-day lag

### Rolling Features

Rolling statistics were generated using historical observations:

- 7-day rolling mean
- 28-day rolling mean
- 7-day rolling standard deviation

Lag and rolling features were shifted before calculation to prevent target leakage.

---

## Model Development

Several models were evaluated:

| Model | RMSLE |
|---|---:|
| Seasonal Naive | 0.5340 |
| Linear Regression | 1.2039 |
| Random Forest | 0.3898 |
| XGBoost | 0.3838 |
| LightGBM | 0.3804 |
| **Tuned LightGBM** | **0.3763** |

The final LightGBM model achieved the best validation performance.

### Final Model

- Algorithm: LightGBM Regressor
- Target transformation: `log1p(sales)`
- Number of estimators: 1,500
- Number of leaves: 128
- Learning rate: 0.04
- Minimum child samples: 20
- L1 regularization: 0.1
- L2 regularization: 1.0

### Validation Performance

**RMSLE: 0.3763**

**RMSE: 219.57**

A chronological validation split was used rather than randomly shuffling the time-series data.

---

## Feature Importance

The model identified several important predictors, including:

- Day
- Day of week
- Week of year
- Store number
- Previous-day sales
- 7-day rolling average
- 28-day rolling average
- 7-day rolling standard deviation
- Month
- Previous-week sales
- Promotion count

This indicates that recent sales behavior and temporal patterns are particularly useful for predicting future sales.

---

## System Architecture

```text
                    ┌─────────────────────┐
                    │   Historical Data   │
                    │      CSV Files      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │                     │
                    │ • Calendar Features │
                    │ • Lag Features      │
                    │ • Rolling Features  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LightGBM Model    │
                    │                     │
                    │   RMSLE: 0.3763     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Flask API       │
                    │                     │
                    │ POST /predict       │
                    │ GET  /history       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ React + TypeScript   │
                    │     Dashboard       │
                    └─────────────────────┘