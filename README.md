# 📈 BBNI Stock Price Forecasting using ARIMA and Bidirectional LSTM

This project aims to analyze and forecast the closing stock prices (**Close Price**) of PT Bank Negara Indonesia Tbk (**BBNI.JK**) for 7 consecutive trading days. It compares the performance of a classical statistical time-series model, **ARIMA (AutoRegressive Integrated Moving Average)**, against a deep learning architecture, **Bidirectional LSTM (BiLSTM)**.

---

## 📌 Table of Contents
- [Background](#-background)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Evaluation Metrics](#-evaluation-metrics)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Usage](#-usage)

---

## 📊 Dataset
- **Data Source:** Yahoo Finance (`yfinance`)
- **Ticker Symbol:** `BBNI.JK`
- **Timeframe:** September 20, 2021 – September 20, 2024
- **Primary Features:** `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
- **Target Variable:** `Close` (Daily Closing Price)

---

## 🏗️ Methodology

### 1. Data Preprocessing
* Handled MultiIndex column structures returned by `yfinance`.
* Missing value inspection and clean-up.
* Train-Test split with an **80% Training** and **20% Testing** ratio.
* Feature scaling using `MinMaxScaler` $[0, 1]$ for the LSTM network.
* Restructuring time-series sequence into a supervised learning format ($t-1 \rightarrow t$).

### 2. Model 1: ARIMA
* Utilized `pmdarima.auto_arima` to search for optimal hyperparameter combinations $(p, d, q)$ based on the lowest AIC score.
* Stationarity testing using the **Augmented Dickey-Fuller (ADF) Test**.
* Autocorrelation analysis via **ACF** and **PACF** plots.

### 3. Model 2: Bidirectional LSTM
Deep Learning architecture built with TensorFlow / Keras:
* **Layer 1:** Bidirectional LSTM (128 units, `return_sequences=True`) + Dropout (0.3)
* **Layer 2:** Bidirectional LSTM (64 units, `return_sequences=True`) + Dropout (0.3)
* **Layer 3:** Bidirectional LSTM (32 units) + Dropout (0.3)
* **Layer 4:** Dense (16 units, ReLU activation)
* **Output:** Dense (1 unit)
* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (MSE)

---

## 📏 Evaluation Metrics
Model performance is evaluated using two primary metrics over the 7-day forecast horizon:
1. **RMSE (Root Mean Squared Error):** Measures the magnitude of prediction errors.
2. **MAPE (Mean Absolute Percentage Error):** Measures the average percentage deviation of predictions from actual prices.

---

## 🛠️ Prerequisites & Installation

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install yfinance pandas numpy scikit-learn tensorflow statsmodels pmdarima matplotlib seaborn
