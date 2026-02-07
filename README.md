# Financial Operations Analytics: E-commerce Revenue & Churn

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Production-green)

## 📌 Project Overview

In the competitive e-commerce landscape, accurate financial forecasting and customer retention are paramount. This project analyzes a complex, multi-table dataset (inspired by the Brazilian E-Commerce Public Dataset by Olist) to predict future revenue and identify customers at risk of churn.

## 🚀 Key Features

- **Complex Data Simulation**: Generates a relational schema with `Orders`, `Order Items`, `Payments`, `Customers`, and `Sellers`.
- **Revenue Forecasting**:
  - Aggregates daily revenue from transactional data.
  - Engineers time-series features (Lags, Rolling Means).
  - Uses **XGBoost Regressor** to forecast 30-day future revenue.
- **Churn Prediction**:
  - defines Churn based on extended inactivity (>90 days).
  - Builds a customer profile (Recency, Frequency, Monetary).
  - Uses **Random Forest Classifier** to predict churn probability.

## 🛠️ Tech Stack

## 🛠️ Tech Stack & Tools

Integrating financial rigor with advanced data science:

### Programming & Algorithms

- **Python**: Financial modeling and forecasting.
- **Scikit-learn & XGBoost**: Churn prediction and revenue forecasting.
- **Matplotlib**: Revenue trend visualization.
- **R Programming**: Time-series decomposition and statistical tests.

### Enterprise Reporting

- **PowerBI**: Financial performance dashboards (Revenue, Churn Rate).
- **Tableau**: Interactive financial storytelling.
- **Qlik Sense**: Data discovery for financial operations.
- **Excel**: Financial modeling and budget planning.

### Data Engineering

- **SQL**: relational database management and complex joins.
- **Google BigQuery**: Cloud data warehouse for historical financial data.
- **Apache**: Batch processing of daily financial records.
- **Talend**: Enterprise grade ETL for financial data.

### Analytics Ecosystem

- **Google Analytics**: Conversion funnel analysis.
- **SAS**: Risk management and compliance analysis.
- **Splunk**: Security and fraud detection logs.

## 📂 Project Structure

```
financial-analytics/
├── data_generator.py     # Generates Olist-style relational data
├── revenue_forecast.py   # Forecasting model (XGBoost)
├── churn_prediction.py   # Churn classification (Random Forest)
├── orders.csv            # (Generated) Orders table
├── order_items.csv       # (Generated) Items table
├── customers.csv         # (Generated) Customers table
├── revenue_forecast.png  # (Generated) Forecast plot
└── README.md             # Project documentation
```

## 📊 Methodology

1. **Relational Data Generation**: We simulate the full lifecycle of an order: Purchase -> Approval -> Delivery.
2. **Forecasting**:
    - **Approach**: Supervised learning transformation of time series.
    - **Features**: `lag_7`, `lag_30`, `rolling_mean_7`.
    - **Result**: Predicts daily revenue with MAE/RMSE metrics.
3. **Churn Analysis**:
    - **Definition**: Users who purchased in the Observation Window but not in the Performance Window.
    - **Features**: Recency (days since last buy), Frequency (count), Monetary (sum).

## 💻 How to Run

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/financial-analytics.git
    cd financial-analytics
    ```

2. **Install dependencies**:

    ```bash
    pip install pandas numpy xgboost scikit-learn faker matplotlib seaborn
    ```

3. **Generate Data**:

    ```bash
    python data_generator.py
    ```

4. **Run Forecast**:

    ```bash
    python revenue_forecast.py
    ```

5. **Run Churn Prediction**:

    ```bash
    python churn_prediction.py
    ```

## 📜 License

This project is licensed under the MIT License.
