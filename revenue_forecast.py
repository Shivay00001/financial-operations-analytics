import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

plt.style.use('ggplot')

def load_data():
    if not os.path.exists("orders.csv") or not os.path.exists("order_items.csv"):
        print("Data not found. Run data_generator.py first.")
        return None
        
    orders = pd.read_csv("orders.csv")
    items = pd.read_csv("order_items.csv")
    
    # Merge
    df = orders.merge(items, on='order_id')
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    return df

def feature_engineering(df):
    print("Preparing time series features...")
    
    # Aggregate daily revenue
    daily_revenue = df.groupby(df['order_purchase_timestamp'].dt.date)['price'].sum().reset_index()
    daily_revenue.columns = ['date', 'revenue']
    daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
    daily_revenue = daily_revenue.sort_values('date')
    
    # Add features
    daily_revenue['day_of_week'] = daily_revenue['date'].dt.dayofweek
    daily_revenue['month'] = daily_revenue['date'].dt.month
    daily_revenue['day_of_year'] = daily_revenue['date'].dt.dayofyear
    
    # Lag features
    for lag in [1, 7, 14, 30]:
        daily_revenue[f'lag_{lag}'] = daily_revenue['revenue'].shift(lag)
        
    # Rolling features
    daily_revenue['rolling_mean_7'] = daily_revenue['revenue'].shift(1).rolling(window=7).mean()
    daily_revenue['rolling_mean_30'] = daily_revenue['revenue'].shift(1).rolling(window=30).mean()
    
    # Drop NaNs created by lags
    daily_revenue = daily_revenue.dropna()
    
    return daily_revenue

def train_forecast_model(data):
    print("Training XGBoost Forecast Model...")
    
    # Split Train/Test (Time-based)
    cutoff = int(len(data) * 0.8)
    train = data.iloc[:cutoff]
    test = data.iloc[cutoff:]
    
    features = ['day_of_week', 'month', 'day_of_year', 'lag_1', 'lag_7', 'lag_14', 'lag_30', 'rolling_mean_7', 'rolling_mean_30']
    target = 'revenue'
    
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    
    model = XGBRegressor(n_estimators=1000, learning_rate=0.01, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Predict
    test['prediction'] = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(test['revenue'], test['prediction'])
    rmse = np.sqrt(mean_squared_error(test['revenue'], test['prediction']))
    
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    
    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(train['date'], train['revenue'], label='Train')
    plt.plot(test['date'], test['revenue'], label='Test Actual')
    plt.plot(test['date'], test['prediction'], label='Test Predicted', linestyle='--')
    plt.title('Daily Revenue Forecast (XGBoost)')
    plt.legend()
    plt.savefig('revenue_forecast.png')
    plt.close()
    
    # Save Feature Importance
    plt.figure(figsize=(10, 6))
    sorted_idx = model.feature_importances_.argsort()
    plt.barh(np.array(features)[sorted_idx], model.feature_importances_[sorted_idx])
    plt.title("XGBoost Feature Importance")
    plt.savefig('forecast_feature_importance.png')
    plt.close()

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        data = feature_engineering(df)
        train_forecast_model(data)
