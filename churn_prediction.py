import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import os

def load_data():
    if not os.path.exists("orders.csv") or not os.path.exists("order_items.csv"):
        print("Data not found.")
        return None
    
    orders = pd.read_csv("orders.csv")
    items = pd.read_csv("order_items.csv")
    
    # Merge to get order value
    order_values = items.groupby('order_id')['price'].sum().reset_index()
    orders = orders.merge(order_values, on='order_id', how='left')
    orders['price'] = orders['price'].fillna(0)
    
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    
    return orders

def prepare_features(orders):
    print("Preparing churn features...")
    
    # Snapshot date: max date + 1 day
    snapshot_date = orders['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    
    # Customer Aggregation
    customer_df = orders.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'count',
        'price': 'sum'
    }).reset_index()
    
    customer_df.rename(columns={
        'order_purchase_timestamp': 'recency',
        'order_id': 'frequency',
        'price': 'monetary'
    }, inplace=True)
    
    # Define Churn: Recency > 90 days
    # Note: In a real simulation, we'd split by time window (observation vs performance).
    # Here, for simplicity in a single script, we label current state.
    # A better approach for ML:
    # Train on data < T-90. Target: Did they buy in [T-90, T]? 
    # But given synthetic data randomly generated over 2 years, let's stick to a simpler heuristic for demonstration.
    
    # Let's say we define "at risk of churning" as Recency > 90.
    # But that's the label we calculate. We need to predict it based on OTHER behaviors?
    # Actually, Recency IS the definition of churn often. A predictive model usually needs a window.
    # Strategy:
    # 1. Split data into Observation Window (Jan 2022 - Sep 2023) and Performance Window (Oct 2023 - Dec 2023).
    # 2. Features built from Observation Window.
    # 3. Target: Did they buy in Performance Window?
    
    cutoff_date = pd.to_datetime("2023-10-01")
    
    # Train Data (Observation)
    train_orders = orders[orders['order_purchase_timestamp'] < cutoff_date]
    
    feature_df = train_orders.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (cutoff_date - x.max()).days,
        'order_id': 'count',
        'price': 'sum'
    }).reset_index()
    feature_df.rename(columns={'order_purchase_timestamp': 'recency', 'order_id': 'frequency', 'price': 'monetary'}, inplace=True)
    
    # Target (Performance)
    future_orders = orders[orders['order_purchase_timestamp'] >= cutoff_date]
    active_customers = future_orders['customer_id'].unique()
    
    feature_df['churned'] = ~feature_df['customer_id'].isin(active_customers)
    feature_df['churned'] = feature_df['churned'].astype(int)
    
    print(f"Churn Rate in Test Window: {feature_df['churned'].mean():.2%}")
    
    return feature_df

def train_churn_model(df):
    print("Training Churn Prediction Model...")
    
    X = df[['recency', 'frequency', 'monetary']]
    y = df['churned']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title('Churn Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig('churn_confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    orders = load_data()
    if orders is not None:
        features = prepare_features(orders)
        train_churn_model(features)
