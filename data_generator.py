import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import os

fake = Faker()
np.random.seed(42)
random.seed(42)

def generate_financial_data(n_orders=20000, n_customers=5000, n_sellers=100):
    print(f"Generating E-commerce data: {n_orders} orders, {n_customers} customers...")
    
    # --- Customers ---
    customer_ids = [fake.uuid4() for _ in range(n_customers)]
    customer_data = []
    states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'DF', 'GO', 'PE']
    for cid in customer_ids:
        customer_data.append([cid, fake.unique.random_int(min=10000, max=99999), fake.city(), random.choice(states)])
    
    customers_df = pd.DataFrame(customer_data, columns=['customer_id', 'customer_unique_id', 'customer_city', 'customer_state'])
    
    # --- Sellers ---
    seller_ids = [fake.uuid4() for _ in range(n_sellers)]
    
    # --- Orders ---
    orders_data = []
    order_items_data = []
    payments_data = []
    
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    order_statuses = ['delivered', 'shipped', 'canceled', 'invoiced', 'processing']
    payment_types = ['credit_card', 'boleto', 'voucher', 'debit_card']
    
    for _ in range(n_orders):
        order_id = fake.uuid4()
        customer_id = random.choice(customer_ids)
        order_status = np.random.choice(order_statuses, p=[0.90, 0.05, 0.02, 0.02, 0.01])
        
        # Timestamps
        purchase_timestamp = fake.date_time_between(start_date=start_date, end_date=end_date)
        
        if order_status == 'canceled':
            approved_at = None
            delivered_date = None
        else:
            approved_at = purchase_timestamp + timedelta(minutes=random.randint(10, 1440))
            delivered_date = approved_at + timedelta(days=random.randint(2, 20))
            
        orders_data.append([order_id, customer_id, order_status, purchase_timestamp, approved_at, delivered_date])
        
        # Order Items
        n_items = np.random.randint(1, 4) # 1 to 3 items
        total_value = 0
        
        for i in range(1, n_items + 1):
            product_id = fake.uuid4()
            seller_id = random.choice(seller_ids)
            price = round(np.random.uniform(10, 500), 2)
            freight_value = round(np.random.uniform(10, 50), 2)
            total_value += price + freight_value
            
            order_items_data.append([order_id, i, product_id, seller_id, price, freight_value])
            
        # Payments
        payment_type = np.random.choice(payment_types, p=[0.75, 0.15, 0.05, 0.05])
        installments = 1
        if payment_type == 'credit_card':
            installments = np.random.randint(1, 12)
            
        payments_data.append([order_id, 1, payment_type, installments, round(total_value, 2)])
        
    orders_df = pd.DataFrame(orders_data, columns=['order_id', 'customer_id', 'order_status', 
                                                   'order_purchase_timestamp', 'order_approved_at', 
                                                   'order_delivered_customer_date'])
    
    order_items_df = pd.DataFrame(order_items_data, columns=['order_id', 'order_item_id', 'product_id', 
                                                             'seller_id', 'price', 'freight_value'])
    
    payments_df = pd.DataFrame(payments_data, columns=['order_id', 'payment_sequential', 'payment_type', 
                                                       'payment_installments', 'payment_value'])
    
    # Save to CSVs
    customers_df.to_csv("customers.csv", index=False)
    orders_df.to_csv("orders.csv", index=False)
    order_items_df.to_csv("order_items.csv", index=False)
    payments_df.to_csv("payments.csv", index=False)
    
    print("Financial data generation complete.")
    print(f"Orders: {orders_df.shape}")
    print(f"Order Items: {order_items_df.shape}")
    print(f"Payments: {payments_df.shape}")

if __name__ == "__main__":
    generate_financial_data(25000, 5000, 100)
