"""
Synthetic Data Generation Module
Generates realistic e-commerce datasets for the ML pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def create_data_directory():
    """Create data directory if it doesn't exist"""
    os.makedirs('data', exist_ok=True)
    print("[✓] Data directory created/verified at: ./data/")

def generate_users(n_users=5000):
    """
    Generate synthetic user data.
    
    Parameters:
    n_users: Number of users to generate
    
    Returns:
    DataFrame with user information
    """
    ages = np.random.normal(35, 15, n_users).astype(int)
    ages = np.clip(ages, 18, 80)
    
    genders = np.random.choice(['Male', 'Female', 'Other'], n_users)
    
    locations = np.random.choice(
        ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
         'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
         'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte'],
        n_users
    )
    
    users_df = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        'age': ages,
        'gender': genders,
        'location': locations
    })
    
    print(f"[✓] Generated {n_users} users")
    return users_df

def generate_products(n_products=200):
    """
    Generate synthetic product data.
    
    Parameters:
    n_products: Number of products to generate
    
    Returns:
    DataFrame with product information
    """
    categories = ['Electronics', 'Fashion', 'Home & Garden', 'Sports', 
                  'Books', 'Beauty', 'Toys', 'Food', 'Furniture']
    
    prices = np.random.gamma(shape=2, scale=50, size=n_products)
    prices = np.clip(prices, 10, 5000)
    
    products_df = pd.DataFrame({
        'product_id': [str(i) for i in range(1, n_products + 1)],
        'category': np.random.choice(categories, n_products),
        'price': prices.round(2)
    })
    
    
    print(f"[✓] Generated {n_products} products")
    return products_df

def generate_transactions(users_df, products_df, n_transactions=15000):
    """
    Generate synthetic transaction data.
    
    Parameters:
    users_df: Users DataFrame
    products_df: Products DataFrame
    n_transactions: Number of transactions to generate
    
    Returns:
    DataFrame with transaction information
    """
    base_date = datetime.now() - timedelta(days=365)
    
    transactions_df = pd.DataFrame({
        'transaction_id': range(1, n_transactions + 1),
        'user_id': np.random.choice(users_df['user_id'], n_transactions),
        'product_id': np.random.choice(products_df['product_id'].astype(str), n_transactions),
        'amount': np.random.gamma(shape=2, scale=50, size=n_transactions).round(2),
        'date': [base_date + timedelta(days=int(x)) for x in np.random.uniform(0, 365, n_transactions)]
    })
    
    # Sort by date for time-series context
    transactions_df = transactions_df.sort_values('date').reset_index(drop=True)
    
    print(f"[✓] Generated {n_transactions} transactions")
    return transactions_df

def generate_clickstream(users_df, products_df, n_events=50000):
    """
    Generate synthetic clickstream data (user interactions).
    
    Parameters:
    users_df: Users DataFrame
    products_df: Products DataFrame
    n_events: Number of events to generate
    
    Returns:
    DataFrame with clickstream information
    """
    base_date = datetime.now() - timedelta(days=30)
    
    event_types = ['view', 'add_to_cart', 'remove_from_cart', 'checkout', 'wishlist']
    
    clickstream_df = pd.DataFrame({
        'event_id': range(1, n_events + 1),
        'user_id': np.random.choice(users_df['user_id'], n_events),
        'product_id': np.random.choice(products_df['product_id'].astype(str), n_events),
        'event_type': np.random.choice(event_types, n_events, p=[0.6, 0.15, 0.05, 0.1, 0.1]),
        'timestamp': [base_date + timedelta(hours=int(x)) for x in np.random.uniform(0, 720, n_events)]
    })
    
    # Sort by timestamp
    clickstream_df = clickstream_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"[✓] Generated {n_events} clickstream events")
    return clickstream_df

def save_datasets(users_df, products_df, transactions_df, clickstream_df):
    """
    Save all datasets to CSV files.
    
    Parameters:
    users_df: Users DataFrame
    products_df: Products DataFrame
    transactions_df: Transactions DataFrame
    clickstream_df: Clickstream DataFrame
    """
    users_df.to_csv('data/users.csv', index=False)
    products_df.to_csv('data/products.csv', index=False)
    transactions_df.to_csv('data/transactions.csv', index=False)
    clickstream_df.to_csv('data/clickstream.csv', index=False)
    
    print("\n[✓] All datasets saved to /data/")
    print(f"  - users.csv: {len(users_df)} rows")
    print(f"  - products.csv: {len(products_df)} rows")
    print(f"  - transactions.csv: {len(transactions_df)} rows")
    print(f"  - clickstream.csv: {len(clickstream_df)} rows")

def main():
    """Main function to orchestrate data generation"""
    print("=" * 60)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 60 + "\n")
    
    create_data_directory()
    
    print("Generating datasets...")
    users_df = generate_users(n_users=5000)
    products_df = generate_products(n_products=200)
    transactions_df = generate_transactions(users_df, products_df, n_transactions=15000)
    clickstream_df = generate_clickstream(users_df, products_df, n_events=50000)
    
    save_datasets(users_df, products_df, transactions_df, clickstream_df)
    
    print("\n[✓] Data generation complete!")
    return users_df, products_df, transactions_df, clickstream_df

if __name__ == "__main__":
    main()
