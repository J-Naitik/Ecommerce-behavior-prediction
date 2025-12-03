"""
Data Preprocessing Module
Handles missing values, encoding, normalization, and data merging.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_raw_data():
    """Load raw CSV files"""
    print("Loading raw data...")
    users = pd.read_csv('data/users.csv')
    products = pd.read_csv('data/products.csv')
    transactions = pd.read_csv('data/transactions.csv')
    clickstream = pd.read_csv('data/clickstream.csv')
    
    print(f"[✓] Loaded {len(users)} users")
    print(f"[✓] Loaded {len(products)} products")
    print(f"[✓] Loaded {len(transactions)} transactions")
    print(f"[✓] Loaded {len(clickstream)} clickstream events\n")
    
    return users, products, transactions, clickstream

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in dataframe.
    
    Parameters:
    df: Input DataFrame
    strategy: 'mean', 'median', or 'drop'
    
    Returns:
    DataFrame with missing values handled
    """
    missing_before = df.isnull().sum().sum()
    
    if strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'drop':
        df = df.dropna()
    
    missing_after = df.isnull().sum().sum()
    
    if missing_before > 0:
        print(f"[✓] Handled missing values: {missing_before} → {missing_after}")
    
    return df

def encode_categorical_features(df, categorical_cols):
    """
    Encode categorical features using Label Encoding.
    
    Parameters:
    df: Input DataFrame
    categorical_cols: List of categorical column names
    
    Returns:
    Encoded DataFrame, encoders dictionary
    """
    encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
            encoders[col] = encoder
    
    print(f"[✓] Encoded {len(categorical_cols)} categorical features")
    return df_encoded, encoders

def convert_dates(df, date_cols):
    """
    Convert string columns to datetime.
    
    Parameters:
    df: Input DataFrame
    date_cols: List of date column names
    
    Returns:
    DataFrame with datetime columns
    """
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    if date_cols:
        print(f"[✓] Converted {len(date_cols)} date columns")
    
    return df

def normalize_numerical_features(df, numerical_cols):
    """
    Normalize numerical features using StandardScaler.
    
    Parameters:
    df: Input DataFrame
    numerical_cols: List of numerical column names
    
    Returns:
    DataFrame with normalized features, scaler object
    """
    df_normalized = df.copy()
    scaler = StandardScaler()
    
    existing_numerical = [col for col in numerical_cols if col in df_normalized.columns]
    df_normalized[existing_numerical] = scaler.fit_transform(df_normalized[existing_numerical])
    
    print(f"[✓] Normalized {len(existing_numerical)} numerical features")
    return df_normalized, scaler

def preprocess_transactions(transactions, products):
    """
    Preprocess transaction data with product information.
    
    Parameters:
    transactions: Transactions DataFrame
    products: Products DataFrame
    
    Returns:
    Preprocessed transactions DataFrame
    """
    print("\n--- Processing Transactions ---")
    
    transactions = handle_missing_values(transactions)
    transactions = convert_dates(transactions, ['date'])
    
    # Merge with products to include category
    transactions = transactions.merge(products[['product_id', 'category', 'price']], 
                                     on='product_id', how='left')
    
    print("[✓] Merged transactions with product information")
    
    return transactions

def preprocess_clickstream(clickstream, products):
    """
    Preprocess clickstream data.
    
    Parameters:
    clickstream: Clickstream DataFrame
    products: Products DataFrame
    
    Returns:
    Preprocessed clickstream DataFrame
    """
    print("\n--- Processing Clickstream ---")
    
    clickstream = handle_missing_values(clickstream)
    clickstream = convert_dates(clickstream, ['timestamp'])
    
    # Merge with products to include category
    clickstream = clickstream.merge(products[['product_id', 'category']], 
                                   on='product_id', how='left')
    
    print("[✓] Merged clickstream with product information")
    
    return clickstream

def preprocess_users(users):
    """
    Preprocess user data.
    
    Parameters:
    users: Users DataFrame
    
    Returns:
    Preprocessed users DataFrame, encoders
    """
    print("\n--- Processing Users ---")
    
    users = handle_missing_values(users)
    users_encoded, encoders = encode_categorical_features(users, ['gender', 'location'])
    
    return users_encoded, encoders

def merge_all_data(users, transactions, clickstream):
    """
    Create unified user-behavior table by merging all datasets.
    
    Parameters:
    users: Users DataFrame
    transactions: Transactions DataFrame
    clickstream: Clickstream DataFrame
    
    Returns:
    Unified DataFrame with all user behaviors
    """
    print("\n--- Creating Unified Dataset ---")
    
    # Aggregate transaction data by user
    transaction_agg = transactions.groupby('user_id').agg({
        'amount': ['sum', 'mean', 'count'],
        'date': ['min', 'max']
    }).reset_index()
    
    transaction_agg.columns = ['user_id', 'total_spent', 'avg_transaction_value', 
                               'transaction_count', 'first_transaction', 'last_transaction']
    
    # Aggregate clickstream data by user
    clickstream_agg = clickstream.groupby('user_id').agg({
        'event_id': 'count',
        'event_type': lambda x: (x == 'add_to_cart').sum()
    }).reset_index()
    
    clickstream_agg.columns = ['user_id', 'total_clicks', 'cart_adds']
    
    # Merge all
    unified_df = users.merge(transaction_agg, on='user_id', how='left')
    unified_df = unified_df.merge(clickstream_agg, on='user_id', how='left')
    
    # Fill NaN with 0 for users with no transactions
    unified_df = unified_df.fillna(0)
    
    print(f"[✓] Created unified dataset with {len(unified_df)} users")
    
    return unified_df

def save_preprocessed_data(unified_df, transactions, clickstream):
    """Save preprocessed data"""
    unified_df.to_csv('processed_data.csv', index=False)
    transactions.to_csv('processed_transactions.csv', index=False)
    clickstream.to_csv('processed_clickstream.csv', index=False)
    
    print("\n[✓] Preprocessed data saved:")
    print("  - processed_data.csv")
    print("  - processed_transactions.csv")
    print("  - processed_clickstream.csv")

def main():
    """Main preprocessing function"""
    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60 + "\n")
    
    users, products, transactions, clickstream = load_raw_data()
    
    print("--- Processing Data ---\n")
    users, encoders = preprocess_users(users)
    transactions = preprocess_transactions(transactions, products)
    clickstream = preprocess_clickstream(clickstream, products)
    
    unified_df = merge_all_data(users, transactions, clickstream)
    
    save_preprocessed_data(unified_df, transactions, clickstream)
    
    print("\n[✓] Preprocessing complete!")
    return unified_df, transactions, clickstream

if __name__ == "__main__":
    main()
