"""
Feature Engineering Module
Creates advanced features for ML models including RFM metrics,
user behavior features, and product popularity scores.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_preprocessed_data():
    """Load preprocessed data"""
    print("Loading preprocessed data...")
    unified_df = pd.read_csv('processed_data.csv')
    transactions = pd.read_csv('processed_transactions.csv')
    clickstream = pd.read_csv('processed_clickstream.csv')
    
    print("[✓] Data loaded\n")
    return unified_df, transactions, clickstream

def calculate_rfm_metrics(transactions):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics.
    
    Parameters:
    transactions: Transactions DataFrame
    
    Returns:
    DataFrame with RFM metrics
    """
    print("Computing RFM Metrics...")
    
    # Reference date (most recent date in data)
    reference_date = pd.to_datetime(transactions['date']).max()
    
    # Recency: Days since last purchase
    recency = transactions.groupby('user_id')['date'].apply(
        lambda x: (reference_date - pd.to_datetime(x).max()).days
    ).reset_index()
    recency.columns = ['user_id', 'recency']
    
    # Frequency: Number of purchases
    frequency = transactions.groupby('user_id').size().reset_index(name='frequency')
    
    # Monetary: Total amount spent
    monetary = transactions.groupby('user_id')['amount'].sum().reset_index()
    monetary.columns = ['user_id', 'monetary_value']
    
    # Merge RFM metrics
    rfm_df = recency.merge(frequency, on='user_id').merge(monetary, on='user_id')
    
    print(f"[✓] RFM metrics calculated for {len(rfm_df)} users")
    return rfm_df

def calculate_user_features(transactions, clickstream):
    """
    Calculate user-level behavioral features.
    
    Parameters:
    transactions: Transactions DataFrame
    clickstream: Clickstream DataFrame
    
    Returns:
    DataFrame with user features
    """
    print("Computing User-Level Features...")
    
    user_features = {}
    
    # Average session length (clicks per session approximation)
    avg_session_length = clickstream.groupby('user_id')['event_id'].count().reset_index()
    avg_session_length.columns = ['user_id', 'avg_session_length']
    
    # Click-to-cart ratio
    clicks_by_user = clickstream.groupby('user_id')['event_id'].count().reset_index()
    carts_by_user = clickstream[clickstream['event_type'] == 'add_to_cart'].groupby('user_id')['event_id'].count().reset_index()
    
    click_to_cart = clicks_by_user.merge(carts_by_user, on='user_id', how='left', suffixes=('_clicks', '_carts'))
    click_to_cart.columns = ['user_id', 'total_clicks', 'cart_adds']
    click_to_cart['cart_adds'] = click_to_cart['cart_adds'].fillna(0)
    click_to_cart['click_to_cart_ratio'] = (click_to_cart['cart_adds'] / click_to_cart['total_clicks']).fillna(0)
    
    # Unique categories viewed
    unique_categories = clickstream.groupby('user_id')['category'].nunique().reset_index()
    unique_categories.columns = ['user_id', 'unique_categories_viewed']
    
    # Categories viewed by user
    categories_per_user = transactions.groupby('user_id')['category'].nunique().reset_index()
    categories_per_user.columns = ['user_id', 'categories_purchased']
    
    # Merge all user features
    user_feat = avg_session_length.merge(click_to_cart[['user_id', 'click_to_cart_ratio']], 
                                         on='user_id', how='left')
    user_feat = user_feat.merge(unique_categories, on='user_id', how='left')
    user_feat = user_feat.merge(categories_per_user, on='user_id', how='left')
    
    user_feat = user_feat.fillna(0)
    
    print(f"[✓] User features calculated for {len(user_feat)} users")
    return user_feat

def calculate_product_features(transactions):
    """
    Calculate product-level features.
    
    Parameters:
    transactions: Transactions DataFrame
    
    Returns:
    DataFrame with product features
    """
    print("Computing Product-Level Features...")
    
    # Popularity score (number of purchases)
    popularity = transactions.groupby('product_id')['transaction_id'].count().reset_index()
    popularity.columns = ['product_id', 'popularity_score']
    
    # Average rating (synthetic - based on purchase frequency and amount consistency)
    purchase_stats = transactions.groupby('product_id').agg({
        'amount': ['mean', 'std'],
        'transaction_id': 'count'
    }).reset_index()
    
    purchase_stats.columns = ['product_id', 'avg_price', 'price_std', 'purchase_count']
    
    # Synthetic rating based on consistency and popularity
    # Products with stable prices and high purchases get higher ratings
    purchase_stats['avg_rating'] = (
        (purchase_stats['purchase_count'] / purchase_stats['purchase_count'].max() * 0.7) +
        ((1 / (1 + purchase_stats['price_std'].fillna(1))) * 0.3) * 5
    ).round(1)
    
    product_feat = popularity.merge(purchase_stats[['product_id', 'avg_rating']], on='product_id')
    
    print(f"[✓] Product features calculated for {len(product_feat)} products")
    return product_feat

def create_user_purchase_behavior(transactions):
    """
    Create purchase behavior labels for classification.
    
    Parameters:
    transactions: Transactions DataFrame
    
    Returns:
    DataFrame with purchase probability indicators
    """
    print("Creating Purchase Behavior Indicators...")
    
    # Identify recent vs. historical transactions
    max_date = pd.to_datetime(transactions['date']).max()
    split_date = max_date - timedelta(days=30)
    
    # Recent purchases (last 30 days)
    recent_purchases = transactions[pd.to_datetime(transactions['date']) >= split_date]
    recent_users = set(recent_purchases['user_id'].unique())
    
    # Historical purchases (more than 30 days ago)
    historical_purchases = transactions[pd.to_datetime(transactions['date']) < split_date]
    historical_users = set(historical_purchases['user_id'].unique())
    
    # Create target: users who purchased historically AND recently = likely to purchase again
    all_users = set(transactions['user_id'].unique())
    purchase_probability = pd.DataFrame({
        'user_id': list(all_users),
        'will_purchase': [1 if uid in recent_users and uid in historical_users else 0 
                         for uid in all_users]
    })
    
    print(f"[✓] Purchase probability indicators created")
    print(f"  - Will purchase: {purchase_probability['will_purchase'].sum()} users")
    
    return purchase_probability

def create_user_item_matrix(transactions):
    """
    Create user-item interaction matrix for recommendation system.
    
    Parameters:
    transactions: Transactions DataFrame
    
    Returns:
    User-item matrix
    """
    print("Creating User-Item Interaction Matrix...")
    
    # Create interaction matrix (user x product) with purchase amounts as weights
    user_item_matrix = transactions.pivot_table(
        index='user_id',
        columns='product_id',
        values='amount',
        aggfunc='sum',
        fill_value=0
    )
    
    print(f"[✓] User-Item matrix created: {user_item_matrix.shape}")
    return user_item_matrix

def combine_all_features(unified_df, rfm_df, user_feat, user_purchase_behavior):
    """
    Combine all engineered features into single user feature table.
    
    Parameters:
    unified_df: Base unified DataFrame
    rfm_df: RFM metrics
    user_feat: User behavior features
    user_purchase_behavior: Purchase probability
    
    Returns:
    Complete user features DataFrame
    """
    print("\nCombining all features...")
    
    # Merge all features
    user_features_complete = unified_df.merge(rfm_df, on='user_id', how='left')
    user_features_complete = user_features_complete.merge(user_feat, on='user_id', how='left')
    user_features_complete = user_features_complete.merge(user_purchase_behavior, on='user_id', how='left')
    
    # Fill any remaining NaN values
    user_features_complete = user_features_complete.fillna(0)
    
    print(f"[✓] Combined features: {len(user_features_complete)} users × {len(user_features_complete.columns)} features")
    
    return user_features_complete

def save_engineered_features(user_features, product_features, user_item_matrix):
    """Save engineered features to CSV"""
    user_features.to_csv('user_features.csv', index=False)
    product_features.to_csv('product_features.csv', index=False)
    user_item_matrix.to_csv('user_item_matrix.csv')
    
    print("\n[✓] Engineered features saved:")
    print("  - user_features.csv")
    print("  - product_features.csv")
    print("  - user_item_matrix.csv")

def main():
    """Main feature engineering function"""
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60 + "\n")
    
    unified_df, transactions, clickstream = load_preprocessed_data()
    
    print("--- Engineering Features ---\n")
    
    rfm_df = calculate_rfm_metrics(transactions)
    user_feat = calculate_user_features(transactions, clickstream)
    product_feat = calculate_product_features(transactions)
    purchase_behavior = create_user_purchase_behavior(transactions)
    user_item_matrix = create_user_item_matrix(transactions)
    
    user_features_complete = combine_all_features(unified_df, rfm_df, user_feat, purchase_behavior)
    
    save_engineered_features(user_features_complete, product_feat, user_item_matrix)
    
    print("\n[✓] Feature engineering complete!")
    return user_features_complete, product_feat, user_item_matrix

if __name__ == "__main__":
    main()
