"""
Recommendation Engine (Item-Based Collaborative Filtering)
Recommends products to users based on item similarity and user purchase history.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def load_data():
    """Load necessary data"""
    print("Loading data for recommendation engine...")
    transactions = pd.read_csv('processed_transactions.csv')
    user_item_matrix = pd.read_csv('user_item_matrix.csv', index_col=0)
    products = pd.read_csv('data/products.csv')
    
    # --- FIX PRODUCT ID DTYPE ---
    user_item_matrix.columns = user_item_matrix.columns.astype(str)
    products['product_id'] = products['product_id'].astype(str)
    transactions['product_id'] = transactions['product_id'].astype(str)

    print(f"[✓] User-item matrix: {user_item_matrix.shape}")
    print(f"[✓] Products: {len(products)}\n")
    
    return transactions, user_item_matrix, products


def compute_item_similarity(user_item_matrix):
    """
    Compute item-to-item similarity matrix using Cosine Similarity.
    
    Parameters:
    user_item_matrix: User × Product purchase matrix
    
    Returns:
    Item similarity matrix
    """
    print("Computing item similarity matrix (Cosine Similarity)...")
    
    # Transpose to get products as rows
    item_matrix = user_item_matrix.T
    
    # Calculate cosine similarity between items
    item_similarity = cosine_similarity(item_matrix)
    
    # Convert to DataFrame for easier access
    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )
    
    print(f"[✓] Item similarity matrix computed: {item_similarity_df.shape}\n")
    
    return item_similarity_df

def get_user_recommendations(user_id, user_item_matrix, item_similarity_df, 
                             products_df, n_recommendations=5):
    """
    Get top-N product recommendations for a user.
    
    Parameters:
    user_id: User ID to get recommendations for
    user_item_matrix: User × Product purchase matrix
    item_similarity_df: Item similarity matrix
    products_df: Products DataFrame with metadata
    n_recommendations: Number of recommendations to return
    
    Returns:
    DataFrame with recommended products
    """
    
    if user_id not in user_item_matrix.index:
        print(f"User {user_id} not found in data")
        return None
    
    # Get items the user has purchased
    user_purchases = user_item_matrix.loc[user_id]
    purchased_items = user_purchases[user_purchases > 0].index.tolist()
    
    if len(purchased_items) == 0:
        print(f"User {user_id} has no purchase history")
        return None
    
    # Calculate recommendation scores
    recommendation_scores = {}
    
    for item in user_item_matrix.columns:
        if item in purchased_items:
            continue  # Skip already purchased items
        
        # Score based on similarity to purchased items
        similarity_scores = []
        for purchased_item in purchased_items:
            similarity = item_similarity_df.loc[purchased_item, item]
            purchase_value = user_purchases[purchased_item]
            similarity_scores.append(similarity * purchase_value)
        
        recommendation_scores[item] = np.mean(similarity_scores)
    
    # Sort by score and get top N
    sorted_recommendations = sorted(recommendation_scores.items(), 
                                   key=lambda x: x[1], reverse=True)
    top_recommendations = sorted_recommendations[:n_recommendations]
    
    # Create results DataFrame
    recommendations_df = pd.DataFrame({
        'product_id': [item[0] for item in top_recommendations],
        'score': [item[1] for item in top_recommendations]
    })
    
    # Merge with product info
    recommendations_df['product_id'] = recommendations_df['product_id'].astype(str)
    recommendations_df = recommendations_df.merge(products_df, on='product_id', how='left')
    
    return recommendations_df

def generate_batch_recommendations(user_item_matrix, item_similarity_df, 
                                  products_df, sample_users=10):
    """
    Generate recommendations for multiple users.
    
    Parameters:
    user_item_matrix: User × Product purchase matrix
    item_similarity_df: Item similarity matrix
    products_df: Products DataFrame
    sample_users: Number of sample users to generate recommendations for
    
    Returns:
    List of recommendation dataframes
    """
    print(f"Generating recommendations for {sample_users} sample users...\n")
    
    all_recommendations = []
    user_list = list(user_item_matrix.index)
    sample_user_ids = np.random.choice(user_list, min(sample_users, len(user_list)), 
                                       replace=False)
    
    for user_id in sample_user_ids:
        recommendations = get_user_recommendations(user_id, user_item_matrix, 
                                                  item_similarity_df, products_df, 
                                                  n_recommendations=5)
        
        if recommendations is not None:
            print(f"User {user_id} - Recommended Products:")
            for idx, row in recommendations.iterrows():
                print(f"  {idx+1}. Product {row['product_id']} ({row['category']}) - "
                      f"Price: ${row['price']:.2f}, Score: {row['score']:.4f}")
            print()
            all_recommendations.append((user_id, recommendations))
    
    return all_recommendations

def save_recommendation_model(item_similarity_df):
    """Save item similarity matrix"""
    item_similarity_df.to_csv('item_similarity_matrix.csv')
    pickle.dump(item_similarity_df, open('item_similarity.pkl', 'wb'))
    
    print("[✓] Recommendation model saved:")
    print("  - item_similarity_matrix.csv")
    print("  - item_similarity.pkl")

def main():
    """Main recommendation function"""
    print("=" * 60)
    print("RECOMMENDATION ENGINE (Item-Based Collaborative Filtering)")
    print("=" * 60 + "\n")
    
    transactions, user_item_matrix, products = load_data()
    
    item_similarity_df = compute_item_similarity(user_item_matrix)
    
    recommendations = generate_batch_recommendations(user_item_matrix, item_similarity_df, 
                                                     products, sample_users=10)
    
    save_recommendation_model(item_similarity_df)
    
    print("[✓] Recommendation engine complete!")
    return item_similarity_df, recommendations

if __name__ == "__main__":
    main()
