"""
Customer Segmentation (K-Means Clustering)
Segments customers into distinct groups based on RFM and behavioral metrics.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

def load_features():
    """Load engineered features"""
    print("Loading feature data...")
    user_features = pd.read_csv('featured-eng/user_features.csv')
    print(f"[✓] Loaded {len(user_features)} user records\n")
    return user_features

def prepare_segmentation_features(user_features):
    """
    Prepare features for clustering.
    
    Parameters:
    user_features: Complete user features DataFrame
    
    Returns:
    Scaled features, original scaler
    """
    print("Preparing segmentation features...")
    
    # Select features for segmentation (RFM + behavioral)
    segmentation_features = [
        'recency', 'frequency', 'monetary_value',
        'avg_session_length', 'click_to_cart_ratio',
        'transaction_count', 'unique_categories_viewed'
    ]
    
    X = user_features[segmentation_features].copy()
    X = X.fillna(0)
    
    # Standardize features (critical for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"[✓] Features scaled: {X_scaled.shape}")
    print(f"  - Recency: {X['recency'].describe()['mean']:.2f} ± {X['recency'].describe()['std']:.2f}")
    print(f"  - Frequency: {X['frequency'].describe()['mean']:.2f} ± {X['frequency'].describe()['std']:.2f}")
    print(f"  - Monetary: {X['monetary_value'].describe()['mean']:.2f} ± {X['monetary_value'].describe()['std']:.2f}\n")
    
    return X_scaled, scaler, segmentation_features

def find_optimal_clusters(X_scaled, max_k=10):
    """
    Find optimal number of clusters using elbow method.
    
    Parameters:
    X_scaled: Scaled feature matrix
    max_k: Maximum number of clusters to try
    
    Returns:
    Inertia values for each k
    """
    print("Finding optimal number of clusters (Elbow Method)...")
    
    inertias = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        print(f"  k={k}: Inertia={kmeans.inertia_:.2f}")
    
    print("[✓] Optimal k appears to be around 3-4\n")
    
    return inertias

def train_segmentation_model(X_scaled, n_clusters=4):
    """
    Train K-Means clustering model.
    
    Parameters:
    X_scaled: Scaled feature matrix
    n_clusters: Number of clusters
    
    Returns:
    Trained K-Means model
    """
    print(f"Training K-Means Clustering ({n_clusters} clusters)...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    print(f"[✓] Model training complete\n")
    return kmeans, cluster_labels

def analyze_segments(user_features, cluster_labels, segmentation_features):
    """
    Analyze and interpret customer segments.
    
    Parameters:
    user_features: Original user features
    cluster_labels: Cluster assignments
    segmentation_features: Feature names used for segmentation
    
    Returns:
    DataFrame with segment analysis
    """
    print("Analyzing Customer Segments...")
    
    user_features_copy = user_features.copy()
    user_features_copy['cluster'] = cluster_labels
    
    segment_analysis = user_features_copy.groupby('cluster')[segmentation_features].mean()
    segment_sizes = user_features_copy['cluster'].value_counts().sort_index()
    
    print(f"\n{'='*70}")
    print("SEGMENT DISTRIBUTION")
    print(f"{'='*70}")
    for cluster_id in sorted(user_features_copy['cluster'].unique()):
        size = segment_sizes[cluster_id]
        pct = 100 * size / len(user_features_copy)
        print(f"Segment {cluster_id}: {size:5d} customers ({pct:5.1f}%)")
    
    print(f"\n{'='*70}")
    print("SEGMENT CHARACTERISTICS")
    print(f"{'='*70}\n")
    
    # Interpretations
    for cluster_id in sorted(segment_analysis.index):
        recency = segment_analysis.loc[cluster_id, 'recency']
        frequency = segment_analysis.loc[cluster_id, 'frequency']
        monetary = segment_analysis.loc[cluster_id, 'monetary_value']
        
        print(f"Segment {cluster_id}:")
        print(f"  Recency:   {recency:.2f} days (last purchase)")
        print(f"  Frequency: {frequency:.2f} purchases")
        print(f"  Monetary:  ${monetary:.2f} total spent")
        
        # Segment interpretation
        if frequency > frequency.max() * 0.7 and monetary > monetary.max() * 0.7:
            interpretation = "HIGH-VALUE CUSTOMERS"
        elif recency < recency.min() * 1.3:
            interpretation = "ACTIVE CUSTOMERS"
        elif frequency < frequency.min() * 1.3:
            interpretation = "AT-RISK CUSTOMERS"
        else:
            interpretation = "MODERATE CUSTOMERS"
        
        print(f"  Type: {interpretation}\n")
    
    return user_features_copy

def save_segmentation_model(kmeans, scaler, cluster_labels):
    """Save clustering model"""
    pickle.dump(kmeans, open('segmentation/kmeans_model.pkl', 'wb'))
    pickle.dump(scaler, open('segmentation/segmentation_scaler.pkl', 'wb'))
    
    print("[✓] Models saved:")
    print("  - kmeans_model.pkl")
    print("  - segmentation_scaler.pkl")

def main():
    """Main segmentation function"""
    print("=" * 60)
    print("CUSTOMER SEGMENTATION (K-Means Clustering)")
    print("=" * 60 + "\n")
    
    user_features = load_features()
    X_scaled, scaler, segmentation_features = prepare_segmentation_features(user_features)
    
    inertias = find_optimal_clusters(X_scaled, max_k=10)
    
    kmeans, cluster_labels = train_segmentation_model(X_scaled, n_clusters=4)
    
    segmented_users = analyze_segments(user_features, cluster_labels, segmentation_features)
    
    save_segmentation_model(kmeans, scaler, cluster_labels)
    
    segmented_users.to_csv('segmentation/user_segments.csv', index=False)
    print("\n[✓] Segmented users saved to 'user_segments.csv'")
    
    print("\n[✓] Customer segmentation complete!")
    return kmeans, cluster_labels, segmented_users

if __name__ == "__main__":
    main()
