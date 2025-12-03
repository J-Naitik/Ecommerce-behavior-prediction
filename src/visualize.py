"""
Visualization Module
Creates comprehensive visualizations for model results and data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def load_model_results():
    """Load model results and data"""
    print("Loading data for visualizations...")
    
    # Load metrics from purchase model
    user_features = pd.read_csv('user_features.csv')
    user_segments = pd.read_csv('user_segments.csv')
    transactions = pd.read_csv('processed_transactions.csv')
    
    # Load model
    model = pickle.load(open('purchase_prediction_model.pkl', 'rb'))
    model_features = pickle.load(open('model_features.pkl', 'rb'))
    
    print("[✓] Data loaded\n")
    
    return user_features, user_segments, transactions, model, model_features

def plot_confusion_matrix(user_features, model, model_features):
    """
    Plot confusion matrix for purchase prediction model.
    
    Parameters:
    user_features: User features DataFrame
    model: Trained model
    model_features: Feature names used in model
    """
    print("Creating confusion matrix visualization...")
    
    # Prepare data
    X = user_features[model_features].fillna(0)
    y = user_features['will_purchase'].values
    
    # Predictions
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['No Purchase', 'Purchase'],
                yticklabels=['No Purchase', 'Purchase'])
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Purchase Prediction Model', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("[✓] Saved: confusion_matrix.png\n")
    plt.close()

def plot_feature_importance(model, model_features):
    """
    Plot feature importance from Random Forest model.
    
    Parameters:
    model: Trained Random Forest model
    model_features: Feature names
    """
    print("Creating feature importance visualization...")
    
    # Get feature importance
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:10]  # Top 10
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(10), importance[indices], color='steelblue')
    ax.set_yticks(range(10))
    ax.set_yticklabels([model_features[i] for i in indices])
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Feature Importance - Purchase Prediction', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f'{width:.4f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("[✓] Saved: feature_importance.png\n")
    plt.close()

def plot_customer_segments(user_segments):
    """
    Plot customer segments using RFM metrics.
    
    Parameters:
    user_segments: Segmented users DataFrame
    """
    print("Creating customer segmentation visualization...")
    
    # Plot: Monetary vs Frequency colored by Cluster
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot 1: Recency vs Frequency
    scatter1 = axes[0].scatter(user_segments['recency'], user_segments['frequency'],
                               c=user_segments['cluster'], cmap='viridis', 
                               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel('Recency (Days)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency (Purchases)', fontsize=11, fontweight='bold')
    axes[0].set_title('Recency vs Frequency', fontsize=12, fontweight='bold')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # Scatter plot 2: Monetary vs Frequency
    scatter2 = axes[1].scatter(user_segments['monetary_value'], user_segments['frequency'],
                               c=user_segments['cluster'], cmap='viridis',
                               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('Monetary Value ($)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency (Purchases)', fontsize=11, fontweight='bold')
    axes[1].set_title('Monetary vs Frequency', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')
    
    fig.suptitle('Customer Segments (K-Means Clustering)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('customer_segments.png', dpi=300, bbox_inches='tight')
    print("[✓] Saved: customer_segments.png\n")
    plt.close()

def plot_transaction_distribution(transactions):
    """
    Plot transaction amount distribution.
    
    Parameters:
    transactions: Transactions DataFrame
    """
    print("Creating transaction distribution visualization...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(transactions['amount'], bins=50, color='steelblue', 
            edgecolor='black', alpha=0.7)
    
    ax.axvline(transactions['amount'].mean(), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: ${transactions["amount"].mean():.2f}')
    ax.axvline(transactions['amount'].median(), color='green', linestyle='--',
              linewidth=2, label=f'Median: ${transactions["amount"].median():.2f}')
    
    ax.set_xlabel('Transaction Amount ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    
    ax.text(0.98, 0.97, f'Total Transactions: {len(transactions):,}\nMean: ${transactions["amount"].mean():.2f}\nStd Dev: ${transactions["amount"].std():.2f}',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('transaction_distribution.png', dpi=300, bbox_inches='tight')
    print("[✓] Saved: transaction_distribution.png\n")
    plt.close()

def plot_correlation_heatmap(user_features):
    """
    Plot correlation heatmap of key features.
    
    Parameters:
    user_features: User features DataFrame
    """
    print("Creating correlation heatmap...")
    
    # Select numeric features for correlation
    numeric_features = ['age', 'recency', 'frequency', 'monetary_value',
                       'avg_session_length', 'click_to_cart_ratio', 
                       'transaction_count']
    
    corr_matrix = user_features[numeric_features].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
               square=True, ax=ax, cbar_kws={'label': 'Correlation'},
               linewidths=1, linecolor='gray')
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("[✓] Saved: correlation_heatmap.png\n")
    plt.close()

def plot_segment_distribution(user_segments):
    """
    Plot distribution of customers across segments.
    
    Parameters:
    user_segments: Segmented users DataFrame
    """
    print("Creating segment distribution visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    segment_counts = user_segments['cluster'].value_counts().sort_index()
    colors = sns.color_palette('viridis', len(segment_counts))
    
    axes[0].pie(segment_counts, labels=[f'Segment {i}' for i in segment_counts.index],
               autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0].set_title('Customer Distribution by Segment', fontsize=12, fontweight='bold')
    
    # Bar chart
    axes[1].bar(segment_counts.index, segment_counts.values, color=colors, edgecolor='black')
    axes[1].set_xlabel('Segment', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
    axes[1].set_title('Customer Count by Segment', fontsize=12, fontweight='bold')
    axes[1].set_xticks(segment_counts.index)
    
    # Add count labels on bars
    for i, v in enumerate(segment_counts.values):
        axes[1].text(segment_counts.index[i], v + 50, str(v), ha='center', fontweight='bold')
    
    fig.suptitle('Customer Segmentation Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('segment_distribution.png', dpi=300, bbox_inches='tight')
    print("[✓] Saved: segment_distribution.png\n")
    plt.close()

def main():
    """Main visualization function"""
    print("=" * 60)
    print("VISUALIZATIONS")
    print("=" * 60 + "\n")
    
    user_features, user_segments, transactions, model, model_features = load_model_results()
    
    plot_confusion_matrix(user_features, model, model_features)
    plot_feature_importance(model, model_features)
    plot_customer_segments(user_segments)
    plot_transaction_distribution(transactions)
    plot_correlation_heatmap(user_features)
    plot_segment_distribution(user_segments)
    
    print("=" * 60)
    print("[✓] All visualizations complete!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("  - customer_segments.png")
    print("  - transaction_distribution.png")
    print("  - correlation_heatmap.png")
    print("  - segment_distribution.png")

if __name__ == "__main__":
    main()
