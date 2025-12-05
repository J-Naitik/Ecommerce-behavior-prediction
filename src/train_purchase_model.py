"""
Purchase Prediction Model (Classification)
Trains a Random Forest classifier to predict if a user will purchase.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import pickle

def load_features():
    """Load engineered features"""
    print("Loading feature data...")
    user_features = pd.read_csv('featured-eng/user_features.csv')
    print(f"[✓] Loaded {len(user_features)} user records\n")
    return user_features

def prepare_data_for_training(user_features):
    """
    Prepare data for model training.
    
    Parameters:
    user_features: Complete user features DataFrame
    
    Returns:
    X, y, train/test splits
    """
    print("Preparing training data...")
    
    # Select relevant features for purchase prediction
    feature_cols = [
        'age', 'gender', 'recency', 'frequency', 'monetary_value',
        'avg_session_length', 'click_to_cart_ratio', 'unique_categories_viewed',
        'cart_adds', 'transaction_count', 'avg_transaction_value'
    ]
    
    # Target variable
    y = user_features['will_purchase'].values
    
    # Features
    X = user_features[feature_cols].copy()
    X = X.fillna(0)
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[✓] Training set: {len(X_train)} samples")
    print(f"[✓] Test set: {len(X_test)} samples")
    print(f"[✓] Features: {len(feature_cols)}")
    print(f"[✓] Target distribution - Purchase: {y.sum()} ({y.mean()*100:.1f}%)\n")
    
    return X, y, X_train, X_test, y_train, y_test, feature_cols

def train_model(X_train, y_train):
    """
    Train Random Forest classifier.
    
    Parameters:
    X_train: Training features
    y_train: Training labels
    
    Returns:
    Trained model
    """
    print("Training Random Forest Classifier...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("[✓] Model training complete\n")
    return model

def evaluate_model(model, X_test, y_test, feature_cols):
    """
    Evaluate model performance on test set.
    
    Parameters:
    model: Trained model
    X_test: Test features
    y_test: Test labels
    feature_cols: Feature column names
    
    Returns:
    Dictionary of metrics
    """
    print("Evaluating Model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Results
    print(f"\n{'='*50}")
    print("MODEL PERFORMANCE METRICS")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print(f"\n{'='*50}")
    print("CONFUSION MATRIX")
    print(f"{'='*50}")
    print(f"True Negatives:  {conf_matrix[0,0]}")
    print(f"False Positives: {conf_matrix[0,1]}")
    print(f"False Negatives: {conf_matrix[1,0]}")
    print(f"True Positives:  {conf_matrix[1,1]}")
    
    print(f"\n{'='*50}")
    print("TOP 10 IMPORTANT FEATURES")
    print(f"{'='*50}")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:30s}: {row['importance']:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'feature_importance': feature_importance,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test
    }
    
    return metrics

def save_model(model, feature_cols):
    """Save trained model"""
    pickle.dump(model, open('train-model/purchase_prediction_model.pkl', 'wb'))
    pickle.dump(feature_cols, open('train-model/model_features.pkl', 'wb'))
    print("\n[✓] Model saved as 'purchase_prediction_model.pkl'")

def main():
    """Main training function"""
    print("=" * 60)
    print("PURCHASE PREDICTION MODEL (Classification)")
    print("=" * 60 + "\n")
    
    user_features = load_features()
    X, y, X_train, X_test, y_train, y_test, feature_cols = prepare_data_for_training(user_features)
    
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, feature_cols)
    
    save_model(model, feature_cols)
    
    print("\n[✓] Purchase prediction model training complete!")
    return model, metrics, feature_cols

if __name__ == "__main__":
    main()
