"""
Main ML Pipeline Orchestration
Executes the complete AI-Driven Consumer Behavior Prediction pipeline.
"""

import sys
import os
from datetime import datetime

# Import all modules
from src.generate_data import main as generate_data
from src.preprocess import main as preprocess_data
from src.feature_engineering import main as engineer_features
from src.train_purchase_model import main as train_purchase_model
from src.segmentation import main as segment_customers
from src.recommend import main as build_recommendations
from src.visualize import main as create_visualizations

def print_banner():
    """Print project banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║     AI-DRIVEN CONSUMER BEHAVIOR PREDICTION FOR E-COMMERCE    ║
║                   Complete ML Pipeline                       ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_step(step_number, step_name):
    """Print step header"""
    print(f"\n{'='*60}")
    print(f"STEP {step_number}: {step_name}")
    print(f"{'='*60}\n")

def main():
    """Execute complete ML pipeline"""
    
    start_time = datetime.now()
    print_banner()
    
    try:
        # Step 1: Generate Synthetic Data
        print_step(1, "SYNTHETIC DATA GENERATION")
        users_df, products_df, transactions_df, clickstream_df = generate_data()
        
        # Step 2: Data Preprocessing
        print_step(2, "DATA PREPROCESSING")
        unified_df, transactions, clickstream = preprocess_data()
        
        # Step 3: Feature Engineering
        print_step(3, "FEATURE ENGINEERING")
        user_features, product_features, user_item_matrix = engineer_features()
        
        # Step 4: Train Purchase Prediction Model
        print_step(4, "PURCHASE PREDICTION MODEL (Classification)")
        purchase_model, metrics, feature_cols = train_purchase_model()
        
        # Step 5: Customer Segmentation
        print_step(5, "CUSTOMER SEGMENTATION (K-Means)")
        kmeans_model, cluster_labels, segmented_users = segment_customers()
        
        # Step 6: Build Recommendation Engine
        print_step(6, "RECOMMENDATION ENGINE (Collaborative Filtering)")
        item_similarity, recommendations = build_recommendations()
        
        # Step 7: Create Visualizations
        print_step(7, "VISUALIZATIONS")
        create_visualizations()
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"\n✓ All steps completed successfully!")
        print(f"✓ Total execution time: {duration.total_seconds():.2f} seconds")
        
        print("\n" + "="*60)
        print("OUTPUT FILES GENERATED")
        print("="*60)
        
        print("\nData Files:")
        print("  ✓ data/users.csv")
        print("  ✓ data/products.csv")
        print("  ✓ data/transactions.csv")
        print("  ✓ data/clickstream.csv")
        
        print("\nProcessed Data:")
        print("  ✓ processed_data.csv")
        print("  ✓ processed_transactions.csv")
        print("  ✓ processed_clickstream.csv")
        
        print("\nEngineered Features:")
        print("  ✓ user_features.csv")
        print("  ✓ product_features.csv")
        print("  ✓ user_item_matrix.csv")
        
        print("\nModel Results:")
        print("  ✓ purchase_prediction_model.pkl")
        print("  ✓ model_features.pkl")
        print("  ✓ user_segments.csv")
        print("  ✓ kmeans_model.pkl")
        print("  ✓ segmentation_scaler.pkl")
        print("  ✓ item_similarity_matrix.csv")
        print("  ✓ item_similarity.pkl")
        
        print("\nVisualizations:")
        print("  ✓ confusion_matrix.png")
        print("  ✓ feature_importance.png")
        print("  ✓ customer_segments.png")
        print("  ✓ transaction_distribution.png")
        print("  ✓ correlation_heatmap.png")
        print("  ✓ segment_distribution.png")
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("""
1. Review the generated visualizations:
   - confusion_matrix.png: Model classification performance
   - feature_importance.png: Top features driving predictions
   - customer_segments.png: Customer segmentation visualization
   - transaction_distribution.png: Purchase amount patterns

2. Analyze the output data:
   - user_features.csv: Complete feature set for all users
   - user_segments.csv: Customer segment assignments
   - item_similarity_matrix.csv: Product recommendations

3. Use the trained models:
   - Load purchase_prediction_model.pkl for new predictions
   - Load kmeans_model.pkl for new customer segmentation
   - Use item_similarity_matrix.csv for product recommendations

4. For academic purposes:
   - Document the methodology and results
   - Compare model performance with baselines
   - Analyze feature importance for business insights
        """)
        
        print("="*60)
        print(f"Pipeline completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during pipeline execution:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
