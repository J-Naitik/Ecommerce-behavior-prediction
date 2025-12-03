# AI-Driven Consumer Behavior Prediction for E-commerce

A complete machine learning pipeline for predicting consumer behavior, segmenting customers, and providing product recommendations using synthetic e-commerce data.

## Project Overview

This project demonstrates a full ML workflow including:
- **Synthetic Data Generation**: Realistic e-commerce datasets (users, products, transactions, clickstream)
- **Data Preprocessing**: Handling missing values, encoding, normalization, and data merging
- **Feature Engineering**: RFM metrics, user behavior features, product popularity scores
- **Machine Learning Models**:
  - Purchase Prediction (Classification)
  - Customer Segmentation (K-Means Clustering)
  - Product Recommendation (Collaborative Filtering)
- **Visualizations**: Confusion matrices, feature importance, clustering plots, correlations

## Project Structure

\`\`\`
ml-pipeline/
├── data/                          # Generated CSV files
│   ├── users.csv
│   ├── products.csv
│   ├── transactions.csv
│   └── clickstream.csv
├── src/
│   ├── generate_data.py          # Synthetic data generation
│   ├── preprocess.py             # Data preprocessing
│   ├── feature_engineering.py    # Feature creation
│   ├── train_purchase_model.py   # Classification model
│   ├── segmentation.py           # K-Means clustering
│   ├── recommend.py              # Collaborative filtering
│   └── visualize.py              # All visualizations
├── main.py                        # Main execution pipeline
├── requirements.txt               # Python dependencies
└── README.md                      # Documentation
\`\`\`

## Installation & Setup

### 1. Clone or Download the Project
\`\`\`bash
cd ml-pipeline
\`\`\`

### 2. Create a Virtual Environment (Recommended)
\`\`\`bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
\`\`\`

### 3. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Running the Pipeline

### Option 1: Run Everything (Recommended for First Run)
\`\`\`bash
python main.py
\`\`\`

This will execute all steps sequentially:
1. Generate synthetic data
2. Preprocess data
3. Engineer features
4. Train purchase prediction model
5. Perform customer segmentation
6. Generate recommendations
7. Create all visualizations

### Option 2: Run Individual Modules
\`\`\`bash
# Generate data only
python src/generate_data.py

# Preprocess data
python src/preprocess.py

# Engineer features
python src/feature_engineering.py

# Train purchase model
python src/train_purchase_model.py

# Segment customers
python src/segmentation.py

# Generate recommendations
python src/recommend.py

# Create visualizations
python src/visualize.py
\`\`\`

## Key Features

### Model 1: Purchase Prediction Classifier
- **Algorithm**: Random Forest Classifier
- **Target**: Predict if user will purchase in next session
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Features**: RFM metrics, click-to-cart ratio, category diversity

### Model 2: Customer Segmentation
- **Algorithm**: K-Means Clustering
- **Features**: RFM metrics, transaction frequency, monetary value
- **Output**: 4 customer segments with interpretations
- **Visualization**: Scatter plot of clusters

### Model 3: Recommendation Engine
- **Algorithm**: Item-Based Collaborative Filtering (Cosine Similarity)
- **Output**: Top-5 recommended products for any user
- **Scalable**: Can recommend for any user in the dataset

## Output Files

After running the pipeline, you'll get:
- **Preprocessed Data**: `processed_data.csv`
- **Engineered Features**: `user_features.csv`
- **Model Predictions**: Confusion matrix, accuracy scores
- **Cluster Labels**: Customer segments with centroids
- **Visualizations**: 
  - `confusion_matrix.png`
  - `feature_importance.png`
  - `customer_segments.png`
  - `transaction_distribution.png`
  - `correlation_heatmap.png`

## System Requirements

- Python 3.8 or higher
- 4GB RAM (minimum)
- 500MB disk space
- Windows, macOS, or Linux

## VS Code Setup

1. Open the project folder in VS Code
2. Select the Python interpreter from your venv:
   - Open Command Palette (Ctrl+Shift+P)
   - Type "Python: Select Interpreter"
   - Choose the one from your venv folder
3. Run main.py using Ctrl+F5 or the Run button

## Academic Notes

This pipeline demonstrates:
- Data engineering best practices
- Feature engineering techniques for customer analytics
- Supervised learning (classification)
- Unsupervised learning (clustering)
- Collaborative filtering for recommendations
- Proper train/test splits and evaluation metrics
- Data visualization techniques

## Author

Created for academic machine learning projects.

## License

MIT License - Feel free to use for educational purposes.
