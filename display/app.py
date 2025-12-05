from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Global variables for data
transactions = None
user_item_matrix = None
products = None
item_similarity_df = None

def load_files():
    global transactions, user_item_matrix, products
    print("Loading files...")
    
    transactions = pd.read_csv("processed_transactions.csv")
    user_item_matrix = pd.read_csv("user_item_matrix.csv", index_col=0)
    products = pd.read_csv("products.csv")
    
    user_item_matrix.columns = user_item_matrix.columns.astype(str)
    products["product_id"] = products["product_id"].astype(str)
    transactions["product_id"] = transactions["product_id"].astype(str)
    
    print("Files loaded successfully.")

def compute_item_similarity():
    global item_similarity_df, user_item_matrix
    print("Computing item similarity matrix...")
    
    item_matrix = user_item_matrix.T
    similarity = cosine_similarity(item_matrix)
    
    item_similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns,
    )
    
    print("Item similarity matrix computed.")

def recommend_for_user(user_id, n=5):
    global user_item_matrix, item_similarity_df, products
    
    if user_id not in user_item_matrix.index:
        return None
    
    user_row = user_item_matrix.loc[user_id]
    purchased = user_row[user_row > 0].index.tolist()
    
    if len(purchased) == 0:
        return None
    
    scores = {}
    
    for item in user_item_matrix.columns:
        if item in purchased:
            continue
        
        sim_list = []
        for p in purchased:
            sim = item_similarity_df.loc[p, item]
            weight = user_row[p]
            sim_list.append(sim * weight)
        
        scores[item] = np.mean(sim_list)
    
    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    
    df = pd.DataFrame(top_items, columns=["product_id", "score"])
    df["product_id"] = df["product_id"].astype(str)
    df = df.merge(products, on="product_id", how="left")
    
    return df

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/home')
def home():
    user_ids = list(user_item_matrix.index)[:50]
    return render_template('home.html', user_ids=user_ids)

@app.route('/products')
def products_page():
    categories = products['category'].unique().tolist() if 'category' in products.columns else []
    return render_template('products.html', categories=categories)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/api/recommendations/<int:user_id>')
def get_recommendations(user_id):
    recs = recommend_for_user(user_id, n=5)
    
    if recs is None:
        return jsonify({'error': 'User not found or no purchase history'}), 404
    
    return jsonify(recs.to_dict('records'))

@app.route('/api/products')
def get_products():
    category = request.args.get('category')
    min_price = request.args.get('min_price', type=float)
    max_price = request.args.get('max_price', type=float)
    sort_by = request.args.get('sort_by', 'name')
    
    filtered = products.copy()
    
    if category and category != 'all':
        filtered = filtered[filtered['category'] == category]
    
    if min_price is not None:
        filtered = filtered[filtered['price'] >= min_price]
    
    if max_price is not None:
        filtered = filtered[filtered['price'] <= max_price]
    
    if sort_by == 'price_asc':
        filtered = filtered.sort_values('price', ascending=True)
    elif sort_by == 'price_desc':
        filtered = filtered.sort_values('price', ascending=False)
    
    return jsonify(filtered.to_dict('records'))

if __name__ == '__main__':
    load_files()
    compute_item_similarity()
    app.run(debug=True)
