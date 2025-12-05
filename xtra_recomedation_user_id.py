import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# LOAD FILES
# ---------------------------------------------------------
def load_files():
    print("Loading files...")

    transactions = pd.read_csv("processed_transactions.csv")
    user_item_matrix = pd.read_csv("user_item_matrix.csv", index_col=0)
    products = pd.read_csv("data/products.csv")

    # Make all product IDs strings (important!)
    user_item_matrix.columns = user_item_matrix.columns.astype(str)
    products["product_id"] = products["product_id"].astype(str)
    transactions["product_id"] = transactions["product_id"].astype(str)

    print("Files loaded successfully.")
    return transactions, user_item_matrix, products


# ---------------------------------------------------------
# COMPUTE ITEM SIMILARITY MATRIX
# ---------------------------------------------------------
def compute_item_similarity(user_item_matrix):
    print("Computing item similarity matrix...")

    item_matrix = user_item_matrix.T
    similarity = cosine_similarity(item_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns,
    )

    print("Item similarity matrix computed.")
    return similarity_df


# ---------------------------------------------------------
# GENERATE USER RECOMMENDATIONS
# ---------------------------------------------------------
def recommend_for_user(user_id, user_item_matrix, item_similarity_df, products_df, n=5):

    if user_id not in user_item_matrix.index:
        print(f"User {user_id} not found.")
        return None

    user_row = user_item_matrix.loc[user_id]
    purchased = user_row[user_row > 0].index.tolist()

    if len(purchased) == 0:
        print(f"User {user_id} has no purchases.")
        return None

    scores = {}

    for item in user_item_matrix.columns:
        if item in purchased:
            continue

        sim_list = []
        for p in purchased:
            sim = item_similarity_df.loc[p, item]
            weight = user_row[p]  # purchase quantity
            sim_list.append(sim * weight)

        scores[item] = np.mean(sim_list)

    # Sort and take top N
    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

    df = pd.DataFrame(top_items, columns=["product_id", "score"])
    df["product_id"] = df["product_id"].astype(str)

    df = df.merge(products_df, on="product_id", how="left")

    return df


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------
def main():
    transactions, user_item_matrix, products = load_files()
    item_similarity = compute_item_similarity(user_item_matrix)
    print("Available user IDs:", list(user_item_matrix.index)[:50])

    while True:
        print("\n------------------------------------")
        raw = input("Enter User ID for recommendations (or 'exit'): ").strip()

        if raw.lower() == "exit":
            break
        
        try:
            user_id = int(raw)
        except ValueError:
            print("Please enter a valid numeric User ID.")
            continue
        


        recs = recommend_for_user(
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            item_similarity_df=item_similarity,
            products_df=products,
            n=5
        )

        if recs is not None:
            print("\nTop Recommendations:")
            print(recs[["product_id", "category", "price", "score"]])
        print("------------------------------------")


if __name__ == "__main__":
    main()
