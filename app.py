from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

app = Flask(__name__)

# Load datasets
users = pd.read_csv("UserDataset.csv")
products = pd.read_csv("market_product_dataset.csv")
ratings = pd.read_csv("RatingsDataset.csv")

# Create user-item matrix
def create_user_item_matrix(ratings_df):
    user_mapper = {user: idx for idx, user in enumerate(ratings_df['userId'].unique())}
    product_mapper = {prod: idx for idx, prod in enumerate(ratings_df['productId'].unique())}

    user_index = ratings_df['userId'].map(user_mapper)
    product_index = ratings_df['productId'].map(product_mapper)

    user_item_matrix = csr_matrix((ratings_df['rating'], (user_index, product_index)),
                                  shape=(len(user_mapper), len(product_mapper)))
    return user_item_matrix, user_mapper, {v: k for k, v in product_mapper.items()}

def recommend_products(user_id, user_item_matrix, user_mapper, reverse_product_mapper, products_df, k=5):
    try:
        user_idx = user_mapper[user_id]
    except KeyError:
        return []  # No recommendations if the user is not found

    # Collaborative Filtering
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_item_matrix)
    user_vector = user_item_matrix[user_idx, :].toarray().flatten()
    distances, indices = knn.kneighbors([user_vector], n_neighbors=k + 10)

    # Filter valid recommendations for CF
    valid_indices = [i for i in indices.flatten()[1:] if i in reverse_product_mapper]
    recommended_ids_cf = [reverse_product_mapper[i] for i in valid_indices]

    # Content-Based Filtering
    tfidf = TfidfVectorizer(stop_words='english')
    products_df['description'] = products_df['description'].fillna('')
    tfidf_matrix = tfidf.fit_transform(products_df['description'])

    user_products = ratings[ratings['userId'] == user_id]['productId'].tolist()
    if user_products:
        user_product_idx = products_df[products_df['productId'].isin(user_products)].index
        user_profiles = tfidf_matrix[user_product_idx]
        content_similarities = cosine_similarity(user_profiles, tfidf_matrix)
        content_scores = content_similarities.mean(axis=0)
        recommended_ids_cb = products_df.iloc[content_scores.argsort()[::-1]]['productId'].tolist()
    else:
        recommended_ids_cb = []

    # Merge Recommendations
    final_recommendations = list(dict.fromkeys(recommended_ids_cb + recommended_ids_cf))

    # Filter by category with safe handling for missing data
    if user_products:
        try:
            user_category = products_df[products_df['productId'] == user_products[-1]]['category'].values[0]
            final_recommendations = [
                prod for prod in final_recommendations
                if not products_df.loc[products_df['productId'] == prod].empty and
                   products_df.loc[products_df['productId'] == prod, 'category'].values[0] == user_category
            ][:k]
        except IndexError:
            # Handle cases where product or category is missing
            final_recommendations = []
    else:
        final_recommendations = []
        #test

    # Get Titles for Recommendations
    recommended_titles = products_df[products_df['productId'].isin(final_recommendations)]['title'].tolist()
    return recommended_titles


# Flask routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/submit', methods=['POST'])
def submit():
    global users, products, ratings  # Declare globals at the beginning

    title = request.form['title']
    category = request.form['category']
    price = float(request.form['price'])
    description = request.form['description']

    # Assign new user ID
    new_user_id = users['userId'].max() + 1

    # Update users dataset
    new_user = pd.DataFrame([{'userId': new_user_id}])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv("UserDataset.csv", index=False)

    # Update products dataset
    new_product_id = products['productId'].max() + 1
    new_product = pd.DataFrame([{'productId': new_product_id, 'title': title, 'category': category, 'price': price, 'description': description}])
    products = pd.concat([products, new_product], ignore_index=True)
    products.to_csv("market_product_dataset.csv", index=False)

    # Simulate a rating and update ratings dataset
    new_rating = pd.DataFrame([{'userId': new_user_id, 'productId': new_product_id, 'rating': 5}])  # Assuming new users like their own input
    ratings = pd.concat([ratings, new_rating], ignore_index=True)
    ratings.to_csv("RatingsDataset.csv", index=False)

    # Retrain the model
    user_item_matrix, user_mapper, reverse_product_mapper = create_user_item_matrix(ratings)

    # Generate recommendations
    recommendations = recommend_products(new_user_id, user_item_matrix, user_mapper, reverse_product_mapper, products, k=5)

    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
