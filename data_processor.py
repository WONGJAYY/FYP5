"""
Data Processor for E-Commerce Recommender System
Processes the raw CSV dataset and creates features for the recommender.
"""

import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def clean_text(text):
    """Clean and normalize text data."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.lower()


def extract_price(prices_str):
    """Extract average price from the prices JSON string."""
    if pd.isna(prices_str) or not isinstance(prices_str, str):
        return None
    try:
        # Find all price amounts in USD
        usd_prices = re.findall(r'"amountMax":(\d+\.?\d*)', prices_str)
        if usd_prices:
            return float(usd_prices[0])
    except:
        pass
    return None


def process_dataset(csv_path='7817_1.csv', output_dir='processed_data'):
    """
    Process the raw CSV dataset and create features for the recommender.
    
    Args:
        csv_path: Path to the raw CSV file
        output_dir: Directory to save processed data
    
    Returns:
        Dictionary containing processed data
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== Process Products =====
    print("\nProcessing products...")
    
    # Get unique products (by id)
    products_df = df.groupby('id').agg({
        'asins': 'first',
        'brand': 'first',
        'categories': 'first',
        'name': 'first',
        'prices': 'first',
        'manufacturer': 'first'
    }).reset_index()
    
    # Clean product names
    products_df['name_clean'] = products_df['name'].apply(clean_text)
    
    # Clean categories
    products_df['categories_clean'] = products_df['categories'].apply(clean_text)
    
    # Clean brand
    products_df['brand_clean'] = products_df['brand'].apply(
        lambda x: clean_text(x) if pd.notna(x) else 'unknown'
    )
    
    # Extract price
    products_df['price'] = products_df['prices'].apply(extract_price)
    
    # Create combined text for content-based filtering
    products_df['content'] = (
        products_df['name_clean'] + ' ' +
        products_df['categories_clean'] + ' ' +
        products_df['brand_clean']
    )
    
    print(f"Found {len(products_df)} unique products")
    
    # ===== Process Reviews =====
    print("\nProcessing reviews...")
    
    # Clean review text
    df['review_text_clean'] = df['reviews.text'].apply(clean_text)
    df['review_title_clean'] = df['reviews.title'].apply(clean_text)
    
    # Extract rating (handle various formats)
    def extract_rating(rating):
        if pd.isna(rating):
            return None
        if isinstance(rating, (int, float)):
            return float(rating)
        try:
            return float(str(rating).split()[0])
        except:
            return None
    
    df['rating'] = df['reviews.rating'].apply(extract_rating)
    
    # Aggregate reviews by product
    reviews_agg = df.groupby('id').agg({
        'review_text_clean': lambda x: ' '.join(x.dropna()),
        'rating': ['mean', 'count'],
        'reviews.username': 'nunique'
    }).reset_index()
    
    reviews_agg.columns = ['id', 'all_reviews', 'avg_rating', 'review_count', 'unique_reviewers']
    
    # Merge with products
    products_df = products_df.merge(reviews_agg, on='id', how='left')
    
    # Fill missing values
    products_df['avg_rating'] = products_df['avg_rating'].fillna(0)
    products_df['review_count'] = products_df['review_count'].fillna(0)
    products_df['all_reviews'] = products_df['all_reviews'].fillna('')
    
    # ===== Create TF-IDF Features =====
    print("\nCreating TF-IDF features...")
    
    # Combine content with reviews for richer features
    products_df['full_content'] = products_df['content'] + ' ' + products_df['all_reviews']
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    
    tfidf_matrix = tfidf.fit_transform(products_df['full_content'])
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # ===== Save Processed Data =====
    print("\nSaving processed data...")
    
    # Save products DataFrame
    products_df.to_pickle(os.path.join(output_dir, 'products.pkl'))
    
    # Save TF-IDF vectorizer and matrix
    with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)
    
    with open(os.path.join(output_dir, 'tfidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    
    # Save feature names for explainability
    feature_names = tfidf.get_feature_names_out()
    with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Create a simple product lookup
    product_lookup = products_df[['id', 'name', 'brand', 'price', 'avg_rating', 'review_count']].copy()
    product_lookup.to_pickle(os.path.join(output_dir, 'product_lookup.pkl'))
    
    print(f"\n[OK] Data processing complete!")
    print(f"   - Products: {len(products_df)}")
    print(f"   - TF-IDF features: {tfidf_matrix.shape[1]}")
    print(f"   - Output directory: {output_dir}")
    
    return {
        'products': products_df,
        'tfidf_vectorizer': tfidf,
        'tfidf_matrix': tfidf_matrix,
        'feature_names': feature_names
    }


if __name__ == '__main__':
    process_dataset()
