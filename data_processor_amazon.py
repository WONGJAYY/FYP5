"""
Data Processor for Amazon Sales Dataset (amazon.csv)
Processes the Amazon CSV dataset and creates features for the recommender.
Output format matches data_processor.py (7817_1.csv) so the recommender works with both.
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


def extract_price_amazon(price_str):
    """Extract numeric price from Amazon price string like '₹399' or '$29.99'."""
    if pd.isna(price_str) or not isinstance(price_str, str):
        return None
    try:
        # Remove currency symbols and commas, extract number
        price_clean = re.sub(r'[^\d.]', '', price_str)
        if price_clean:
            return float(price_clean)
    except:
        pass
    return None


def extract_brand_from_name(name):
    """Try to extract brand from product name (first word or two)."""
    if pd.isna(name) or not isinstance(name, str):
        return 'unknown'
    # Take the first word as the brand (common pattern in Amazon product names)
    words = name.strip().split()
    if words:
        return words[0]
    return 'unknown'


def parse_rating_count(rating_count_str):
    """Parse rating count from string like '24,269'."""
    if pd.isna(rating_count_str) or not isinstance(rating_count_str, str):
        return 0
    try:
        return int(re.sub(r'[^\d]', '', rating_count_str))
    except:
        return 0


def process_dataset(csv_path='amazon.csv', output_dir='processed_data_amazon'):
    """
    Process the Amazon CSV dataset and create features for the recommender.
    Output format matches data_processor.py so the recommender works seamlessly.
    
    Args:
        csv_path: Path to the Amazon CSV file
        output_dir: Directory to save processed data
    
    Returns:
        Dictionary containing processed data
    """
    print("Loading Amazon dataset...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== Handle multiple reviews per product =====
    # amazon.csv has comma-separated reviews in review_title and review_content
    # We need to split them or keep as-is for aggregation
    print("\nProcessing products...")
    
    # Get unique products (by product_id)
    products_df = df.groupby('product_id').agg({
        'product_name': 'first',
        'category': 'first',
        'discounted_price': 'first',
        'actual_price': 'first',
        'discount_percentage': 'first',
        'rating': 'first',
        'rating_count': 'first',
        'about_product': 'first',
        'img_link': 'first',
    }).reset_index()
    
    # Rename to match 7817 format
    products_df = products_df.rename(columns={
        'product_id': 'id',
        'product_name': 'name',
        'category': 'categories',
    })
    
    # Extract brand from product name
    products_df['brand'] = products_df['name'].apply(extract_brand_from_name)
    
    # Clean product names
    products_df['name_clean'] = products_df['name'].apply(clean_text)
    
    # Clean categories (replace | with comma for consistency)
    products_df['categories_clean'] = products_df['categories'].apply(
        lambda x: clean_text(str(x).replace('|', ' ')) if pd.notna(x) else ''
    )
    
    # Clean brand
    products_df['brand_clean'] = products_df['brand'].apply(
        lambda x: clean_text(x) if pd.notna(x) else 'unknown'
    )
    
    # Extract price (use discounted_price, fallback to actual_price)
    products_df['price'] = products_df['discounted_price'].apply(extract_price_amazon)
    products_df.loc[products_df['price'].isna(), 'price'] = (
        products_df.loc[products_df['price'].isna(), 'actual_price'].apply(extract_price_amazon)
    )
    
    # Create combined text for content-based filtering
    # Include about_product as additional content
    products_df['about_clean'] = products_df['about_product'].apply(clean_text)
    products_df['content'] = (
        products_df['name_clean'] + ' ' +
        products_df['categories_clean'] + ' ' +
        products_df['brand_clean'] + ' ' +
        products_df['about_clean']
    )
    
    print(f"Found {len(products_df)} unique products")
    
    # ===== Process Reviews =====
    print("\nProcessing reviews...")
    
    # Clean review text
    df['review_text_clean'] = df['review_content'].apply(clean_text)
    df['review_title_clean'] = df['review_title'].apply(clean_text)
    
    # Aggregate reviews by product
    reviews_agg = df.groupby('product_id').agg({
        'review_text_clean': lambda x: ' '.join(x.dropna()),
        'user_name': 'nunique'
    }).reset_index()
    
    reviews_agg.columns = ['id', 'all_reviews', 'unique_reviewers']
    
    # Merge with products
    products_df = products_df.merge(reviews_agg, on='id', how='left')
    
    # Set avg_rating and review_count from the dataset columns
    products_df['avg_rating'] = pd.to_numeric(products_df['rating'], errors='coerce').fillna(0)
    products_df['review_count'] = products_df['rating_count'].apply(parse_rating_count)
    
    # Fill missing values
    products_df['all_reviews'] = products_df['all_reviews'].fillna('')
    
    # Add asins column (same as id for Amazon dataset)
    products_df['asins'] = products_df['id']
    
    # Add manufacturer column (same as brand)
    products_df['manufacturer'] = products_df['brand']
    
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
    
    # Create a simple product lookup (matching 7817 format exactly)
    product_lookup = products_df[['id', 'name', 'brand', 'price', 'avg_rating', 'review_count']].copy()
    product_lookup.to_pickle(os.path.join(output_dir, 'product_lookup.pkl'))
    
    print(f"\n[OK] Amazon data processing complete!")
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
