"""
Content-Based Recommender System
Uses TF-IDF features and cosine similarity for product recommendations.
"""

import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Content-based filtering recommender using TF-IDF and cosine similarity.
    """
    
    def __init__(self, data_dir='processed_data'):
        """
        Initialize the recommender.
        
        Args:
            data_dir: Directory containing processed data files
        """
        self.data_dir = data_dir
        self.products = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.feature_names = None
        self.similarity_matrix = None
        self._loaded = False
    
    def load(self):
        """Load processed data and compute similarity matrix."""
        if self._loaded:
            return
        
        print("Loading recommender data...")
        
        # Load products
        with open(os.path.join(self.data_dir, 'products.pkl'), 'rb') as f:
            self.products = pickle.load(f)
        
        # Ensure numeric columns are properly typed (some datasets store as strings)
        import pandas as pd
        if 'avg_rating' in self.products.columns:
            self.products['avg_rating'] = pd.to_numeric(self.products['avg_rating'], errors='coerce').fillna(0)
        if 'review_count' in self.products.columns:
            self.products['review_count'] = pd.to_numeric(self.products['review_count'], errors='coerce').fillna(0).astype(int)
        
        # Load TF-IDF matrix
        with open(os.path.join(self.data_dir, 'tfidf_matrix.pkl'), 'rb') as f:
            self.tfidf_matrix = pickle.load(f)
        
        # Load TF-IDF vectorizer
        with open(os.path.join(self.data_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        # Load feature names
        with open(os.path.join(self.data_dir, 'feature_names.pkl'), 'rb') as f:
            self.feature_names = pickle.load(f)
        
        # Compute similarity matrix
        print("Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create product index mapping
        self.product_id_to_idx = {
            pid: idx for idx, pid in enumerate(self.products['id'].values)
        }
        self.idx_to_product_id = {
            idx: pid for pid, idx in self.product_id_to_idx.items()
        }
        
        self._loaded = True
        print(f"[OK] Recommender loaded with {len(self.products)} products")
    
    def get_product_by_id(self, product_id):
        """Get product information by ID."""
        if not self._loaded:
            self.load()
        
        product = self.products[self.products['id'] == product_id]
        if len(product) == 0:
            return None
        return product.iloc[0].to_dict()
    
    def get_product_by_idx(self, idx):
        """Get product information by index."""
        if not self._loaded:
            self.load()
        
        if idx < 0 or idx >= len(self.products):
            return None
        return self.products.iloc[idx].to_dict()
    
    def get_all_products(self):
        """Get all products as a list of dictionaries."""
        if not self._loaded:
            self.load()
        
        return self.products[['id', 'name', 'brand', 'price', 'avg_rating', 'review_count']].to_dict('records')
    
    def recommend(self, product_id, n_recommendations=5):
        """
        Get product recommendations based on similarity.
        
        Args:
            product_id: ID of the source product
            n_recommendations: Number of recommendations to return
        
        Returns:
            List of recommended products with similarity scores
        """
        if not self._loaded:
            self.load()
        
        if product_id not in self.product_id_to_idx:
            return []
        
        idx = self.product_id_to_idx[product_id]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort by similarity (descending)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations (excluding the product itself)
        recommendations = []
        for i, (rec_idx, score) in enumerate(sim_scores[1:n_recommendations + 1]):
            product = self.get_product_by_idx(rec_idx)
            if product:
                recommendations.append({
                    'product': product,
                    'similarity_score': float(score),
                    'rank': i + 1
                })
        
        return recommendations
    
    def get_top_features(self, product_id, n_features=10):
        """
        Get the top TF-IDF features for a product.
        
        Args:
            product_id: ID of the product
            n_features: Number of top features to return
        
        Returns:
            List of (feature_name, weight) tuples
        """
        if not self._loaded:
            self.load()
        
        if product_id not in self.product_id_to_idx:
            return []
        
        idx = self.product_id_to_idx[product_id]
        
        # Get TF-IDF weights for this product
        tfidf_row = self.tfidf_matrix[idx].toarray().flatten()
        
        # Get top features
        top_indices = np.argsort(tfidf_row)[::-1][:n_features]
        
        features = []
        for i in top_indices:
            if tfidf_row[i] > 0:
                features.append((self.feature_names[i], float(tfidf_row[i])))
        
        return features
    
    def get_similarity_explanation(self, source_id, target_id):
        """
        Explain why two products are similar.
        
        Args:
            source_id: ID of the source product
            target_id: ID of the target product
        
        Returns:
            Dictionary with explanation data
        """
        if not self._loaded:
            self.load()
        
        if source_id not in self.product_id_to_idx or target_id not in self.product_id_to_idx:
            return None
        
        source_idx = self.product_id_to_idx[source_id]
        target_idx = self.product_id_to_idx[target_id]
        
        # Get TF-IDF vectors
        source_vec = self.tfidf_matrix[source_idx].toarray().flatten()
        target_vec = self.tfidf_matrix[target_idx].toarray().flatten()
        
        # Find common important features
        common_features = []
        for i in range(len(source_vec)):
            if source_vec[i] > 0 and target_vec[i] > 0:
                common_features.append({
                    'feature': self.feature_names[i],
                    'source_weight': float(source_vec[i]),
                    'target_weight': float(target_vec[i]),
                    'combined': float(source_vec[i] * target_vec[i])
                })
        
        # Sort by combined weight
        common_features = sorted(common_features, key=lambda x: x['combined'], reverse=True)[:10]
        
        return {
            'similarity_score': float(self.similarity_matrix[source_idx][target_idx]),
            'common_features': common_features,
            'source_product': self.get_product_by_id(source_id),
            'target_product': self.get_product_by_id(target_id)
        }


def load_recommender(data_dir='processed_data'):
    """Helper function to load and return a recommender instance."""
    recommender = ContentBasedRecommender(data_dir)
    recommender.load()
    return recommender


if __name__ == '__main__':
    # Test the recommender
    recommender = load_recommender()
    
    products = recommender.get_all_products()
    if products:
        test_product = products[0]
        print(f"\nTest product: {test_product['name'][:50]}...")
        
        recommendations = recommender.recommend(test_product['id'], n_recommendations=3)
        print(f"\nTop 3 recommendations:")
        for rec in recommendations:
            print(f"  {rec['rank']}. {rec['product']['name'][:50]}... (score: {rec['similarity_score']:.3f})")
