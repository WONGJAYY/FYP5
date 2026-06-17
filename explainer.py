"""
Explainability Module for E-Commerce Recommender
Provides SHAP, LIME, and LLM-based explanations for recommendations.
"""

import numpy as np
import pickle
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()


class SHAPExplainer:
    """
    SHAP-based explainer for feature importance.
    Uses kernel SHAP for model-agnostic explanations.
    """
    
    def __init__(self, recommender):
        """
        Initialize SHAP explainer.
        
        Args:
            recommender: ContentBasedRecommender instance
        """
        self.recommender = recommender
        if not recommender._loaded:
            recommender.load()
    
    def explain_recommendation(self, source_id, target_id, n_features=10):
        """
        Explain why a product was recommended using feature importance.
        
        This is a simplified SHAP-like explanation based on TF-IDF feature overlap.
        For production, you would use actual SHAP values with a trained model.
        
        Args:
            source_id: ID of the source product
            target_id: ID of the target (recommended) product
            n_features: Number of top features to show
        
        Returns:
            Dictionary with SHAP-style feature importance
        """
        explanation = self.recommender.get_similarity_explanation(source_id, target_id)
        if not explanation:
            return None
        
        # Convert to SHAP-style output
        features = []
        values = []
        
        for feat in explanation['common_features'][:n_features]:
            features.append(feat['feature'])
            values.append(feat['combined'])
        
        # Normalize to sum to similarity score
        total = sum(values) if values else 1
        normalized_values = [v / total * explanation['similarity_score'] for v in values]
        
        return {
            'feature_names': features,
            'shap_values': normalized_values,
            'base_value': 0.0,
            'similarity_score': explanation['similarity_score'],
            'explanation_type': 'SHAP'
        }
    
    def get_feature_importance_plot_data(self, source_id, target_id, n_features=10):
        """Get data formatted for creating a SHAP-style bar plot."""
        explanation = self.explain_recommendation(source_id, target_id, n_features)
        if not explanation:
            return None
        
        return {
            'features': explanation['feature_names'],
            'importance': explanation['shap_values'],
            'title': f"Feature Importance (Similarity: {explanation['similarity_score']:.3f})"
        }


class LIMEExplainer:
    """
    LIME-based explainer for local interpretable explanations.
    """
    
    def __init__(self, recommender):
        """
        Initialize LIME explainer.
        
        Args:
            recommender: ContentBasedRecommender instance
        """
        self.recommender = recommender
        if not recommender._loaded:
            recommender.load()
    
    def explain_recommendation(self, source_id, target_id, n_features=10):
        """
        Explain a recommendation using LIME-style local explanations.
        
        This provides a different perspective by showing which features
        contribute positively or negatively to the recommendation.
        
        Args:
            source_id: ID of the source product
            target_id: ID of the target (recommended) product
            n_features: Number of features to explain
        
        Returns:
            Dictionary with LIME-style explanations
        """
        if not self.recommender._loaded:
            self.recommender.load()
        
        source_idx = self.recommender.product_id_to_idx.get(source_id)
        target_idx = self.recommender.product_id_to_idx.get(target_id)
        
        if source_idx is None or target_idx is None:
            return None
        
        # Get TF-IDF vectors
        source_vec = self.recommender.tfidf_matrix[source_idx].toarray().flatten()
        target_vec = self.recommender.tfidf_matrix[target_idx].toarray().flatten()
        
        # Calculate contribution of each feature
        contributions = []
        for i in range(len(source_vec)):
            if source_vec[i] > 0 or target_vec[i] > 0:
                # Positive contribution if both have the feature
                # Negative if only one has it
                if source_vec[i] > 0 and target_vec[i] > 0:
                    contribution = source_vec[i] * target_vec[i]
                    polarity = 'positive'
                elif source_vec[i] > 0:
                    contribution = -source_vec[i] * 0.1
                    polarity = 'negative'
                else:
                    contribution = -target_vec[i] * 0.1
                    polarity = 'negative'
                
                contributions.append({
                    'feature': self.recommender.feature_names[i],
                    'contribution': contribution,
                    'polarity': polarity,
                    'source_weight': float(source_vec[i]),
                    'target_weight': float(target_vec[i])
                })
        
        # Sort by absolute contribution
        contributions = sorted(contributions, key=lambda x: abs(x['contribution']), reverse=True)
        
        # Get top positive and negative
        positive = [c for c in contributions if c['polarity'] == 'positive'][:n_features // 2]
        negative = [c for c in contributions if c['polarity'] == 'negative'][:n_features // 2]
        
        return {
            'positive_features': positive,
            'negative_features': negative,
            'all_contributions': contributions[:n_features],
            'explanation_type': 'LIME'
        }


class LLMExplainer:
    """
    LLM-based explainer using NVIDIA NIM (Llama 3.3 70B).
    Generates natural language explanations for recommendations.
    """
    
    def __init__(self, recommender):
        """
        Initialize LLM explainer.
        
        Args:
            recommender: ContentBasedRecommender instance
        """
        self.recommender = recommender
        self.api_key = os.getenv('NVIDIA_API_KEY')
        self.api_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.model = "meta/llama-3.3-70b-instruct"
        
        if not recommender._loaded:
            recommender.load()
    
    def _call_llm(self, prompt, max_tokens=500):
        """Call the NVIDIA NIM API."""
        if not self.api_key:
            return "Error: NVIDIA API key not configured."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful e-commerce assistant that explains product recommendations in a friendly, concise manner. Focus on the key features that make products similar."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            return f"Error calling LLM API: {str(e)}"
    
    def explain_recommendation(self, source_id, target_id):
        """
        Generate a natural language explanation for a recommendation.
        
        Args:
            source_id: ID of the source product
            target_id: ID of the target (recommended) product
        
        Returns:
            Natural language explanation string
        """
        # Get product details
        source = self.recommender.get_product_by_id(source_id)
        target = self.recommender.get_product_by_id(target_id)
        
        if not source or not target:
            return "Unable to generate explanation: product not found."
        
        # Get common features
        explanation = self.recommender.get_similarity_explanation(source_id, target_id)
        common_features = [f['feature'] for f in explanation['common_features'][:5]] if explanation else []
        similarity_score = explanation['similarity_score'] if explanation else 0.0
        
        prompt = f"""Explain why this product was recommended based on the user's interest:

User was viewing: {source.get('name', 'Unknown Product')}
- Brand: {source.get('brand', 'Unknown')}
- Rating: {float(source.get('avg_rating', 0)):.1f}/5

Recommended product: {target.get('name', 'Unknown Product')}
- Brand: {target.get('brand', 'Unknown')}
- Rating: {float(target.get('avg_rating', 0)):.1f}/5

Key matching features: {', '.join(common_features) if common_features else 'Similar category'}
Similarity score: {similarity_score:.2f}

Provide a brief, friendly explanation (2-3 sentences) of why this recommendation makes sense."""

        return self._call_llm(prompt)
    
    def summarize_reviews(self, product_id):
        """
        Generate a summary of product reviews.
        
        Args:
            product_id: ID of the product
        
        Returns:
            Review summary string
        """
        product = self.recommender.get_product_by_id(product_id)
        if not product:
            return "Product not found."
        
        reviews = product.get('all_reviews', '')[:2000]  # Limit context
        
        if not reviews.strip():
            return "No reviews available for this product."
        
        prompt = f"""Summarize the following customer reviews for "{product.get('name', 'this product')}" in 3-4 bullet points:

Reviews:
{reviews}

Focus on:
- Key pros mentioned by customers
- Any cons or concerns
- Overall sentiment"""

        return self._call_llm(prompt)
    
    def chat(self, user_message, product_context=None, semantic_searcher=None):
        """
        Handle a chat message with optional product context and semantic search.

        When a semantic_searcher is provided, the method first searches the
        product database for items matching the user's query, then feeds those
        real product details to the LLM so its response is grounded in actual
        catalog data (RAG pattern).

        Args:
            user_message: User's chat message
            product_context: Optional dictionary with current product info
            semantic_searcher: Optional SemanticSearcher instance for RAG

        Returns:
            AI response string
        """
        context = ""

        # ── RAG: Search the real product database ──
        search_results = None
        if semantic_searcher is not None:
            try:
                search_results = semantic_searcher.search(user_message, top_k=5)
                products_context = semantic_searcher.format_results_for_llm(
                    search_results, max_products=3
                )
                context += f"""Here are the most relevant products from our catalog that match the user's query:

{products_context}

"""
            except Exception as e:
                context += f"(Product search encountered an error: {e})\n\n"

        # ── Optional: currently selected product ──
        if product_context:
            try:
                context += f"""The user is currently viewing this product:
- Product: {product_context.get('name', 'Unknown')}
- Brand: {product_context.get('brand', 'Unknown')}
- Price: ${float(product_context.get('price', 0)):.2f}
- Rating: {float(product_context.get('avg_rating', 0)):.1f}/5

"""
            except (ValueError, TypeError):
                pass

        prompt = f"""{context}User question: {user_message}

IMPORTANT: You must ONLY recommend or mention products that appear in the catalog data above. Do NOT invent or hallucinate products that are not listed. If no relevant products were found, tell the user honestly and suggest they refine their query.

Provide a helpful, friendly response as an e-commerce shopping assistant. When recommending products, mention their name, price, rating, and a brief reason why they are a good match."""

        return self._call_llm(prompt, max_tokens=500)


def create_explainers(recommender):
    """Create all explainer instances."""
    return {
        'shap': SHAPExplainer(recommender),
        'lime': LIMEExplainer(recommender),
        'llm': LLMExplainer(recommender)
    }


if __name__ == '__main__':
    from recommender import load_recommender
    
    # Test explainers
    recommender = load_recommender()
    explainers = create_explainers(recommender)
    
    products = recommender.get_all_products()
    if len(products) >= 2:
        source = products[0]
        recs = recommender.recommend(source['id'], n_recommendations=1)
        if recs:
            target = recs[0]['product']
            
            print("Testing SHAP explainer...")
            shap_exp = explainers['shap'].explain_recommendation(source['id'], target['id'])
            print(f"SHAP features: {shap_exp['feature_names'][:3] if shap_exp else 'N/A'}")
            
            print("\nTesting LIME explainer...")
            lime_exp = explainers['lime'].explain_recommendation(source['id'], target['id'])
            print(f"LIME positive: {[f['feature'] for f in lime_exp['positive_features'][:3]] if lime_exp else 'N/A'}")
