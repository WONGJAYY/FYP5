"""
Explainability Module for E-Commerce Recommender
Provides SHAP, LIME, and LLM-based explanations for recommendations.
"""

import numpy as np
import pickle
import os
from dotenv import load_dotenv
import requests
import shap
from lime.lime_text import LimeTextExplainer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()


class SHAPExplainer:
    """
    SHAP-based explainer for feature importance.
    Uses Kernel/Linear SHAP for model-agnostic explanations.
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
        
        Uses actual SHAP LinearExplainer to compute feature attributions.
        
        Args:
            source_id: ID of the source product
            target_id: ID of the target (recommended) product
            n_features: Number of top features to show
        
        Returns:
            Dictionary with SHAP feature importance
        """
        source_idx = self.recommender.product_id_to_idx.get(source_id)
        target_idx = self.recommender.product_id_to_idx.get(target_id)
        
        if source_idx is None or target_idx is None:
            return None
            
        source_vec = self.recommender.tfidf_matrix[source_idx]
        target_vec = self.recommender.tfidf_matrix[target_idx]
        
        # Calculate coefficients for the linear cosine similarity surrogate model:
        # sim(x) = (x . source_vec) / (||x|| * ||source_vec||)
        # We approximate ||x|| as ||target_vec|| (locally constant)
        target_norm = np.linalg.norm(target_vec.toarray())
        source_norm = np.linalg.norm(source_vec.toarray())
        denom = target_norm * source_norm if (target_norm * source_norm) > 0 else 1.0
        
        coef = source_vec.toarray().flatten() / denom
        
        # Define background dataset (sample first 100 rows to keep it fast)
        n_background = min(100, self.recommender.tfidf_matrix.shape[0])
        background_data = self.recommender.tfidf_matrix[:n_background].toarray()
        
        # Instantiate SHAP LinearExplainer
        explainer = shap.LinearExplainer((coef, 0.0), background_data)
        
        # Compute SHAP values for the target vector
        shap_values_obj = explainer(target_vec.toarray())
        shap_vals = shap_values_obj.values[0]
        
        # Extract features and their SHAP values
        feature_importance = []
        for i in range(len(shap_vals)):
            if shap_vals[i] != 0:
                feature_importance.append({
                    'feature': self.recommender.feature_names[i],
                    'value': float(shap_vals[i])
                })
        
        # Sort descending by absolute attribution
        feature_importance = sorted(feature_importance, key=lambda x: abs(x['value']), reverse=True)[:n_features]
        
        features = [x['feature'] for x in feature_importance]
        values = [x['value'] for x in feature_importance]
        
        similarity_score = float(self.recommender.similarity_matrix[source_idx][target_idx])
        
        return {
            'feature_names': features,
            'shap_values': values,
            'base_value': float(explainer.expected_value),
            'similarity_score': similarity_score,
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
        
        Uses the actual LimeTextExplainer on target text against source TF-IDF vectors.
        
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
        
        # Get target product's text content and source's vector
        target_product = self.recommender.products.iloc[target_idx]
        target_text = target_product.get('full_content', '')
        source_vec = self.recommender.tfidf_matrix[source_idx]
        
        if not target_text.strip():
            return {
                'positive_features': [],
                'negative_features': [],
                'all_contributions': [],
                'explanation_type': 'LIME'
            }
        
        def predict_fn(texts):
            vecs = self.recommender.tfidf_vectorizer.transform(texts)
            sims = cosine_similarity(vecs, source_vec).flatten()
            sims = np.clip(sims, 0.0, 1.0)
            return np.column_stack([1.0 - sims, sims])
            
        explainer = LimeTextExplainer(class_names=["dissimilar", "similar"])
        
        # Explain label 1 (similar) using 1000 perturbation samples for responsiveness
        exp = explainer.explain_instance(
            target_text,
            predict_fn,
            labels=(1,),
            num_features=n_features,
            num_samples=1000
        )
        
        lime_results = exp.as_list(label=1)
        
        positive = []
        negative = []
        all_contributions = []
        
        for word, val in lime_results:
            feature_idx = self.recommender.tfidf_vectorizer.vocabulary_.get(word)
            source_w = 0.0
            target_w = 0.0
            if feature_idx is not None:
                source_w = float(source_vec[0, feature_idx])
                target_w = float(self.recommender.tfidf_matrix[target_idx, feature_idx])
                
            contribution_info = {
                'feature': str(word),
                'contribution': float(val),
                'polarity': 'positive' if val > 0 else 'negative',
                'source_weight': source_w,
                'target_weight': target_w
            }
            
            all_contributions.append(contribution_info)
            if val > 0:
                positive.append(contribution_info)
            else:
                negative.append(contribution_info)
                
        positive = sorted(positive, key=lambda x: abs(x['contribution']), reverse=True)
        negative = sorted(negative, key=lambda x: abs(x['contribution']), reverse=True)
        all_contributions = sorted(all_contributions, key=lambda x: abs(x['contribution']), reverse=True)
        
        return {
            'positive_features': positive,
            'negative_features': negative,
            'all_contributions': all_contributions[:n_features],
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
