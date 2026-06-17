"""
Semantic Search Module for E-Commerce Recommender
Uses sentence-transformers to enable natural language product search.
Supports RAG (Retrieval-Augmented Generation) for the AI Chat feature.
"""

import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearcher:
    """
    Semantic search engine for products using sentence-transformers.
    Converts product text and user queries into dense vectors,
    then finds the most relevant products via cosine similarity.
    """

    def __init__(self, recommender, data_dir='processed_data'):
        """
        Initialize the semantic searcher.

        Args:
            recommender: ContentBasedRecommender instance (must be loaded)
            data_dir: Directory for caching embeddings
        """
        self.recommender = recommender
        self.data_dir = data_dir
        self.model = None
        self.product_embeddings = None
        self.product_texts = []
        self.product_records = []
        self._loaded = False

        # Path to cache embeddings so we don't recompute every time
        self.cache_path = os.path.join(data_dir, 'semantic_embeddings.pkl')

    def load(self):
        """Load or compute product embeddings."""
        if self._loaded:
            return

        # Lazy import so the module doesn't fail if sentence-transformers
        # isn't installed yet (graceful degradation).
        from sentence_transformers import SentenceTransformer

        print("[SemanticSearch] Loading sentence-transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Ensure recommender data is loaded
        if not self.recommender._loaded:
            self.recommender.load()

        # Build product text corpus
        self._build_product_corpus()

        # Try to load cached embeddings
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    cached = pickle.load(f)
                # Verify cache matches current product count
                if cached['n_products'] == len(self.product_texts):
                    self.product_embeddings = cached['embeddings']
                    print(f"[SemanticSearch] Loaded cached embeddings for {len(self.product_texts)} products")
                    self._loaded = True
                    return
            except Exception:
                pass  # Cache invalid, recompute

        # Compute embeddings
        print(f"[SemanticSearch] Computing embeddings for {len(self.product_texts)} products...")
        self.product_embeddings = self.model.encode(
            self.product_texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        # Cache to disk
        with open(self.cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.product_embeddings,
                'n_products': len(self.product_texts)
            }, f)

        print("[SemanticSearch] Embeddings computed and cached.")
        self._loaded = True

    def _build_product_corpus(self):
        """Build text corpus from product data for embedding."""
        products_df = self.recommender.products
        self.product_texts = []
        self.product_records = []

        for _, row in products_df.iterrows():
            # Combine name, brand, and content for a rich text representation
            name = str(row.get('name', ''))
            brand = str(row.get('brand', ''))
            content = str(row.get('content', ''))

            # Truncate reviews to keep embedding focused
            reviews_snippet = str(row.get('all_reviews', ''))[:500]

            text = f"{name}. Brand: {brand}. {content}. {reviews_snippet}"
            self.product_texts.append(text)

            # Store product record for easy retrieval
            try:
                price_val = float(row.get('price', 0))
            except (ValueError, TypeError):
                price_val = 0.0
            try:
                rating_val = float(row.get('avg_rating', 0))
            except (ValueError, TypeError):
                rating_val = 0.0
            try:
                review_count_val = int(float(row.get('review_count', 0)))
            except (ValueError, TypeError):
                review_count_val = 0

            self.product_records.append({
                'id': row.get('id', ''),
                'name': name,
                'brand': brand,
                'price': price_val,
                'avg_rating': rating_val,
                'review_count': review_count_val,
                'reviews_snippet': reviews_snippet[:300]
            })

    def search(self, query, top_k=5):
        """
        Search for products matching a natural language query.

        Args:
            query: Natural language search string (e.g. "I want a wifi router")
            top_k: Number of results to return

        Returns:
            List of dicts, each containing product info and similarity score,
            sorted by relevance (highest first).
        """
        if not self._loaded:
            self.load()

        # Encode the query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.product_embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            result = self.product_records[idx].copy()
            result['relevance_score'] = float(similarities[idx])
            results.append(result)

        return results

    def format_results_for_llm(self, results, max_products=3):
        """
        Format search results into a text context block for the LLM prompt.

        Args:
            results: List of search results from self.search()
            max_products: Max number of products to include

        Returns:
            Formatted string with product details for LLM context
        """
        if not results:
            return "No matching products found in our catalog."

        lines = []
        for i, product in enumerate(results[:max_products], 1):
            lines.append(f"Product {i}:")
            lines.append(f"  Name: {product['name']}")
            lines.append(f"  Brand: {product['brand']}")
            lines.append(f"  Price: ${product['price']:.2f}" if product['price'] > 0 else "  Price: N/A")
            lines.append(f"  Rating: {product['avg_rating']:.1f}/5 ({product['review_count']} reviews)")
            lines.append(f"  Relevance: {product['relevance_score']:.1%}")
            if product.get('reviews_snippet'):
                lines.append(f"  Customer feedback: {product['reviews_snippet'][:150]}...")
            lines.append("")

        return "\n".join(lines)
