import os
import pandas as pd
from recommender import ContentBasedRecommender
from evaluator import RecommenderEvaluator

def evaluate_dataset(data_dir):
    rec = ContentBasedRecommender(data_dir=data_dir)
    rec.load()
    ev = RecommenderEvaluator(rec, data_dir=data_dir)
    
    rating_eval = ev.evaluate_rating_prediction(test_size=0.2, n_neighbors=5)
    ranking_eval = ev.evaluate_ranking_metrics(n_samples=100, k=5)
    sim_eval = ev.evaluate_similarity_quality(n_samples=50, n_recommendations=5)
    
    total_reviews = int(sum(p.get('review_count', 0) for p in rec.get_all_products()))
    avg_rating = pd.to_numeric(rec.products['avg_rating'], errors='coerce').fillna(0).mean()
    rated_products = len(rec.products[pd.to_numeric(rec.products['avg_rating'], errors='coerce').fillna(0) > 0])
    
    return {
        'rec': rec,
        'rating': rating_eval,
        'ranking': ranking_eval,
        'similarity': sim_eval,
        'total_products': len(rec.products),
        'total_reviews': total_reviews,
        'avg_rating': avg_rating,
        'rated_products': rated_products
    }

def main():
    print("Running cross-dataset comparison evaluation. Please wait...")
    eco = evaluate_dataset("processed_data")
    ama = evaluate_dataset("processed_data_amazon")
    
    print("\n" + "="*80)
    print(f"{'CROSS-DATASET COMPARISON SYSTEM REPORT':^80}")
    print("="*80)
    
    print("\n1. DATASET CHARACTERISTICS")
    print("-" * 80)
    print(f"{'Metric':<24} | {'E-Commerce (7817_1.csv)':<24} | {'Amazon Sales (amazon.csv)':<24}")
    print("-" * 80)
    print(f"{'Total Products':<24} | {eco['total_products']:<24,} | {ama['total_products']:<24,}")
    print(f"{'Total Reviews':<24} | {eco['total_reviews']:<24,} | {ama['total_reviews']:<24,}")
    print(f"{'Average Rating':<24} | {eco['avg_rating']:<24.4f} | {ama['avg_rating']:<24.4f}")
    eco_rated_pct = (eco['rated_products']/eco['total_products']) * 100
    ama_rated_pct = (ama['rated_products']/ama['total_products']) * 100
    eco_rated_str = f"{eco['rated_products']} ({eco_rated_pct:.2f}%)"
    ama_rated_str = f"{ama['rated_products']} ({ama_rated_pct:.2f}%)"
    print(f"{'Products with Ratings':<24} | {eco_rated_str:<24} | {ama_rated_str:<24}")
    print("-" * 80)
    
    print("\n2. RATING PREDICTION METRICS (RMSE & MAE)")
    print("-" * 80)
    print(f"{'':<20} | {'E-Commerce Products':<20} | {'Amazon Sales Dataset':<20}")
    print(f"{'Model / Approach':<20} | {'RMSE':<9} | {'MAE':<8} | {'RMSE':<9} | {'MAE':<8}")
    print("-" * 80)
    
    eco_comps = eco['rating']['comparisons']
    ama_comps = ama['rating']['comparisons']
    
    for model in ['Baseline', 'CF', 'Proposed XAI Model']:
        e_rmse = eco_comps.get(model, {}).get('rmse', 0)
        e_mae = eco_comps.get(model, {}).get('mae', 0)
        a_rmse = ama_comps.get(model, {}).get('rmse', 0)
        a_mae = ama_comps.get(model, {}).get('mae', 0)
        print(f"{model:<20} | {e_rmse:<9.4f} | {e_mae:<8.4f} | {a_rmse:<9.4f} | {a_mae:<8.4f}")
    print("-" * 80)
    
    print("\n3. RECOMMENDATION QUALITY & RANKING METRICS")
    print("-" * 80)
    print(f"{'Metric':<24} | {'E-Commerce Products (K=5)':<24} | {'Amazon Sales Dataset (K=5)':<24}")
    print("-" * 80)
    print(f"{'Precision@5':<24} | {eco['ranking']['precision@k']:<24.4f} | {ama['ranking']['precision@k']:<24.4f}")
    print(f"{'NDCG@5':<24} | {eco['ranking']['ndcg@k']:<24.4f} | {ama['ranking']['ndcg@k']:<24.4f}")
    eco_cov_str = f"{eco['ranking']['coverage']:.2%}"
    ama_cov_str = f"{ama['ranking']['coverage']:.2%}"
    eco_bc_str = f"{eco['similarity']['avg_brand_consistency']:.2%}"
    ama_bc_str = f"{ama['similarity']['avg_brand_consistency']:.2%}"
    print(f"{'Catalog Coverage':<24} | {eco_cov_str:<24} | {ama_cov_str:<24}")
    print(f"{'Brand Consistency':<24} | {eco_bc_str:<24} | {ama_bc_str:<24}")
    print(f"{'Average Similarity':<24} | {eco['similarity']['avg_similarity_score']:<24.4f} | {ama['similarity']['avg_similarity_score']:<24.4f}")
    print("="*80)

if __name__ == '__main__':
    main()