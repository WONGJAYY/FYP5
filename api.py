"""
FastAPI Backend for E-Commerce Recommender System
Provides REST API endpoints for recommendations, explanations, and chat.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

from recommender import ContentBasedRecommender
from explainer import SHAPExplainer, LIMEExplainer, LLMExplainer

# Initialize FastAPI app
app = FastAPI(
    title="Explainable AI E-Commerce Recommender API",
    description="AI-powered product recommendations with SHAP, LIME & LLM explanations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
recommender = None
explainers = None


def get_recommender():
    """Get or initialize the recommender."""
    global recommender
    if recommender is None:
        recommender = ContentBasedRecommender()
        recommender.load()
    return recommender


def get_explainers():
    """Get or initialize the explainers."""
    global explainers
    if explainers is None:
        rec = get_recommender()
        explainers = {
            'shap': SHAPExplainer(rec),
            'lime': LIMEExplainer(rec),
            'llm': LLMExplainer(rec)
        }
    return explainers


# Pydantic models
class ChatRequest(BaseModel):
    message: str
    product_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str


class ProductResponse(BaseModel):
    id: str
    name: str
    brand: Optional[str]
    price: Optional[float]
    avg_rating: float
    review_count: int


class RecommendationResponse(BaseModel):
    product: dict
    similarity_score: float
    rank: int


class ExplanationResponse(BaseModel):
    explanation_type: str
    data: dict


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Explainable AI E-Commerce Recommender",
        "version": "1.0.0"
    }


@app.get("/products", response_model=List[dict])
async def get_products(limit: int = 100, offset: int = 0):
    """Get list of all products."""
    try:
        rec = get_recommender()
        products = rec.get_all_products()
        return products[offset:offset + limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/{product_id}")
async def get_product(product_id: str):
    """Get a specific product by ID."""
    try:
        rec = get_recommender()
        product = rec.get_product_by_id(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        return product
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/{product_id}")
async def get_recommendations(product_id: str, n: int = 5):
    """Get product recommendations."""
    try:
        rec = get_recommender()
        recommendations = rec.recommend(product_id, n_recommendations=n)
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found")
        return recommendations
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/shap/{source_id}/{target_id}")
async def get_shap_explanation(source_id: str, target_id: str):
    """Get SHAP-based explanation for a recommendation."""
    try:
        exps = get_explainers()
        explanation = exps['shap'].explain_recommendation(source_id, target_id)
        if not explanation:
            raise HTTPException(status_code=404, detail="Unable to generate explanation")
        return explanation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/lime/{source_id}/{target_id}")
async def get_lime_explanation(source_id: str, target_id: str):
    """Get LIME-based explanation for a recommendation."""
    try:
        exps = get_explainers()
        explanation = exps['lime'].explain_recommendation(source_id, target_id)
        if not explanation:
            raise HTTPException(status_code=404, detail="Unable to generate explanation")
        return explanation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/llm/{source_id}/{target_id}")
async def get_llm_explanation(source_id: str, target_id: str):
    """Get LLM-based natural language explanation for a recommendation."""
    try:
        exps = get_explainers()
        explanation = exps['llm'].explain_recommendation(source_id, target_id)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reviews/summary/{product_id}")
async def get_review_summary(product_id: str):
    """Get AI-generated review summary for a product."""
    try:
        exps = get_explainers()
        summary = exps['llm'].summarize_reviews(product_id)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the AI assistant."""
    try:
        exps = get_explainers()
        
        product_context = None
        if request.product_id:
            rec = get_recommender()
            product_context = rec.get_product_by_id(request.product_id)
        
        response = exps['llm'].chat(request.message, product_context)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        rec = get_recommender()
        products = rec.get_all_products()
        
        total_reviews = sum(p.get('review_count', 0) for p in products)
        avg_rating = sum(p.get('avg_rating', 0) for p in products) / len(products) if products else 0
        
        return {
            "total_products": len(products),
            "total_reviews": int(total_reviews),
            "average_rating": round(avg_rating, 2),
            "explainability_methods": ["SHAP", "LIME", "LLM"],
            "status": "active"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
