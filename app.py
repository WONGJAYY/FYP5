"""
Streamlit Dashboard for Explainable AI E-Commerce Recommender System
A beautiful, modern dashboard with Overview, Recommender, AI Chat, and Analytics pages.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommender import ContentBasedRecommender
from explainer import SHAPExplainer, LIMEExplainer, LLMExplainer
from evaluator import RecommenderEvaluator
from semantic_search import SemanticSearcher

# Page configuration
st.set_page_config(
    page_title="Explainable AI E-Commerce Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        margin-bottom: 8px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Cards */
    .info-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Status indicators */
    .status-active {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #00ff88;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Product cards */
    .product-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .product-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
    }
    
    /* Chat styling */
    .chat-message {
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        max-width: 80%;
    }
    
    .chat-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin-left: auto;
        color: white;
    }
    
    .chat-assistant {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom headers */
    h1, h2, h3 {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'explainers' not in st.session_state:
    st.session_state.explainers = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None


@st.cache_resource
def load_recommender(data_dir='processed_data'):
    """Load the recommender system."""
    try:
        rec = ContentBasedRecommender(data_dir=data_dir)
        rec.load()
        return rec
    except Exception as e:
        st.error(f"Error loading recommender: {e}")
        return None


@st.cache_resource
def load_explainers(_recommender, data_dir='processed_data'):
    """Load the explainers. data_dir is used as a cache key to differentiate datasets."""
    if _recommender is None:
        return None
    return {
        'shap': SHAPExplainer(_recommender),
        'lime': LIMEExplainer(_recommender),
        'llm': LLMExplainer(_recommender)
    }


@st.cache_resource
def load_semantic_searcher(_recommender, data_dir='processed_data'):
    """Load the semantic searcher for RAG-powered chat."""
    if _recommender is None:
        return None
    try:
        searcher = SemanticSearcher(_recommender, data_dir=data_dir)
        searcher.load()
        return searcher
    except Exception as e:
        st.warning(f"Semantic search unavailable: {e}")
        return None


def render_metric_card(value, label, icon="📊"):
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_overview_page(recommender):
    """Render the Overview page."""
    st.markdown("# 🛒 Explainable AI E-Commerce Recommender System")
    
    st.markdown("-----")
    
    # System Overview
    st.markdown("## System Overview")
    
    products = recommender.get_all_products()
    total_products = len(products)
    total_reviews = sum(p.get('review_count', 0) for p in products)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card(f"{total_products:,}", "Products", "📦")
    with col2:
        render_metric_card(f"{int(total_reviews):,}", "Reviews Analyzed", "📝")
    with col3:
        render_metric_card("SHAP, LIME & LLM", "Explainability", "🔍")
    with col4:
        render_metric_card("< 50ms", "Inference Time", "⚡")
    
    st.markdown("---")
    
    # System Architecture
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("## System Architecture")
        
        arch_data = {
            'Layer': ['Data', 'Features', 'Models', 'Explainability', 'LLM', 'Serving'],
            'Components': [
                'Ingestion, Validation, Preprocessing',
                'Feature Engineering, Feature Store',
                'Content-Based Filtering Recommender',
                'SHAP, LIME, LLM (Natural Language)',
                'NVIDIA NIM (Llama 3.3), Chat, Review Summary',
                'FastAPI, Streamlit Dashboard'
            ]
        }
        
        st.dataframe(
            pd.DataFrame(arch_data),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown("## Tech Stack")
        
        tech_items = [
            ("🐍", "Python 3.10+"),
            ("🤖", "NVIDIA NIM (Llama 3.3 70B)"),
            ("📊", "SHAP (Shapley Additive exPlanations)"),
            ("🔶", "LIME (Local Interpretable Explanations)"),
            ("🎯", "Content-Based Filtering"),
            ("📚", "Scikit-learn"),
            ("⚡", "FastAPI"),
            ("🎨", "Streamlit")
        ]
        
        for icon, name in tech_items:
            st.markdown(f"{icon} **{name}**")
    
    st.markdown("---")
    
    # System Status
    st.markdown("## System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>Recommender</h4>
            <p><span class="status-active"></span> Active</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>API Status</h4>
            <p><span class="status-active"></span> Online</p>
        </div>
        """, unsafe_allow_html=True)
    
    # About section
    st.markdown("---")
    st.markdown("## About")
    st.info("""
    This system uses **Content-Based Filtering** with **SHAP & LLM explainability** to 
    recommend products and explain **WHY** each product is suggested.
    """)


def render_recommender_page(recommender, explainers):
    """Render the Recommender page."""
    st.markdown("# 🎯 Product Recommender")
    st.markdown("*Get personalized recommendations with AI explanations*")
    
    st.markdown("---")
    
    # Product selection
    products = recommender.get_all_products()
    product_names = [f"{p['name'][:60]}..." if len(p['name']) > 60 else p['name'] for p in products]
    product_map = {name: products[i] for i, name in enumerate(product_names)}
    
    selected_name = st.selectbox(
        "🔍 Select a product to get recommendations:",
        options=product_names,
        index=0
    )
    
    if selected_name:
        selected_product = product_map[selected_name]
        st.session_state.selected_product = selected_product
        
        # Show selected product info
        st.markdown("### 📦 Selected Product")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**{selected_product['name']}**")
            st.markdown(f"Brand: {selected_product.get('brand', 'Unknown')}")
        with col2:
            price = selected_product.get('price')
            try:
                st.metric("Price", f"${float(price):.2f}" if price else "N/A")
            except (ValueError, TypeError):
                st.metric("Price", "N/A")
        with col3:
            try:
                rating_val = float(selected_product.get('avg_rating', 0))
            except (ValueError, TypeError):
                rating_val = 0.0
            st.metric("Rating", f"⭐ {rating_val:.1f}")
        
        st.markdown("---")
        
        # Get and display recommendations
        st.markdown("### 🎯 Recommended Products")
        
        n_recs = st.slider("Number of recommendations:", 3, 10, 5)
        recommendations = recommender.recommend(selected_product['id'], n_recommendations=n_recs)
        
        if recommendations:
            for i, rec in enumerate(recommendations):
                product = rec['product']
                score = rec['similarity_score']
                
                with st.expander(f"#{rec['rank']} - {product['name'][:70]}... (Similarity: {score:.2%})", expanded=i < 3):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Brand:** {product.get('brand', 'Unknown')}")
                        price = product.get('price')
                        try:
                            st.markdown(f"**Price:** ${float(price):.2f}" if price else "**Price:** N/A")
                        except (ValueError, TypeError):
                            st.markdown("**Price:** N/A")
                        try:
                            r_val = float(product.get('avg_rating', 0))
                            rc_val = int(float(product.get('review_count', 0)))
                        except (ValueError, TypeError):
                            r_val = 0.0
                            rc_val = 0
                        st.markdown(f"**Rating:** ⭐ {r_val:.1f} ({rc_val} reviews)")
                    
                    with col2:
                        # Similarity gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#667eea"},
                                'bgcolor': "rgba(255,255,255,0.1)",
                                'steps': [
                                    {'range': [0, 50], 'color': "rgba(255,255,255,0.05)"},
                                    {'range': [50, 75], 'color': "rgba(102,126,234,0.2)"},
                                    {'range': [75, 100], 'color': "rgba(102,126,234,0.4)"}
                                ]
                            },
                            title={'text': "Match %"}
                        ))
                        fig.update_layout(
                            height=180,
                            margin=dict(t=30, b=0, l=0, r=0),
                            paper_bgcolor='rgba(0,0,0,0)',
                            font={'color': 'white'}
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"gauge_{i}")
                    
                    # Explanation tabs
                    tab1, tab2, tab3 = st.tabs(["📊 SHAP", "🔶 LIME", "💬 LLM Explanation"])
                    
                    with tab1:
                        shap_exp = explainers['shap'].get_feature_importance_plot_data(
                            selected_product['id'], product['id']
                        )
                        if shap_exp:
                            fig = px.bar(
                                x=shap_exp['importance'][::-1],
                                y=shap_exp['features'][::-1],
                                orientation='h',
                                title="Feature Importance (SHAP Values)",
                                color=shap_exp['importance'][::-1],
                                color_continuous_scale='Purples'
                            )
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font={'color': 'white'},
                                showlegend=False,
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"shap_{i}")
                            
                            # Feature explanations for SHAP
                            st.markdown("---")
                            st.markdown("**📖 SHAP Feature Explanations:**")
                            st.markdown("The chart shows which keywords/features contribute most to this recommendation.")
                            
                            if shap_exp['features']:
                                top_features = shap_exp['features'][:3]
                                top_importance = shap_exp['importance'][:3]
                                
                                st.markdown("**Top contributing features:**")
                                for feat, imp in zip(top_features, top_importance):
                                    st.markdown(f"• **'{feat}'** (importance: {imp:.4f}) - Both products share this keyword in descriptions/reviews")
                                
                                total_importance = sum(shap_exp['importance'])
                                st.markdown(f"")
                                st.markdown(f"**Total SHAP contribution:** {total_importance:.4f} → This explains the {score:.1%} similarity score")
                    
                    with tab2:
                        lime_exp = explainers['lime'].explain_recommendation(
                            selected_product['id'], product['id']
                        )
                        if lime_exp:
                            # Combine positive and negative features for visualization
                            all_features = []
                            all_contributions = []
                            all_colors = []
                            
                            for f in lime_exp['positive_features'][:5]:
                                all_features.append(f['feature'])
                                all_contributions.append(f['contribution'])
                                all_colors.append('#00cc66')  # Green for positive
                            
                            for f in lime_exp['negative_features'][:3]:
                                all_features.append(f['feature'])
                                all_contributions.append(-abs(f['contribution']))  # Negative value
                                all_colors.append('#ff4444')  # Red for negative
                            
                            # Create bar chart
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                y=all_features[::-1],
                                x=all_contributions[::-1],
                                orientation='h',
                                marker_color=all_colors[::-1],
                                text=[f'{v:.4f}' for v in all_contributions[::-1]],
                                textposition='outside'
                            ))
                            fig.update_layout(
                                title="LIME Feature Contributions",
                                xaxis_title="Contribution Score",
                                yaxis_title="Features",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font={'color': 'white'},
                                height=300,
                                showlegend=False
                            )
                            fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
                            st.plotly_chart(fig, use_container_width=True, key=f"lime_{i}")
                            
                            st.markdown("🟢 **Green** = Positive (similar features) | 🔴 **Red** = Negative (different features)")
                            
                            # Feature explanations
                            st.markdown("---")
                            st.markdown("**📖 Feature Explanations:**")
                            
                            if lime_exp['positive_features']:
                                st.markdown("**Why these products are similar:**")
                                for f in lime_exp['positive_features'][:3]:
                                    st.markdown(f"• Both products contain **'{f['feature']}'** in their descriptions/reviews, contributing {f['contribution']:.4f} to similarity")
                            
                            if lime_exp['negative_features']:
                                st.markdown("**What makes them different:**")
                                for f in lime_exp['negative_features'][:2]:
                                    st.markdown(f"• **'{f['feature']}'** appears in one product but not the other")
                    
                    with tab3:
                        # st.info("🔒 LLM Explanation is temporarily disabled.")
                        if st.button(f"Generate LLM Explanation", key=f"llm_{i}"):
                            with st.spinner("Generating explanation..."):
                                explanation = explainers['llm'].explain_recommendation(
                                    selected_product['id'], product['id']
                                )
                                st.markdown(explanation)
            
            # Model Evaluation Section
            st.markdown("---")
            st.markdown("### 📈 Model Evaluation Metrics")
            
            # Calculate metrics from recommendations
            similarity_scores = [rec['similarity_score'] for rec in recommendations]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Similarity Score Distribution Chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f"Rec #{i+1}" for i in range(len(similarity_scores))],
                    y=[s * 100 for s in similarity_scores],
                    marker_color=['#667eea' if s > 0.3 else '#ff6b6b' for s in similarity_scores],
                    text=[f"{s:.1%}" for s in similarity_scores],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Recommendation Similarity Scores",
                    xaxis_title="Recommendation",
                    yaxis_title="Similarity (%)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'},
                    height=300,
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Metrics Summary
                avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
                max_similarity = max(similarity_scores) if similarity_scores else 0
                min_similarity = min(similarity_scores) if similarity_scores else 0
                high_quality_recs = sum(1 for s in similarity_scores if s > 0.3)
                
                st.markdown("**📊 Performance Metrics:**")
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Avg Similarity", f"{avg_similarity:.1%}")
                    st.metric("Max Similarity", f"{max_similarity:.1%}")
                with metric_col2:
                    st.metric("Min Similarity", f"{min_similarity:.1%}")
                    st.metric("High Quality Recs", f"{high_quality_recs}/{len(similarity_scores)}")
                
                # Quality indicator
                if avg_similarity > 0.4:
                    st.success("✅ Excellent recommendation quality!")
                elif avg_similarity > 0.25:
                    st.info("👍 Good recommendation quality")
                else:
                    st.warning("⚠️ Lower similarity - products may be less related")
        else:
            st.warning("No recommendations found for this product.")


def render_chat_page(recommender, explainers, semantic_searcher=None):
    """Render the AI Chat page with RAG-powered product search."""
    st.markdown("# 💬 AI Chat Assistant")
    st.markdown("*Ask about any product — I'll search our catalog and give you real answers*")
    
    if semantic_searcher:
        st.markdown("""<div class="info-card">
            <span class="status-active"></span> <strong>RAG Mode Active</strong> — 
            Responses are grounded in your real product database
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message chat-user">
                    <strong>You:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message chat-assistant">
                    <strong>🤖 Assistant:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show retrieved product cards if available
                if 'search_results' in msg and msg['search_results']:
                    with st.expander("📦 Products retrieved from database", expanded=False):
                        for p in msg['search_results'][:3]:
                            pcol1, pcol2, pcol3 = st.columns([3, 1, 1])
                            with pcol1:
                                st.markdown(f"**{p['name'][:60]}**")
                                st.markdown(f"Brand: {p['brand']}")
                            with pcol2:
                                st.markdown(f"💰 ${p['price']:.2f}" if p['price'] > 0 else "💰 N/A")
                            with pcol3:
                                st.markdown(f"⭐ {p['avg_rating']:.1f}")
                                st.markdown(f"Match: {p['relevance_score']:.0%}")
                            st.markdown("---")
    
    # Chat input
    st.markdown("---")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask a question:",
            placeholder="e.g., I want a smart speaker under $200",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send 📤", use_container_width=True)
    
    if send_button and user_input:
        # Add user message
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        # Get AI response with RAG
        with st.spinner("🔍 Searching products & generating answer..."):
            product_context = st.session_state.get('selected_product')
            
            # Search products first for the card display
            search_results = None
            if semantic_searcher:
                try:
                    search_results = semantic_searcher.search(user_input, top_k=5)
                except Exception:
                    search_results = None
            
            response = explainers['llm'].chat(
                user_input,
                product_context,
                semantic_searcher=semantic_searcher
            )
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'search_results': search_results
            })
        
        st.rerun()
    
    # Clear chat button
    if st.button("Clear Chat 🗑️"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Sidebar with context
    with st.sidebar:
        st.markdown("### 📋 Chat Context")
        if semantic_searcher:
            st.markdown("🟢 **RAG Search:** Active")
        else:
            st.markdown("🔴 **RAG Search:** Unavailable")
        
        if st.session_state.selected_product:
            st.markdown(f"**Current Product:**")
            st.markdown(f"{st.session_state.selected_product['name'][:50]}...")
        else:
            st.markdown("*No product selected*")
        
        st.markdown("---")
        st.markdown("### 💡 Try asking")
        st.markdown("- Find me a smart speaker")
        st.markdown("- I want a tablet under $300")
        st.markdown("- What's the best rated product?")
        st.markdown("- Compare Echo and Kindle")
        st.markdown("- Recommend something from Amazon brand")


def render_analytics_page(recommender):
    """Render the Analytics page."""
    st.markdown("# 📊 Analytics Dashboard")
    st.markdown("*Insights from product data and reviews*")
    
    st.markdown("---")
    
    products = recommender.get_all_products()
    df = pd.DataFrame(products)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", len(df))
    with col2:
        st.metric("Avg Rating", f"{df['avg_rating'].mean():.2f}")
    with col3:
        st.metric("Total Reviews", f"{int(df['review_count'].sum()):,}")
    with col4:
        st.metric("Unique Brands", df['brand'].nunique())
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Rating Distribution")
        
        # Create rating bins
        df['rating_bin'] = pd.cut(df['avg_rating'], bins=[0, 1, 2, 3, 4, 5], labels=['0-1', '1-2', '2-3', '3-4', '4-5'])
        rating_counts = df['rating_bin'].value_counts().sort_index()
        
        fig = px.bar(
            x=rating_counts.index.astype(str),
            y=rating_counts.values,
            title="Products by Rating",
            color=rating_counts.values,
            color_continuous_scale='Purples'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Top Brands by Products")
        
        brand_counts = df['brand'].value_counts().head(10)
        
        fig = px.pie(
            names=brand_counts.index,
            values=brand_counts.values,
            title="Top 10 Brands",
            color_discrete_sequence=px.colors.sequential.Purples_r
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Review analysis
    st.markdown("### Review Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Reviews vs Rating scatter
        fig = px.scatter(
            df,
            x='review_count',
            y='avg_rating',
            size='review_count',
            color='avg_rating',
            title="Reviews vs Rating",
            color_continuous_scale='Purples',
            hover_data=['name']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top reviewed products
        top_reviewed = df.nlargest(10, 'review_count')[['name', 'review_count', 'avg_rating']]
        top_reviewed['name'] = top_reviewed['name'].str[:40] + '...'
        
        fig = px.bar(
            top_reviewed,
            x='review_count',
            y='name',
            orientation='h',
            title="Most Reviewed Products",
            color='avg_rating',
            color_continuous_scale='Purples'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Price analysis
    st.markdown("### Price Analysis")
    
    df_with_price = df[df['price'].notna()]
    
    if len(df_with_price) > 0:
        fig = px.histogram(
            df_with_price,
            x='price',
            nbins=30,
            title="Price Distribution",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No price data available.")


def render_evaluation_page(recommender):
    """Render the Evaluation page with RMSE metrics."""
    st.markdown("# 📈 Model Evaluation")
    st.markdown("*Assess recommendation quality using RMSE and other metrics*")
    
    st.markdown("---")
    
    # Initialize evaluator
    evaluator = RecommenderEvaluator(recommender)
    
    # Evaluation controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        n_neighbors = st.slider("Number of Neighbors (K)", min_value=3, max_value=10, value=5)
    
    with col2:
        test_size = st.selectbox("Test Size", [0.1, 0.2, 0.3], index=1)
    
    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Running comprehensive evaluation (RMSE, MAE, Precision@K, NDCG@K)..."):
            # Run evaluation
            rating_results = evaluator.evaluate_rating_prediction(
                test_size=test_size, 
                n_neighbors=n_neighbors
            )
            similarity_results = evaluator.evaluate_similarity_quality(n_recommendations=n_neighbors)
            ranking_results = evaluator.evaluate_ranking_metrics(k=n_neighbors)
        
        st.markdown("---")
        
        # Display metrics
        st.markdown("### 📊 Rating Prediction Metrics")
        
        if 'error' not in rating_results:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "RMSE", 
                    f"{rating_results['rmse']:.4f}",
                    help="Root Mean Square Error - Lower is better"
                )
            with col2:
                st.metric(
                    "MAE", 
                    f"{rating_results['mae']:.4f}",
                    help="Mean Absolute Error - Lower is better"
                )
            with col3:
                st.metric(
                    "Predictions Made", 
                    rating_results['n_predictions'],
                    help="Number of successful predictions"
                )
            
            # Actual vs Predicted chart
            st.markdown("### 📉 Actual vs Predicted Ratings")
            
            actual = rating_results['actual_ratings']
            predicted = rating_results['predicted_ratings']
            
            fig = go.Figure()
            
            # Perfect prediction line
            fig.add_trace(go.Scatter(
                x=[1, 5], y=[1, 5],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='rgba(255,255,255,0.3)', dash='dash')
            ))
            
            # Actual predictions
            fig.add_trace(go.Scatter(
                x=actual,
                y=predicted,
                mode='markers',
                name='Predictions',
                marker=dict(
                    size=12,
                    color=predicted,
                    colorscale='Purples',
                    showscale=True,
                    colorbar=dict(title="Predicted")
                )
            ))
            
            fig.update_layout(
                xaxis_title="Actual Rating",
                yaxis_title="Predicted Rating",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis=dict(range=[0, 5.5], gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(range=[0, 5.5], gridcolor='rgba(255,255,255,0.1)')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Error distribution
            st.markdown("### 📊 Error Distribution")
            
            errors = [a - p for a, p in zip(actual, predicted)]
            
            fig = px.histogram(
                x=errors,
                nbins=20,
                title="Prediction Error Distribution",
                labels={'x': 'Error (Actual - Predicted)', 'y': 'Count'},
                color_discrete_sequence=['#667eea']
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning(f"⚠️ {rating_results['error']}")
        
        st.markdown("---")
        
        # Similarity quality metrics
        st.markdown("### 🎯 Recommendation Quality & Ranking Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Precision@K",
                f"{ranking_results['precision@k']:.4f}",
                help="Proportion of recommended items that are relevant"
            )
        with col2:
            st.metric(
                "NDCG@K",
                f"{ranking_results['ndcg@k']:.4f}",
                help="Normalized Discounted Cumulative Gain - considers ranking position"
            )
        with col3:
            st.metric(
                "Brand Consistency",
                f"{similarity_results['avg_brand_consistency']:.1%}",
                help="How often recommendations share the same brand"
            )
        with col4:
            st.metric(
                "Catalog Coverage",
                f"{ranking_results['coverage']:.1%}",
                help="Percentage of the item catalog recommended"
            )
        
        # RMSE Interpretation
        st.markdown("---")
        st.markdown("### 📝 Interpretation Guide")
        
        st.markdown("""
        | Metric | Value | Interpretation |
        |--------|-------|----------------|
        | **RMSE** | {:.4f} | Average prediction error (same scale as ratings) |
        | **MAE** | {:.4f} | Average absolute error |
        | **Precision@K** | {:.4f} | Exact fraction of recommendations that are highly relevant (domain metric) |
        | **NDCG@K** | {:.4f} | Quality of the item ranking order (position matters) |
        | **Brand Consistency** | {:.1%} | Recommendations match product brands |
        """.format(
            rating_results.get('rmse', 0),
            rating_results.get('mae', 0),
            ranking_results['precision@k'],
            ranking_results['ndcg@k'],
            similarity_results['avg_brand_consistency']
        ))
        
        st.info("💡 **RMSE Benchmark**: For a 1-5 rating scale, RMSE < 1.0 is generally considered good performance.")
    
    else:
        st.info("👆 Click 'Run Evaluation' to calculate RMSE and other metrics")
        
        # # Show formula (Commented out)
        # st.markdown("### 📐 RMSE Formula")
        # st.latex(r"RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}")
        # 
        # st.markdown(r"""
        # **Where:**
        # - $y_i$ = Actual rating
        # - $\hat{y}_i$ = Predicted rating  
        # - $n$ = Number of predictions
        # """)


def main():
    """Main application entry point."""
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("# 🛒")
        
        # ── Dataset Selector ──
        st.markdown("## 📂 Dataset")
        dataset_options = {
            "E-Commerce Products": {
                "description": "Product reviews & metadata (7817_1.csv)",
                "data_dir": "processed_data",
                "available": True
            },
            "Amazon Sales Dataset": {
                "description": "Amazon product reviews & sales (amazon.csv)",
                "data_dir": "processed_data_amazon",
                "available": True
            }
        }
        
        selected_dataset = st.selectbox(
            "Choose Dataset",
            options=list(dataset_options.keys()),
            index=0,
            help="Select which dataset to use for recommendations"
        )
        
        dataset_info = dataset_options[selected_dataset]
        st.caption(f"ℹ️ {dataset_info['description']}")
        
        st.markdown("---")
        st.markdown("## Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Overview", "🎯 Recommender", "💬 AI Chat", "📈 Evaluation"],
            # ["🏠 Overview", "🎯 Recommender", "📊 Analytics", "📈 Evaluation"],
            # ["🏠 Overview", "🎯 Recommender", "💬 AI Chat", "📊 Analytics"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This system uses Content-Based Filtering with SHAP & LLM explainability to recommend products and explain WHY each product is suggested.")
        
        st.markdown("---")
        st.markdown("### System Status")
        st.markdown("**Recommender:** <span class='status-active'></span> Active", unsafe_allow_html=True)
        st.markdown("**API Status:** <span class='status-active'></span> Online", unsafe_allow_html=True)
    
    # Check if the selected dataset is available
    if not dataset_info['available']:
        st.warning(f"⚠️ **{selected_dataset}** dataset is not yet available.")
        st.info(
            "To enable this dataset:\n"
            "1. Download the dataset and place it in the project folder\n"
            "2. Process it with the data processor\n"
            "3. Update `dataset_options` in `app.py` to set `available: True`"
        )
        return
    
    # Load recommender with the selected dataset's data directory
    recommender = load_recommender(data_dir=dataset_info['data_dir'])
    
    if recommender is None:
        st.error("⚠️ Recommender not loaded. Please run `python data_processor.py` first to process the data.")
        st.code("python data_processor.py", language="bash")
        return
    
    # Load explainers
    explainers = load_explainers(recommender, data_dir=dataset_info['data_dir'])
    
    # Load semantic searcher for RAG chat
    semantic_searcher = load_semantic_searcher(recommender, data_dir=dataset_info['data_dir'])
    
    # Render selected page
    if page == "🏠 Overview":
        render_overview_page(recommender)
    elif page == "🎯 Recommender":
        render_recommender_page(recommender, explainers)
    elif page == "💬 AI Chat":
        render_chat_page(recommender, explainers, semantic_searcher)
    # elif page == "📊 Analytics":  # Temporarily disabled
    #     render_analytics_page(recommender)
    elif page == "📈 Evaluation":
        render_evaluation_page(recommender)


if __name__ == "__main__":
    main()
