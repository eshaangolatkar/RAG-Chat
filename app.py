# Updated app.py with enhanced location handling and RAG improvements

import os
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import streamlit_authenticator as stauth
from typing import List, Tuple, Dict, Optional
from rank_bm25 import BM25Okapi

# Authentication Configuration
# Simple credentials for testing
credentials = {
    'usernames': {
        'admin': {'name': 'Admin', 'password': 'admin123'},
        'demo': {'name': 'Demo User', 'password': 'demo123'}
    }
}

# Hash the passwords
hashed_passwords = stauth.Hasher(['admin123', 'demo123']).generate()
credentials['usernames']['admin']['password'] = hashed_passwords[0]
credentials['usernames']['demo']['password'] = hashed_passwords[1]

# Set page config first
st.set_page_config(page_title="Enhanced RAG People Finder", layout="wide")

# Initialize authenticator
authenticator = stauth.Authenticate(credentials, 'rag_app', 'secret_key', 30)

# Authentication logic
name, authentication_status, username = authenticator.login('Login', 'main')

# Handle authentication states
if authentication_status == False:
    st.error('Username/password is incorrect')
    st.stop()
elif authentication_status == None:
    st.warning('Please enter your username and password')
    st.info('Use demo credentials: username="demo", password="demo"')
    st.stop()

# Only execute the rest of the app if authenticated
if authentication_status == True:
    # Welcome message and logout
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write(f'Welcome *{name}*')
    with col2:
        authenticator.logout('Logout', 'main')

    # Import our enhanced modules (only after authentication)
    try:
        from location_handler import DynamicLocationMatcher, enhanced_location_filtering
        from rag_improvements import EnhancedRAGSystem
    except ImportError as e:
        st.error(f"Required modules not found: {e}")
        st.info("Please ensure location_handler.py and rag_improvements.py are available")
        st.stop()

    # -------- Config ----------
    MODEL_NAME = "all-MiniLM-L6-v2"
    EMB_PATH = "data/emb_combined.npy"
    META_CSV = "data/meta_founders.csv"
    TOP_K = 5

    @st.cache_resource
    def load_model():
        return SentenceTransformer(MODEL_NAME)

    @st.cache_data
    def load_meta():
        if not os.path.exists(META_CSV):
            raise FileNotFoundError(f"{META_CSV} not found. Run indexing.py first.")
        return pd.read_csv(META_CSV)

    @st.cache_resource
    def load_embeddings():
        if not os.path.exists(EMB_PATH):
            raise FileNotFoundError(f"{EMB_PATH} not found. Run indexing.py first.")
        return np.load(EMB_PATH)

    @st.cache_resource
    def get_location_matcher():
        """Initialize the dynamic location matcher"""
        return DynamicLocationMatcher()

    @st.cache_resource
    def get_rag_system():
        """Initialize the enhanced RAG system"""
        return EnhancedRAGSystem()

    def preprocess_locations(df):
        """Enhanced location preprocessing"""
        df["city_clean"] = df["location"].fillna("").apply(
            lambda x: x.split(",")[0].strip().lower() if "," in x else ""
        )
        df["country_clean"] = df["location"].fillna("").apply(
            lambda x: x.split(",")[-1].strip().lower() if "," in x else x.strip().lower()
        )
        return df

    def enhanced_search_pipeline(query: str, model, df, embeddings, 
                               location_matcher, rag_system, k=TOP_K):
        """
        Enhanced search pipeline:
        1. Mandatory location filtering (enhanced)
        2. Advanced RAG on filtered subset
        """
        
        # STAGE 1: Enhanced location filtering
        filtered_df, location_message = enhanced_location_filtering(query, df, location_matcher)
        
        if filtered_df.empty:
            return [], location_message
        
        # STAGE 2: Enhanced RAG on filtered subset
        # Get embeddings for filtered subset
        filtered_indices = filtered_df.index.tolist()
        filtered_embeddings = embeddings[filtered_indices]
        
        # Build RAG indices for filtered data
        rag_system.build_indices(filtered_df, filtered_embeddings)
        
        # Process with advanced RAG
        rag_results = rag_system.process_complex_query(query, k)
        
        # Convert back to original indices and add explanations
        final_results = []
        for doc_id, score, explanation in rag_results:
            original_idx = filtered_df.index[doc_id]
            final_results.append((original_idx, score, explanation))
        
        rag_message = f"✅ Ranked {len(final_results)} results using advanced RAG"
        combined_message = f"{location_message} | {rag_message}"
        
        return final_results, combined_message

    # ---------- Main App UI ----------
    st.title("Enhanced RAG People Finder")
    st.write("**Advanced Features:** Dynamic location matching + Fusion retrieval + Contextual reranking")

    # Load resources with error handling
    try:
        with st.spinner("Loading models and data..."):
            model = load_model()
            df = load_meta()
            df = preprocess_locations(df)
            embeddings = load_embeddings()
            
            # Initialize enhanced components
            location_matcher = get_location_matcher()
            rag_system = get_rag_system()
        
        st.success("✅ System initialized successfully!")
        
    except FileNotFoundError as e:
        st.error(f"❌ Required files not found: {e}")
        st.info("Please run the following commands first:")
        st.code("""
python generate_dataset.py
python indexing.py
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading system: {e}")
        st.stop()

    # Query input
    st.markdown("---")
    query = st.text_input(
        "Enter your query (e.g., 'AI startup in blr', 'healthtech founder in bangalor')",
        key="query_input",
        help="Location is mandatory. Try variations like 'blr', 'bangalore', 'bengaluru'",
        placeholder="Type your search query here..."
    )

    col1, col2, col3 = st.columns([2,1,1])
    with col2:
        top_k = st.slider("Top K", 1, 10, TOP_K, key="topk_slider")
    with col3:
        show_debug = st.checkbox("Debug Info", key="debug_check")

    if st.button("🔍 Enhanced Search", key="search_btn", type="primary", disabled=not query.strip()):
        if not query.strip():
            st.warning("Please enter a search query")
        else:
            with st.spinner("Processing with advanced RAG..."):
                
                results, explanation = enhanced_search_pipeline(
                    query, model, df, embeddings, location_matcher, rag_system, k=top_k
                )
                
                # Display status
                if "❌" in explanation:
                    st.error(explanation)
                elif "ℹ️" in explanation:
                    st.info(explanation)
                else:
                    st.success(explanation)
                
                if not results:
                    st.warning("No results found. Please check your location spelling and try again.")
                    
                    # Show location suggestions
                    if show_debug:
                        st.subheader("💡 Location Suggestions")
                        sample_locations = df['location'].value_counts().head(10)
                        st.write("**Popular locations in database:**")
                        for loc, count in sample_locations.items():
                            city = loc.split(',')[0] if ',' in loc else loc
                            st.write(f"- {city} ({count} founders)")
                else:
                    st.success(f"🎯 Found {len(results)} highly relevant matches:")
                    
                    for i, (idx, score, explanation) in enumerate(results, 1):
                        row = df.iloc[idx]
                        
                        # Enhanced result display
                        st.subheader(f"{i}. {row['founder_name']} — {row['role']} @ {row['company']}")
                        st.caption(f"📍 {row['location']} • 🆔 {row['id']} • 🎯 Score: {score:.3f}")
                        
                        # Show idea
                        st.write(f"**💡 Idea:** {row['idea']}")
                        
                        # Show match explanation
                        if explanation['explanations']:
                            explanation_text = " | ".join(explanation['explanations'])
                            confidence_emoji = "🟢" if explanation['confidence'] > 0.7 else "🟡" if explanation['confidence'] > 0.4 else "🔴"
                            st.caption(f"{confidence_emoji} **Why this matches:** {explanation_text}")
                        
                        # Expandable details
                        with st.expander(f"📋 Full Details for {row['founder_name']}"):
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.write(f"**About:** {row['about']}")
                                st.write(f"**Keywords:** {row['keywords']}")
                                st.write(f"**Stage:** {row['stage']}")
                            
                            with col_b:
                                st.write(f"**Email:** {row.get('email', 'N/A')}")
                                st.write(f"**LinkedIn:** {row.get('linked_in', 'N/A')}")
                                if 'notes' in row and pd.notna(row['notes']):
                                    st.write(f"**Notes:** {row['notes']}")
                            
                            # Debug info for this result
                            if show_debug:
                                st.json({
                                    "match_confidence": explanation['confidence'],
                                    "matched_fields": explanation['matched_fields'],
                                    "all_explanations": explanation['explanations']
                                })
                        
                        st.divider()

    # Debug panel (only shown to authenticated users)
    if show_debug:
        st.sidebar.header("🔧 Debug Information")
        
        if query:
            # Enhanced location extraction debug
            debug_info = location_matcher.debug_location_matching(query)
            
            st.sidebar.write("**Query Analysis:**")
            st.sidebar.write(f"Original: {debug_info['original_query']}")
            st.sidebar.write(f"Extracted: {debug_info['extracted_locations']}")
            
            # Show variations for each location
            for loc, variations in debug_info['location_variations'].items():
                st.sidebar.write(f"**Variations for '{loc}':**")
                st.sidebar.write(variations[:5])  # Show first 5 variations
            
            # Show dataset sample
            st.sidebar.write("**Sample Dataset Locations:**")
            for i, loc in enumerate(debug_info['dataset_sample_locations'][:5], 1):
                st.sidebar.write(f"{i}. {loc}")
            
            # Show matching attempts
            for query_loc, attempts in debug_info['matching_attempts'].items():
                st.sidebar.write(f"**Matching '{query_loc}':**")
                for method, results in attempts.items():
                    if results:
                        st.sidebar.write(f"  {method}: {len(results)} matches")
                        if results:
                            st.sidebar.write(f"    Best: {results[0].get('location', 'N/A')}")
                    else:
                        st.sidebar.write(f"  {method}: 0 matches")
            
            # Query processing debug
            if rag_system and hasattr(rag_system, 'df') and rag_system.df is not None:
                expanded_queries = rag_system.query_expansion(query)
                st.sidebar.write("**Query Expansions:**")
                for i, exp_query in enumerate(expanded_queries[:3], 1):
                    st.sidebar.write(f"{i}. {exp_query}")
        
        # Dataset stats
        st.sidebar.write("**Dataset Statistics:**")
        st.sidebar.metric("Total Records", len(df))
        st.sidebar.metric("Unique Locations", df['location'].nunique())
        
        # Top locations with city names
        st.sidebar.write("**Top Locations:**")
        top_locations = df['location'].value_counts().head(5)
        for loc, count in top_locations.items():
            city = loc.split(',')[0] if ',' in loc else loc
            st.sidebar.write(f"• {city}: {count}")
        
        # Show some sample BLR/Bangalore locations
        bangalore_locations = df[df['location'].str.contains('Bangalore|Bengaluru', case=False, na=False)]
        if not bangalore_locations.empty:
            st.sidebar.write("**Bangalore/BLR Locations in Dataset:**")
            unique_blr = bangalore_locations['location'].unique()[:3]
            for loc in unique_blr:
                st.sidebar.write(f"• {loc}")

    # Footer (only shown to authenticated users)
    st.markdown("---")
    st.markdown("**Enhanced Features:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🌍 Smart Location Matching:**
        - Handles typos (blr → Bangalore)
        - City variations (Bengaluru ↔ Bangalore)
        - Fuzzy string matching
        - Semantic similarity
        """)

    with col2:
        st.markdown("""
        **🤖 Advanced RAG:**
        - Query expansion
        - Hypothetical Document Embedding
        - Fusion retrieval (5 strategies)
        - Contextual reranking
        """)

    with col3:
        st.markdown("""
        **📊 Intelligent Ranking:**
        - Multi-factor scoring
        - Diversity promotion
        - Completeness weighting
        - Explainable matches
        """)

    # Installation instructions (only shown to authenticated users)
    with st.expander("📦 Installation & Requirements"):
        st.code("""
# Install required packages
pip install streamlit sentence-transformers pandas numpy scikit-learn rank-bm25 fuzzywuzzy python-levenshtein faker streamlit-authenticator

# Run the app
streamlit run app.py

# Generate dataset (if needed)
python generate_dataset.py

# Build indices (if needed)  
python indexing.py
        """, language="bash")

