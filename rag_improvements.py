# rag_improvements.py - Enhanced RAG techniques for Streamlit deployment

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import streamlit as st
from typing import List, Tuple, Dict
import re

class EnhancedRAGSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.bm25_index = None
        self.embeddings = None
        self.df = None
        
    def build_indices(self, df: pd.DataFrame, embeddings: np.ndarray):
        """Build BM25 and semantic indices"""
        self.df = df
        self.embeddings = embeddings
        
        # Build BM25 index
        documents = []
        for _, row in df.iterrows():
            doc_text = f"{row['founder_name']} {row['role']} {row['company']} {row['idea']} {row['about']} {row['keywords']} {row['stage']}"
            documents.append(doc_text.lower().split())
        
        self.bm25_index = BM25Okapi(documents)
    
    # 1. Query Transformation Techniques
    def query_expansion(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        expansions = {
            'ai': ['artificial intelligence', 'machine learning', 'ml', 'deep learning'],
            'ml': ['machine learning', 'ai', 'artificial intelligence'],
            'fintech': ['financial technology', 'finance', 'banking', 'payments'],
            'healthtech': ['health technology', 'medical', 'healthcare', 'biotech'],
            'edtech': ['education technology', 'learning', 'education'],
            'startup': ['company', 'business', 'venture'],
            'founder': ['ceo', 'entrepreneur', 'co-founder'],
        }
        
        expanded_queries = [query]
        query_lower = query.lower()
        
        for term, synonyms in expansions.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    expanded_queries.append(expanded_query)
        
        return expanded_queries
    
    def hypothetical_document_embedding(self, query: str) -> str:
        """HyDE: Generate hypothetical answer to improve retrieval"""
        # Simple implementation - create hypothetical founder profile
        query_lower = query.lower()
        
        # Extract key terms
        keywords = []
        if 'ai' in query_lower or 'artificial intelligence' in query_lower:
            keywords.append('AI')
        if 'health' in query_lower or 'medical' in query_lower:
            keywords.append('healthtech')
        if 'finance' in query_lower or 'fintech' in query_lower:
            keywords.append('fintech')
        if 'education' in query_lower or 'edtech' in query_lower:
            keywords.append('edtech')
        
        # Generate hypothetical document
        hypothetical = f"""
        I am a founder with experience in {', '.join(keywords) if keywords else 'technology'}. 
        I have built products that solve real-world problems in this space. 
        My background includes both technical and business expertise.
        I am working on innovative solutions and looking to connect with like-minded people.
        """
        
        return hypothetical.strip()
    
    # 2. Fusion Retrieval
    def fusion_retrieval(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        """Combine multiple retrieval strategies with RRF (Reciprocal Rank Fusion)"""
        if self.df is None or self.embeddings is None or self.bm25_index is None:
            return []
        
        # Strategy 1: Basic semantic search
        semantic_results = self._semantic_search(query, k)
        
        # Strategy 2: Query expansion + semantic search  
        expanded_queries = self.query_expansion(query)
        expansion_results = []
        for exp_query in expanded_queries[:3]:  # Limit to avoid noise
            expansion_results.extend(self._semantic_search(exp_query, k//2))
        
        # Strategy 3: HyDE
        hypothetical_doc = self.hypothetical_document_embedding(query)
        hyde_results = self._semantic_search(hypothetical_doc, k)
        
        # Strategy 4: BM25
        bm25_results = self._bm25_search(query, k)
        
        # Strategy 5: Field-specific search (if query targets specific field)
        field_results = self._field_specific_search(query, k)
        
        # Combine using Reciprocal Rank Fusion
        all_strategies = {
            'semantic': semantic_results,
            'expansion': expansion_results,
            'hyde': hyde_results,
            'bm25': bm25_results,
            'field': field_results
        }
        
        return self._reciprocal_rank_fusion(all_strategies, k)
    
    def _semantic_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Basic semantic search"""
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:k]
        return [(idx, similarities[idx]) for idx in top_indices]
    
    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """BM25 keyword search"""
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(idx, scores[idx]) for idx in top_indices]
    
    def _field_specific_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Search specific fields based on query intent"""
        query_lower = query.lower()
        
        # Determine which field to focus on
        field_weights = {}
        if any(word in query_lower for word in ['name', 'founder', 'who is']):
            field_weights['founder_name'] = 2.0
        if any(word in query_lower for word in ['company', 'startup', 'working at']):
            field_weights['company'] = 2.0
        if any(word in query_lower for word in ['idea', 'building', 'product']):
            field_weights['idea'] = 2.0
        if any(word in query_lower for word in ['background', 'experience', 'about']):
            field_weights['about'] = 2.0
        if any(word in query_lower for word in ['tech', 'keywords', 'domain']):
            field_weights['keywords'] = 2.0
        
        if not field_weights:
            return self._semantic_search(query, k)
        
        # Weight the search based on detected intent
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Apply field-based boosting (simplified for demo)
        boosted_scores = similarities.copy()
        
        top_indices = np.argsort(boosted_scores)[::-1][:k]
        return [(idx, boosted_scores[idx]) for idx in top_indices]
    
    def _reciprocal_rank_fusion(self, strategy_results: Dict[str, List[Tuple[int, float]]], 
                               k: int = 60) -> List[Tuple[int, float]]:
        """Combine results using Reciprocal Rank Fusion"""
        rrf_scores = {}
        
        for strategy_name, results in strategy_results.items():
            for rank, (doc_id, score) in enumerate(results, 1):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                
                # RRF formula: 1 / (k + rank)
                rrf_scores[doc_id] += 1 / (k + rank)
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(doc_id, score) for doc_id, score in sorted_results]
    
    # 3. Contextual Reranking
    def contextual_reranking(self, query: str, initial_results: List[Tuple[int, float]], 
                           top_k: int = 10) -> List[Tuple[int, float]]:
        """Rerank results considering query context and result diversity"""
        if len(initial_results) <= top_k:
            return initial_results
        
        # Take top candidates for reranking
        candidates = initial_results[:top_k * 2]
        
        # Rerank based on multiple factors
        reranked = []
        for doc_id, score in candidates:
            row = self.df.iloc[doc_id]
            
            # Factor 1: Query-document alignment
            alignment_score = self._calculate_alignment_score(query, row)
            
            # Factor 2: Completeness (how much info is available)
            completeness_score = self._calculate_completeness_score(row)
            
            # Factor 3: Diversity bonus (avoid too similar results)
            diversity_score = self._calculate_diversity_score(row, reranked)
            
            # Combine scores
            final_score = (score * 0.6 + 
                          alignment_score * 0.2 + 
                          completeness_score * 0.1 + 
                          diversity_score * 0.1)
            
            reranked.append((doc_id, final_score))
        
        # Sort by final score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]
    
    def _calculate_alignment_score(self, query: str, row: pd.Series) -> float:
        """Calculate how well the result aligns with query intent"""
        query_lower = query.lower()
        score = 0.0
        
        # Check for exact keyword matches
        if any(word in row['keywords'].lower() for word in query_lower.split()):
            score += 0.3
        
        # Check for role alignment
        if any(word in row['role'].lower() for word in ['founder', 'ceo'] 
               if 'founder' in query_lower):
            score += 0.2
        
        # Check for stage alignment
        if any(stage in query_lower for stage in ['seed', 'series', 'growth']):
            if any(stage in row['stage'].lower() for stage in ['seed', 'series', 'growth']):
                score += 0.2
        
        return score
    
    def _calculate_completeness_score(self, row: pd.Series) -> float:
        """Score based on how complete the profile is"""
        fields = ['founder_name', 'company', 'idea', 'about', 'keywords']
        filled_fields = sum(1 for field in fields if len(str(row[field]).strip()) > 0)
        return filled_fields / len(fields)
    
    def _calculate_diversity_score(self, row: pd.Series, existing_results: List[Tuple[int, float]]) -> float:
        """Promote diversity in results"""
        if not existing_results:
            return 1.0
        
        # Check if we already have similar results
        current_keywords = set(row['keywords'].lower().split(', '))
        current_company = row['company'].lower()
        
        penalty = 0.0
        for doc_id, _ in existing_results:
            existing_row = self.df.iloc[doc_id]
            existing_keywords = set(existing_row['keywords'].lower().split(', '))
            existing_company = existing_row['company'].lower()
            
            # Penalize similar keywords
            if len(current_keywords.intersection(existing_keywords)) > 0:
                penalty += 0.2
            
            # Penalize same company
            if current_company == existing_company:
                penalty += 0.5
        
        return max(0.0, 1.0 - penalty)
    
    # 4. Advanced Query Processing
    def process_complex_query(self, query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
        """Process query with advanced RAG techniques"""
        # Step 1: Fusion retrieval
        fusion_results = self.fusion_retrieval(query, k * 3)
        
        # Step 2: Contextual reranking
        reranked_results = self.contextual_reranking(query, fusion_results, k)
        
        # Step 3: Add explanations
        final_results = []
        for doc_id, score in reranked_results:
            row = self.df.iloc[doc_id]
            explanation = self._generate_explanation(query, row, score)
            final_results.append((doc_id, score, explanation))
        
        return final_results
    
    def _generate_explanation(self, query: str, row: pd.Series, score: float) -> Dict:
        """Generate explanation for why this result matches"""
        explanations = []
        query_lower = query.lower()
        
        # Check keyword matches
        query_words = set(query_lower.split())
        keyword_matches = []
        for keyword in row['keywords'].split(', '):
            if keyword.lower() in query_lower:
                keyword_matches.append(keyword)
        
        if keyword_matches:
            explanations.append(f"Keywords: {', '.join(keyword_matches)}")
        
        # Check idea matches
        idea_words = set(row['idea'].lower().split())
        if len(query_words.intersection(idea_words)) > 0:
            explanations.append("Similar idea description")
        
        # Check role matches
        if any(word in row['role'].lower() for word in query_words):
            explanations.append(f"Role: {row['role']}")
        
        # Check company matches
        if any(word in row['company'].lower() for word in query_words):
            explanations.append(f"Company: {row['company']}")
        
        return {
            'explanations': explanations,
            'confidence': min(score, 1.0),
            'matched_fields': len(explanations)
        }