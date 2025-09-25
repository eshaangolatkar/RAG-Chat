# location_handler.py - Dynamic location matching without hardcoding

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process
import re
from typing import List, Dict, Tuple, Optional
import streamlit as st

class DynamicLocationMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.location_embeddings = None
        self.unique_locations = []
        self.city_variations = {}
        
    def build_location_index(self, df: pd.DataFrame):
        """Build dynamic location index from the dataset itself"""
        # Extract all unique locations from dataset
        locations = df['location'].dropna().unique().tolist()
        self.unique_locations = locations
        
        # Create embeddings for all locations
        self.location_embeddings = self.model.encode(
            locations, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        
        # Build city variations dictionary dynamically
        self._build_city_variations(locations)
        
    def _build_city_variations(self, locations: List[str]):
        """Dynamically extract city variations from location data"""
        self.city_variations = {}
        
        for location in locations:
            # Extract city name (before comma)
            if ',' in location:
                city = location.split(',')[0].strip().lower()
                country = location.split(',')[-1].strip().lower()
                
                # Group cities by country
                if country not in self.city_variations:
                    self.city_variations[country] = []
                self.city_variations[country].append(city)
                
                # Add common variations
                variations = self._generate_variations(city)
                for var in variations:
                    if country not in self.city_variations:
                        self.city_variations[country] = []
                    if var not in self.city_variations[country]:
                        self.city_variations[country].append(var)
    
    def _generate_variations(self, city: str) -> List[str]:
        """Generate common variations for a city name"""
        variations = [city]
        
        # Comprehensive abbreviations and variations for Indian cities
        city_mappings = {
            'bangalore': ['bengaluru', 'blr', 'bang', 'bangalore', 'bengaluru'],
            'bengaluru': ['bangalore', 'blr', 'bang', 'bangalore', 'bengaluru'],
            'mumbai': ['bombay', 'bom', 'mumbai', 'bombay'],
            'bombay': ['mumbai', 'bom', 'mumbai', 'bombay'],
            'delhi': ['new delhi', 'ndl', 'delhi', 'new delhi'],
            'new delhi': ['delhi', 'ndl', 'delhi', 'new delhi'],
            'chennai': ['madras', 'maa', 'chennai', 'madras'],
            'madras': ['chennai', 'maa', 'chennai', 'madras'],
            'kolkata': ['calcutta', 'ccu', 'kolkata', 'calcutta'],
            'calcutta': ['kolkata', 'ccu', 'kolkata', 'calcutta'],
            'hyderabad': ['hyd', 'secunderabad', 'hyderabad'],
            'pune': ['poona', 'pnq', 'pune', 'poona'],
            'ahmedabad': ['amdavad', 'amd', 'ahmedabad', 'amdavad'],
            'thiruvananthapuram': ['trivandrum', 'tvm', 'thiruvananthapuram', 'trivandrum'],
            'kochi': ['cochin', 'kochi', 'cochin'],
            'cochin': ['kochi', 'kochi', 'cochin'],
            'gurgaon': ['gurugram', 'ggn', 'gurgaon', 'gurugram'],
            'gurugram': ['gurgaon', 'ggn', 'gurgaon', 'gurugram'],
        }
        
        if city in city_mappings:
            variations.extend(city_mappings[city])
        
        # Also add reverse mappings for abbreviations
        abbrev_to_city = {
            'blr': ['bangalore', 'bengaluru'],
            'bom': ['mumbai', 'bombay'],
            'ndl': ['delhi', 'new delhi'],
            'maa': ['chennai', 'madras'],
            'ccu': ['kolkata', 'calcutta'],
            'hyd': ['hyderabad'],
            'pnq': ['pune', 'poona'],
            'amd': ['ahmedabad', 'amdavad'],
            'tvm': ['thiruvananthapuram', 'trivandrum'],
            'ggn': ['gurgaon', 'gurugram'],
        }
        
        if city in abbrev_to_city:
            variations.extend(abbrev_to_city[city])
            
        return list(set(variations))  # Remove duplicates
    
    def extract_locations_from_query(self, query: str) -> List[str]:
        """Extract location terms from query using multiple strategies"""
        query_lower = query.lower().strip()
        extracted_locations = []
        
        # Pattern-based extraction
        location_patterns = [
            r'\bin\s+([a-zA-Z\s,.-]+?)(?:\s+(?:with|for|at|founder|startup|company|looking|seeking|who)|[.!?]|$)',
            r'\bfrom\s+([a-zA-Z\s,.-]+?)(?:\s+(?:with|for|at|founder|startup|company|looking|seeking)|[.!?]|$)',  
            r'\bat\s+([a-zA-Z\s,.-]+?)(?:\s+(?:with|for|at|founder|startup|company|looking|seeking)|[.!?]|$)',
            r'([a-zA-Z\s,.-]+?)\s+(?:based|located)(?:\s|[.!?]|$)',
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                location = match.strip()
                location = re.sub(r'\b(startup|founder|company|with|for|at|the|a|an|who|based|located)\b', '', location).strip()
                if len(location) > 1 and location not in ['', 'ai', 'ml', 'tech']:
                    extracted_locations.append(location)
        
        # If no patterns found, use fuzzy matching against known locations
        if not extracted_locations:
            extracted_locations = self._fuzzy_location_extraction(query_lower)
        
        return list(set(extracted_locations))
    
    def _fuzzy_location_extraction(self, query: str) -> List[str]:
        """Use fuzzy matching to find locations in query"""
        words = query.split()
        potential_locations = []
        
        # Create a list of all known cities from our variations
        all_cities = []
        for country, cities in self.city_variations.items():
            all_cities.extend(cities)
        
        # Check each word/phrase against known cities
        for i, word in enumerate(words):
            word_clean = word.strip('.,!?;:"()[]{}')
            
            # Single word matching
            if len(word_clean) > 2:
                matches = process.extractBests(word_clean, all_cities, 
                                             scorer=fuzz.ratio, score_cutoff=70, limit=3)
                potential_locations.extend([match[0] for match in matches])
            
            # Two word combinations
            if i < len(words) - 1:
                two_word = f"{word_clean} {words[i+1].strip('.,!?;:\"()[]{}')}"
                matches = process.extractBests(two_word, all_cities, 
                                             scorer=fuzz.ratio, score_cutoff=75, limit=2)
                potential_locations.extend([match[0] for match in matches])
        
        return list(set(potential_locations))
    
    def find_matching_locations(self, query_locations: List[str], 
                              similarity_threshold: float = 0.75) -> List[Dict]:
        """Find matching locations using multiple strategies"""
        if not query_locations or not self.unique_locations:
            return []
        
        all_matches = []
        
        for query_loc in query_locations:
            matches = []
            
            # Strategy 1: Exact string matching
            exact_matches = self._exact_string_matching(query_loc)
            matches.extend(exact_matches)
            
            # Strategy 2: Fuzzy string matching
            if not matches:
                fuzzy_matches = self._fuzzy_string_matching(query_loc)
                matches.extend(fuzzy_matches)
            
            # Strategy 3: Semantic similarity
            if not matches:
                semantic_matches = self._semantic_matching(query_loc, similarity_threshold)
                matches.extend(semantic_matches)
            
            # Strategy 4: Partial matching
            if not matches:
                partial_matches = self._partial_matching(query_loc)
                matches.extend(partial_matches)
            
            all_matches.extend(matches)
        
        return all_matches
    
    def _exact_string_matching(self, query_loc: str) -> List[Dict]:
        """Find exact string matches with enhanced abbreviation handling"""
        query_lower = query_loc.lower()
        matches = []
        
        # Get all variations for this query location
        query_variations = self._generate_variations(query_lower)
        
        for location in self.unique_locations:
            location_lower = location.lower()
            
            # Extract city and country from full location string
            if ',' in location_lower:
                city_part = location_lower.split(',')[0].strip()
                country_part = location_lower.split(',')[-1].strip()
                location_parts = [city_part, country_part, location_lower]
            else:
                location_parts = [location_lower]
            
            # Check if any query variation matches any location part
            for query_var in query_variations:
                for loc_part in location_parts:
                    if (query_var == loc_part or 
                        query_var in loc_part or 
                        loc_part in query_var or
                        # Check if query variation matches city part specifically
                        (query_var == city_part if ',' in location_lower else False)):
                        matches.append({
                            'location': location,
                            'method': 'exact',
                            'score': 1.0,
                            'query': query_loc,
                            'matched_variation': query_var
                        })
                        break
                if matches and matches[-1]['location'] == location:
                    break  # Avoid duplicate matches for same location
        
        return matches
    
    def _fuzzy_string_matching(self, query_loc: str, threshold: int = 80) -> List[Dict]:
        """Find fuzzy string matches"""
        matches = []
        fuzzy_results = process.extractBests(
            query_loc, self.unique_locations, 
            scorer=fuzz.ratio, score_cutoff=threshold, limit=5
        )
        
        for location, score in fuzzy_results:
            matches.append({
                'location': location,
                'method': 'fuzzy',
                'score': score / 100.0,
                'query': query_loc
            })
        
        return matches
    
    def _semantic_matching(self, query_loc: str, threshold: float) -> List[Dict]:
        """Find semantically similar locations"""
        matches = []
        
        if self.location_embeddings is not None:
            query_embedding = self.model.encode([query_loc], normalize_embeddings=True)
            similarities = cosine_similarity(query_embedding, self.location_embeddings)[0]
            
            for i, sim in enumerate(similarities):
                if sim >= threshold:
                    matches.append({
                        'location': self.unique_locations[i],
                        'method': 'semantic',
                        'score': sim,
                        'query': query_loc
                    })
        
        return matches
    
    def _partial_matching(self, query_loc: str) -> List[Dict]:
        """Find partial matches (last resort)"""
        query_lower = query_loc.lower()
        matches = []
        
        for location in self.unique_locations:
            location_lower = location.lower()
            
            # Check if any word from query is in location
            query_words = query_lower.split()
            location_words = location_lower.split()
            
            for q_word in query_words:
                if len(q_word) > 2:
                    for l_word in location_words:
                        if (q_word in l_word or l_word in q_word) and len(l_word) > 2:
                            matches.append({
                                'location': location,
                                'method': 'partial',
                                'score': 0.5,
                                'query': query_loc
                            })
                            break
        
        return matches
    
    def debug_location_matching(self, query: str) -> Dict:
        """Debug method to understand location matching process"""
        debug_info = {
            'original_query': query,
            'extracted_locations': [],
            'location_variations': {},
            'dataset_sample_locations': [],
            'matching_attempts': {}
        }
        
        # Extract locations
        extracted_locations = self.extract_locations_from_query(query)
        debug_info['extracted_locations'] = extracted_locations
        
        # Show variations for each extracted location
        for loc in extracted_locations:
            variations = self._generate_variations(loc.lower())
            debug_info['location_variations'][loc] = variations
        
        # Show sample dataset locations
        if self.unique_locations:
            debug_info['dataset_sample_locations'] = self.unique_locations[:10]
        
        # Test each matching strategy
        for query_loc in extracted_locations:
            debug_info['matching_attempts'][query_loc] = {
                'exact_matches': self._exact_string_matching(query_loc),
                'fuzzy_matches': self._fuzzy_string_matching(query_loc, threshold=70),
                'semantic_matches': self._semantic_matching(query_loc, threshold=0.6),
                'partial_matches': self._partial_matching(query_loc)
            }
        
        return debug_info
    
    def filter_dataframe_by_locations(self, df: pd.DataFrame, 
                                    query: str) -> Tuple[pd.DataFrame, str]:
        """Main function to filter dataframe by locations"""
        
        # Build index if not already built
        if not self.unique_locations:
            self.build_location_index(df)
        
        # Extract locations from query
        query_locations = self.extract_locations_from_query(query)
        
        if not query_locations:
            return pd.DataFrame(), "❌ No location found in query. Please specify a location (e.g., 'AI startup in Bangalore')"
        
        # Find matching locations
        matching_locations = self.find_matching_locations(query_locations)
        
        if not matching_locations:
            suggestions = self._get_location_suggestions(query_locations)
            return pd.DataFrame(), f"❌ No matches found for: {', '.join(query_locations)}. Try: {suggestions}"
        
        # Filter dataframe
        matched_location_names = [match['location'] for match in matching_locations]
        filtered_df = df[df['location'].isin(matched_location_names)]
        
        # Create informative message
        best_matches = sorted(matching_locations, key=lambda x: x['score'], reverse=True)[:3]
        match_info = []
        for match in best_matches:
            match_method = match.get('method', 'unknown')
            match_score = match.get('score', 0)
            match_info.append(f"{match['location']} ({match_method}: {match_score:.2f})")
        
        message = f"✅ Found {len(filtered_df)} results matching: {', '.join(match_info)}"
        
        return filtered_df, message
    
    def _get_location_suggestions(self, failed_locations: List[str]) -> str:
        """Provide location suggestions when no matches found"""
        suggestions = []
        
        # Get top cities from dataset
        if self.unique_locations:
            # Sort by frequency or just take first few
            top_locations = self.unique_locations[:5]
            suggestions.extend([loc.split(',')[0] for loc in top_locations])
        
        # Add some common variations
        common_suggestions = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Pune']
        suggestions.extend(common_suggestions)
        
        return ', '.join(list(set(suggestions))[:5])

# Usage example for integration
@st.cache_resource
def get_location_matcher():
    return DynamicLocationMatcher()

def enhanced_location_filtering(query: str, df: pd.DataFrame, 
                              location_matcher: DynamicLocationMatcher) -> Tuple[pd.DataFrame, str]:
    """Enhanced location filtering using the dynamic matcher"""
    
    # Build index if not already built
    if not location_matcher.unique_locations:
        location_matcher.build_location_index(df)
    
    # Filter by locations
    filtered_df, message = location_matcher.filter_dataframe_by_locations(df, query)
    
    return filtered_df, message