import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List
import json

class KeywordAnalyzer:
    def __init__(self, config_path: str):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        with open(config_path) as f:
            self.config = json.load(f)
    
    def calculate_scores(self, df: pd.DataFrame, keywords: Dict[str, List[str]]) -> pd.Series:
        df['combined_text'] = (
            df['Title'] + ' ' + 
            df['Abstract'] + ' ' + 
            df['Author Keywords'].fillna('')
        )
        
        # Calculate primary keyword scores
        primary_scores = self._calculate_keyword_group_scores(
            df['combined_text'], 
            keywords['primary']
        )
        
        # Calculate secondary keyword scores
        secondary_scores = self._calculate_keyword_group_scores(
            df['combined_text'], 
            keywords['secondary']
        )
        
        # Combine scores using weights
        weights = self.config['scoring']['keyword_hierarchy']
        final_scores = (
            weights['primary_weight'] * primary_scores +
            weights['secondary_weight'] * secondary_scores
        )
        
        return final_scores
    
    def _calculate_keyword_group_scores(self, texts: pd.Series, keywords: List[str]) -> pd.Series:
        if not keywords:
            return pd.Series(0, index=texts.index)
            
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        keyword_vector = self.vectorizer.transform([' '.join(keywords)])
        
        return pd.Series(
            cosine_similarity(tfidf_matrix, keyword_vector).flatten()
        )