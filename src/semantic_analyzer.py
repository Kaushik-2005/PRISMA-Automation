import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

class SemanticAnalyzer:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
        self.model = SentenceTransformer(self.config['model']['name'])
    
    def calculate_scores(self, df: pd.DataFrame, research_objective: str) -> pd.Series:
        texts = df['Title'] + ' ' + df['Abstract']
        
        paper_embeddings = self.model.encode(texts.tolist())
        objective_embedding = self.model.encode(research_objective)
        
        return pd.Series(
            cosine_similarity([objective_embedding], paper_embeddings)[0]
        )