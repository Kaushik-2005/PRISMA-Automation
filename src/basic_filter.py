import pandas as pd
import json

class BasicFilter:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
    
    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        filters = self.config['filters']
        
        return df[
            (df['Year'] >= filters['year']['min']) &
            (df['Year'] <= filters['year']['max']) &
            (df['Cited by'] >= filters['citations']['min']) &
            (df['Document Type'].isin(filters['document_types']))
        ]