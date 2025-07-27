import pandas as pd
from typing import Optional
import logging
import json

class DataLoader:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
        self.logger = logging.getLogger(__name__)
        
    def load_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            
            missing_fields = [
                field for field in self.config['data']['required_fields'] 
                if field not in df.columns
            ]
            
            if missing_fields:
                self.logger.error(f"Missing required fields: {missing_fields}")
                return None
                
            return self._clean_data(df)
            
        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}")
            return None
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=['DOI'], keep='first')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Cited by'] = pd.to_numeric(df['Cited by'], errors='coerce').fillna(0)
        
        for col in ['Title', 'Abstract', 'Author Keywords']:
            if col in df.columns:
                df[col] = df[col].str.strip()
        
        return df.dropna(subset=['Title', 'Abstract'])