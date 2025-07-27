import logging
from pathlib import Path
import json
from typing import List, Optional, Dict
import pandas as pd

from src.data import DataLoader
from src.basic_filter import BasicFilter
from src.keyword_analyzer import KeywordAnalyzer
from src.semantic_analyzer import SemanticAnalyzer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run_pipeline(
    csv_path: str,
    research_objective: str,
    keywords: Dict[str, List[str]],
    output_path: Optional[str] = None
) -> Optional[pd.DataFrame]:
    setup_logging()
    logger = logging.getLogger(__name__)
    config_path = Path(__file__).parent / 'config.json'
    
    try:
        # Load configuration
        with open(config_path) as f:
            config = json.load(f)
        
        # Initialize components
        data_loader = DataLoader(config_path)
        basic_filter = BasicFilter(config_path)
        keyword_analyzer = KeywordAnalyzer(config_path)
        semantic_analyzer = SemanticAnalyzer(config_path)
        
        # Load and process data
        logger.info("Loading data...")
        df = data_loader.load_csv(csv_path)
        if df is None:
            return None
        
        initial_count = len(df)
        logger.info(f"Loaded {initial_count} papers")
        
        # Apply filters
        logger.info("Applying basic filters...")
        df = basic_filter.apply_filters(df)
        logger.info(f"Remaining papers after filtering: {len(df)}")
        
        # Calculate scores
        logger.info("Calculating relevance scores...")
        df['keyword_score'] = keyword_analyzer.calculate_scores(df, keywords)
        df['semantic_score'] = semantic_analyzer.calculate_scores(df, research_objective)
        
        # Calculate final score
        df['final_score'] = (
            config['scoring']['keyword_weight'] * df['keyword_score'] +
            config['scoring']['semantic_weight'] * df['semantic_score']
        )
        
        # Sort and filter results
        df = df.sort_values('final_score', ascending=False)
        df = df[df['final_score'] >= config['scoring']['similarity_threshold']]
        
        logger.info(f"Final number of papers: {len(df)}")
        
        if output_path:
            # Convert DataFrame to list of dictionaries
            papers = df.to_dict(orient='records')
            
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'total_papers': len(papers),
                        'research_objective': research_objective,
                        'keywords': keywords,
                        'date_generated': pd.Timestamp.now().strftime('%Y-%m-%d')
                    },
                    'papers': papers
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Results saved to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return None

if __name__ == "__main__":
    config_path = Path(__file__).parent / 'config.json'
    
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    results = run_pipeline(
        csv_path="scopus.csv",
        research_objective=config['research']['objective'],
        keywords=config['research']['keywords'],
        output_path="filtered_papers.json"  # Changed extension to .json
    )