#!/usr/bin/env python3
"""
Data preprocessing script for GeoLingua project.
Converts raw collected data into training-ready datasets.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List

# Add src and config to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
config_path = os.path.join(project_root, 'config')

sys.path.insert(0, src_path)
sys.path.insert(0, config_path)

from data.preprocessors import DatasetPreprocessor
from data.loaders import DataLoader
from config.data_config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_raw_data(data_dir: str) -> Dict[str, Dict]:
    """
    Load raw data for all regions.
    
    Args:
        data_dir: Directory containing raw data files
        
    Returns:
        Dictionary mapping region names to their raw data
    """
    raw_data = {}
    
    if not os.path.exists(data_dir):
        print(f"Raw data directory not found: {data_dir}")
        return raw_data
    
    for filename in os.listdir(data_dir):
        if filename.endswith('_data.json'):
            region = filename.replace('_data.json', '')
            filepath = os.path.join(data_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                raw_data[region] = data
                print(f"Loaded raw data for {region}: {len(data.get('reddit', {}).get('posts', []))} posts, {len(data.get('news', []))} news articles, {len(data.get('wikipedia', []))} Wikipedia articles")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return raw_data

def preprocess_region_data(region: str, raw_data: Dict, preprocessor: DatasetPreprocessor) -> List[Dict]:
    """
    Preprocess data for a specific region.
    
    Args:
        region: Region name
        raw_data: Raw data for the region
        preprocessor: DatasetPreprocessor instance
        
    Returns:
        List of processed training examples
    """
    training_examples = []
    
    # Process Reddit posts
    reddit_posts = raw_data.get('reddit', {}).get('posts', [])
    if reddit_posts:
        processed_posts = preprocessor.preprocess_dataset(reddit_posts, 'reddit_posts')
        post_examples = preprocessor.create_training_examples(processed_posts, region)
        training_examples.extend(post_examples)
        print(f"  Reddit posts: {len(reddit_posts)} -> {len(processed_posts)} processed -> {len(post_examples)} examples")
    
    # Process Reddit comments
    reddit_comments = raw_data.get('reddit', {}).get('comments', [])
    if reddit_comments:
        processed_comments = preprocessor.preprocess_dataset(reddit_comments, 'reddit_comments')
        comment_examples = preprocessor.create_training_examples(processed_comments, region)
        training_examples.extend(comment_examples)
        print(f"  Reddit comments: {len(reddit_comments)} -> {len(processed_comments)} processed -> {len(comment_examples)} examples")
    
    # Process news articles
    news_articles = raw_data.get('news', [])
    if news_articles:
        processed_news = preprocessor.preprocess_dataset(news_articles, 'news_articles')
        news_examples = preprocessor.create_training_examples(processed_news, region)
        training_examples.extend(news_examples)
        print(f"  News articles: {len(news_articles)} -> {len(processed_news)} processed -> {len(news_examples)} examples")
    
    # Process Wikipedia articles
    wikipedia_articles = raw_data.get('wikipedia', [])
    if wikipedia_articles:
        # Wikipedia articles are already in a good format, just create examples
        wiki_examples = preprocessor.create_training_examples(wikipedia_articles, region)
        training_examples.extend(wiki_examples)
        print(f"  Wikipedia articles: {len(wikipedia_articles)} -> {len(wiki_examples)} examples")
    
    return training_examples

def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess raw data for GeoLingua training")
    
    parser.add_argument(
        '--raw-data-dir',
        type=str,
        default=RAW_DATA_PATH,
        help=f"Directory containing raw data (default: {RAW_DATA_PATH})"
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=PROCESSED_DATA_PATH,
        help=f"Directory for processed data (default: {PROCESSED_DATA_PATH})"
    )
    
    parser.add_argument(
        '--regions',
        nargs='+',
        default=None,
        help="Specific regions to process (default: all regions)"
    )
    
    parser.add_argument(
        '--max-per-region',
        type=int,
        default=1000,
        help="Maximum examples per region (default: 1000)"
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data preprocessing...")
    
    # Load raw data
    raw_data = load_raw_data(args.raw_data_dir)
    
    if not raw_data:
        logger.error("No raw data found. Please run data collection first.")
        return 1
    
    # Filter regions if specified
    if args.regions:
        raw_data = {region: data for region, data in raw_data.items() if region in args.regions}
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor()
    
    # Process each region
    all_training_examples = []
    
    for region, data in raw_data.items():
        logger.info(f"Processing region: {region}")
        
        try:
            region_examples = preprocess_region_data(region, data, preprocessor)
            all_training_examples.extend(region_examples)
            
            logger.info(f"Region {region}: {len(region_examples)} training examples")
            
        except Exception as e:
            logger.error(f"Error processing region {region}: {e}")
            continue
    
    if not all_training_examples:
        logger.error("No training examples generated. Check your data and preprocessing settings.")
        return 1
    
    # Balance dataset
    logger.info("Balancing dataset...")
    balanced_examples = preprocessor.balance_dataset(all_training_examples, args.max_per_region)
    
    # Save processed data
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'processed_dataset.json')
    
    try:
        preprocessor.save_processed_dataset(balanced_examples, output_file)
        logger.info(f"Saved {len(balanced_examples)} training examples to {output_file}")
        
        # Print summary statistics
        region_counts = {}
        source_counts = {}
        
        for example in balanced_examples:
            region = example['region']
            source = example['source_type']
            
            region_counts[region] = region_counts.get(region, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info("Dataset Summary:")
        logger.info(f"  Total examples: {len(balanced_examples)}")
        logger.info(f"  Regions: {region_counts}")
        logger.info(f"  Top sources: {dict(sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5])}")
        
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        return 1
    
    logger.info("Data preprocessing completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main()) 