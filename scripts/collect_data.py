#!/usr/bin/env python3
"""
Data collection script for GeoLingua project.
Collects linguistic data from various sources for different geographic regions.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.collectors import DataCollectionManager
from config.data_config import REDDIT_SUBREDDITS, RAW_DATA_PATH


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_collection.log'),
            logging.StreamHandler()
        ]
    )


def collect_region_data(region: str, output_dir: str, force: bool = False):
    """
    Collect data for a specific region.
    
    Args:
        region: Region name to collect data for
        output_dir: Directory to save collected data
        force: Whether to overwrite existing data
    """
    logger = logging.getLogger(__name__)
    
    # Check if region exists
    if region not in REDDIT_SUBREDDITS:
        logger.error(f"Region '{region}' not found. Available regions: {list(REDDIT_SUBREDDITS.keys())}")
        return False
    
    # Set up output path
    output_path = os.path.join(output_dir, f"{region}_data.json")
    
    # Check if file exists and force is False
    if os.path.exists(output_path) and not force:
        logger.info(f"Data for region '{region}' already exists at {output_path}")
        logger.info("Use --force to overwrite existing data")
        return True
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting data collection for region: {region}")
    
    try:
        # Initialize data collection manager
        collector = DataCollectionManager()
        
        # Collect data
        data = collector.collect_all_data(region, output_path)
        
        # Print summary
        logger.info(f"Data collection completed for region: {region}")
        logger.info(f"Summary:")
        logger.info(f"  Reddit posts: {len(data['reddit']['posts'])}")
        logger.info(f"  Reddit comments: {len(data['reddit']['comments'])}")
        logger.info(f"  News articles: {len(data['news'])}")
        logger.info(f"  Wikipedia articles: {len(data['wikipedia'])}")
        logger.info(f"  Data saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error collecting data for region '{region}': {e}")
        return False


def collect_all_regions(output_dir: str, force: bool = False):
    """
    Collect data for all configured regions.
    
    Args:
        output_dir: Directory to save collected data
        force: Whether to overwrite existing data
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data collection for all regions")
    
    success_count = 0
    total_regions = len(REDDIT_SUBREDDITS)
    
    for region in REDDIT_SUBREDDITS.keys():
        logger.info(f"Processing region {success_count + 1}/{total_regions}: {region}")
        
        if collect_region_data(region, output_dir, force):
            success_count += 1
            logger.info(f"Successfully collected data for region: {region}")
        else:
            logger.error(f"Failed to collect data for region: {region}")
    
    logger.info(f"Data collection completed. Success: {success_count}/{total_regions}")
    return success_count == total_regions


def validate_data(data_dir: str):
    """
    Validate collected data files.
    
    Args:
        data_dir: Directory containing data files
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Validating collected data...")
    
    validation_results = {}
    
    for region in REDDIT_SUBREDDITS.keys():
        data_file = os.path.join(data_dir, f"{region}_data.json")
        
        if not os.path.exists(data_file):
            validation_results[region] = {'status': 'missing', 'details': 'Data file not found'}
            continue
        
        try:
            # Load and validate data
            collector = DataCollectionManager()
            data = collector.load_data(data_file)
            
            if not data:
                validation_results[region] = {'status': 'invalid', 'details': 'Failed to load data'}
                continue
            
            # Check data structure
            required_keys = ['region', 'reddit', 'news', 'wikipedia', 'collection_metadata']
            missing_keys = [key for key in required_keys if key not in data]
            
            if missing_keys:
                validation_results[region] = {
                    'status': 'invalid', 
                    'details': f'Missing keys: {missing_keys}'
                }
                continue
            
            # Check data counts
            reddit_posts = len(data['reddit']['posts'])
            reddit_comments = len(data['reddit']['comments'])
            news_articles = len(data['news'])
            wikipedia_articles = len(data['wikipedia'])
            
            validation_results[region] = {
                'status': 'valid',
                'details': {
                    'reddit_posts': reddit_posts,
                    'reddit_comments': reddit_comments,
                    'news_articles': news_articles,
                    'wikipedia_articles': wikipedia_articles,
                    'total_items': reddit_posts + reddit_comments + news_articles + wikipedia_articles
                }
            }
            
        except Exception as e:
            validation_results[region] = {'status': 'error', 'details': str(e)}
    
    # Print validation results
    logger.info("Validation Results:")
    for region, result in validation_results.items():
        status = result['status']
        logger.info(f"  {region}: {status}")
        
        if status == 'valid':
            details = result['details']
            logger.info(f"    Reddit posts: {details['reddit_posts']}")
            logger.info(f"    Reddit comments: {details['reddit_comments']}")
            logger.info(f"    News articles: {details['news_articles']}")
            logger.info(f"    Wikipedia articles: {details['wikipedia_articles']}")
            logger.info(f"    Total items: {details['total_items']}")
        else:
            logger.warning(f"    Details: {result['details']}")
    
    return validation_results


def main():
    """Main function to handle command line arguments and execute data collection."""
    parser = argparse.ArgumentParser(
        description="Collect linguistic data for GeoLingua project"
    )
    
    parser.add_argument(
        '--region', '-r',
        type=str,
        help=f"Specific region to collect data for. Options: {list(REDDIT_SUBREDDITS.keys())}"
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help="Collect data for all configured regions"
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=RAW_DATA_PATH,
        help=f"Output directory for collected data (default: {RAW_DATA_PATH})"
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help="Force overwrite existing data files"
    )
    
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help="Validate existing data files"
    )
    
    parser.add_argument(
        '--log-level', '-l',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if not args.region and not args.all and not args.validate:
        logger.error("Please specify --region, --all, or --validate")
        parser.print_help()
        return 1
    
    if args.region and args.all:
        logger.error("Cannot specify both --region and --all")
        return 1
    
    # Execute requested operation
    try:
        if args.validate:
            logger.info("Validating existing data files...")
            validation_results = validate_data(args.output_dir)
            
            # Check if all regions are valid
            valid_regions = [r for r, result in validation_results.items() 
                           if result['status'] == 'valid']
            
            if len(valid_regions) == len(REDDIT_SUBREDDITS):
                logger.info("All data files are valid!")
                return 0
            else:
                logger.warning(f"Valid regions: {len(valid_regions)}/{len(REDDIT_SUBREDDITS)}")
                return 1
        
        elif args.all:
            logger.info("Collecting data for all regions...")
            success = collect_all_regions(args.output_dir, args.force)
            return 0 if success else 1
        
        elif args.region:
            logger.info(f"Collecting data for region: {args.region}")
            success = collect_region_data(args.region, args.output_dir, args.force)
            return 0 if success else 1
    
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())