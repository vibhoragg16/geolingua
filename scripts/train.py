#!/usr/bin/env python3
"""
Training script for GeoLingua project.
Trains the language model with geographic adaptation using GRPO techniques.
"""

import os
import sys
import argparse
import logging
import json
import torch
import wandb
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.base_model import GeoLinguaModel
from models.grpo_trainer import GRPOTrainer
from models.geographic_adapter import GeographicAdapter
from data.loaders import GeoLinguaDataLoader
from data.preprocessors import GeoLinguaPreprocessor
from config.model_config import *
from config.data_config import REGIONS, RAW_DATA_PATH, PROCESSED_DATA_PATH


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_and_preprocess_data(regions: List[str], data_dir: str, processed_dir: str, 
                           force_preprocess: bool = False) -> Dict:
    """
    Load and preprocess data for training.
    
    Args:
        regions: List of regions to include
        data_dir: Directory containing raw data
        processed_dir: Directory for processed data
        force_preprocess: Whether to force reprocessing
    
    Returns:
        Dictionary containing processed datasets
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading and preprocessing data for regions: {regions}")
    
    # Initialize preprocessor
    preprocessor = GeoLinguaPreprocessor(
        max_length=MAX_LENGTH,
        regions=regions
    )
    
    # Check if processed data exists
    processed_file = os.path.join(processed_dir, "processed_data.json")
    
    if os.path.exists(processed_file) and not force_preprocess:
        logger.info(f"Loading existing processed data from {processed_file}")
        try:
            with open(processed_file, 'r') as f:
                processed_data = json.load(f)
            return processed_data
        except Exception as e:
            logger.warning(f"Failed to load processed data: {e}")
            logger.info("Proceeding with data preprocessing...")
    
    # Load raw data
    raw_data = {}
    for region in regions:
        data_file = os.path.join(data_dir, f"{region}_data.json")
        
        if not os.path.exists(data_file):
            logger.error(f"Data file not found for region {region}: {data_file}")
            continue
        
        try:
            with open(data_file, 'r') as f:
                raw_data[region] = json.load(f)
            logger.info(f"Loaded data for region {region}")
        except Exception as e:
            logger.error(f"Failed to load data for region {region}: {e}")
            continue
    
    if not raw_data:
        raise ValueError("No data loaded. Please check data files.")
    
    # Preprocess data
    logger.info("Starting data preprocessing...")
    processed_data = preprocessor.preprocess_all_regions(raw_data)
    
    # Save processed data
    os.makedirs(processed_dir, exist_ok=True)
    with open(processed_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    logger.info(f"Processed data saved to {processed_file}")
    return processed_data


def initialize_model(model_name: str, regions: List[str], device: str) -> GeoLinguaModel:
    """
    Initialize the GeoLingua model with geographic adapters.
    
    Args:
        model_name: Base model name
        regions: List of regions
        device: Device to use
    
    Returns:
        Initialized GeoLinguaModel
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Initializing model: {model_name}")
    
    # Initialize base model
    model = GeoLinguaModel(
        model_name=model_name,
        regions=regions,
        lora_config={
            'r': LORA_R,
            'lora_alpha': LORA_ALPHA,
            'lora_dropout': LORA_DROPOUT,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        }
    )
    
    # Move to device
    model = model.to(device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    return model


def train_model(model: GeoLinguaModel, processed_data: Dict, 
                output_dir: str, config: Dict) -> str:
    """
    Train the GeoLingua model using GRPO.
    
    Args:
        model: GeoLingua model to train
        processed_data: Processed training data
        output_dir: Directory to save model checkpoints
        config: Training configuration
    
    Returns:
        Path to best model checkpoint
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting GRPO training...")
    
    # Initialize data loader
    data_loader = GeoLinguaDataLoader(
        processed_data=processed_data,
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        regions=config['regions']
    )
    
    # Get training datasets
    train_datasets = data_loader.get_training_datasets()
    eval_datasets = data_loader.get_evaluation_datasets()
    
    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        output_dir=output_dir,
        logging_steps=config.get('logging_steps', 100),
        eval_steps=config.get('eval_steps', 500),
        save_steps=config.get('save_steps', 1000),
        warmup_steps=config.get('warmup_steps', 100),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        weight_decay=config.get('weight_decay', 0.01),
        use_wandb=config.get('use_wandb', False)
    )
    
    # Start training
    best_model_path = trainer.train()
    
    logger.info(f"Training completed. Best model saved to: {best_model_path}")
    return best_model_path


def validate_config(config: Dict) -> bool:
    """
    Validate training configuration.
    
    Args:
        config: Training configuration
    
    Returns:
        True if config is valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    required_keys = ['regions', 'batch_size', 'learning_rate', 'num_epochs', 'max_length']
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False
    
    # Validate regions
    for region in config['regions']:
        if region not in REGIONS:
            logger.error(f"Invalid region: {region}. Available regions: {list(REGIONS.keys())}")
            return False
    
    # Validate numeric values
    if config['batch_size'] <= 0:
        logger.error("Batch size must be positive")
        return False
    
    if config['learning_rate'] <= 0:
        logger.error("Learning rate must be positive")
        return False
    
    if config['num_epochs'] <= 0:
        logger.error("Number of epochs must be positive")
        return False
    
    if config['max_length'] <= 0:
        logger.error("Max length must be positive")
        return False
    
    return True


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train GeoLingua model with geographic adaptation"
    )
    
    parser.add_argument(
        '--model-name', '-m',
        type=str,
        default=MODEL_NAME,
        help=f"Base model name (default: {MODEL_NAME})"
    )
    
    parser.add_argument(
        '--regions', '-r',
        nargs='+',
        default=list(REGIONS.keys()),
        help=f"Regions to train on (default: all regions)"
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default=RAW_DATA_PATH,
        help=f"Directory containing raw data (default: {RAW_DATA_PATH})"
    )
    
    parser.add_argument(
        '--processed-dir', '-p',
        type=str,
        default=PROCESSED_DATA_PATH,
        help=f"Directory for processed data (default: {PROCESSED_DATA_PATH})"
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./checkpoints',
        help="Output directory for model checkpoints (default: ./checkpoints)"
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )
    
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    
    parser.add_argument(
        '--num-epochs', '-e',
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of epochs (default: {NUM_EPOCHS})"
    )
    
    parser.add_argument(
        '--max-length', '-ml',
        type=int,
        default=MAX_LENGTH,
        help=f"Maximum sequence length (default: {MAX_LENGTH})"
    )
    
    parser.add_argument(
        '--force-preprocess',
        action='store_true',
        help="Force reprocessing of data"
    )
    
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help="Use Weights & Biases for logging"
    )
    
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='geolingua',
        help="Weights & Biases project name (default: geolingua)"
    )
    
    parser.add_argument(
        '--log-level', '-l',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help="Device to use (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"training_{timestamp}.log"
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create training configuration
    config = {
        'model_name': args.model_name,
        'regions': args.regions,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'max_length': args.max_length,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'device': device,
        'logging_steps': 100,
        'eval_steps': 500,
        'save_steps': 1000,
        'warmup_steps': 100,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        'weight_decay': 0.01
    }
    
    # Validate configuration
    if not validate_config(config):
        logger.error("Invalid configuration. Exiting.")
        return 1
    
    logger.info("Starting GeoLingua training...")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    try:
        # Initialize Weights & Biases if requested
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                config=config,
                name=f"geolingua_training_{timestamp}"
            )
        
        # Load and preprocess data
        processed_data = load_and_preprocess_data(
            regions=args.regions,
            data_dir=args.data_dir,
            processed_dir=args.processed_dir,
            force_preprocess=args.force_preprocess
        )
        
        # Initialize model
        model = initialize_model(
            model_name=args.model_name,
            regions=args.regions,
            device=device
        )
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Train model
        best_model_path = train_model(
            model=model,
            processed_data=processed_data,
            output_dir=args.output_dir,
            config=config
        )
        
        # Save final configuration
        config_file = os.path.join(args.output_dir, 'training_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Best model: {best_model_path}")
        logger.info(f"Configuration saved: {config_file}")
        
        if args.use_wandb:
            wandb.finish()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if args.use_wandb:
            wandb.finish()
        return 130
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.use_wandb:
            wandb.finish()
        return 1


if __name__ == "__main__":
    exit(main())