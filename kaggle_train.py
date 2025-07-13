#!/usr/bin/env python3
"""
Kaggle-optimized training script for GeoLingua project.
Trains the language model with geographic adaptation using GRPO techniques on Kaggle.
"""

import os
import sys
import json
import logging
import torch
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
# Create GeographicDataset objects from the data
from src.models.grpo_trainer import GeographicDataset

# Add src and config to path for Kaggle
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
src_path = os.path.join(project_root, 'src')
config_path = os.path.join(project_root, 'config')

sys.path.insert(0, src_path)
sys.path.insert(0, config_path)

from src.models.basemodel import GeoLinguaModel
from src.models.grpo_trainer import GRPOTrainer, GRPOTrainingConfig
from src.data.loaders import DataLoader
from config.model_config import *
from config.data_config import *

def setup_logging():
    """Setup logging configuration for Kaggle."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_processed_data(data_path: str = "/kaggle/input/geolingua-geographic-language-model-data/processed_dataset.json"):
    """
    Load the already processed dataset.
    
    Args:
        data_path: Path to the processed dataset
        
    Returns:
        List of training examples
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}. Please run preprocessing first.")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        
        logger.info(f"Loaded {len(processed_data)} training examples from {data_path}")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise

def split_data_stratified(processed_data: List[Dict], 
                         train_ratio: float = 0.7, 
                         val_ratio: float = 0.15, 
                         test_ratio: float = 0.15,
                         random_seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train/validation/test sets with stratification by region.
    
    Args:
        processed_data: List of processed data examples
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set  
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    logger = logging.getLogger(__name__)
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Group data by region for stratification
    region_data = defaultdict(list)
    for item in processed_data:
        region = item.get('region', 'unknown')
        region_data[region].append(item)
    
    logger.info(f"Data distribution by region:")
    for region, items in region_data.items():
        logger.info(f"  {region}: {len(items)} examples")
    
    train_data, val_data, test_data = [], [], []
    
    # Split each region's data proportionally
    for region, items in region_data.items():
        # Shuffle items for this region
        random.shuffle(items)
        
        n_items = len(items)
        train_end = int(n_items * train_ratio)
        val_end = train_end + int(n_items * val_ratio)
        
        # Split the data
        train_data.extend(items[:train_end])
        val_data.extend(items[train_end:val_end])
        test_data.extend(items[val_end:])
        
        logger.info(f"  {region}: train={train_end}, val={val_end-train_end}, test={n_items-val_end}")
    
    # Shuffle the final splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    logger.info(f"Final split sizes:")
    logger.info(f"  Train: {len(train_data)} examples")
    logger.info(f"  Validation: {len(val_data)} examples")
    logger.info(f"  Test: {len(test_data)} examples")
    
    return train_data, val_data, test_data

def save_data_splits(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict], 
                     output_dir: str = "/kaggle/working/data/processed"):
    """
    Save the data splits to separate files for later use.
    
    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        output_dir: Directory to save splits
    """
    os.makedirs(output_dir, exist_ok=True)
    
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_path = os.path.join(output_dir, f"{split_name}_split.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Saved {split_name} split to {output_path}")

def initialize_model(regions: Optional[List[str]] = None) -> GeoLinguaModel:
    """
    Initialize the GeoLingua model.
    
    Args:
        regions: List of regions to include
        
    Returns:
        Initialized GeoLinguaModel
    """
    logger = logging.getLogger(__name__)
    
    if regions is None:
        regions = ['us_south', 'uk', 'australia', 'india', 'nigeria']
    
    logger.info(f"Initializing model: {MODEL_NAME}")
    logger.info(f"Regions: {regions}")
    
    # Initialize model
    model = GeoLinguaModel(
        model_name=MODEL_NAME,
        regions=regions,
        lora_config={
            'r': LORA_R,
            'lora_alpha': LORA_ALPHA,
            'lora_dropout': LORA_DROPOUT,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']  # Explicit target modules
        }
    )
    
    # Move to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    logger.info(f"Model initialized on {device}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model

def create_training_config():
    """Create training configuration optimized for Kaggle."""
    return {
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'max_length': MAX_LENGTH,
        'logging_steps': 50,  # More frequent logging on Kaggle
        'eval_steps': 200,
        'save_steps': 500,
        'warmup_steps': 100,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        'weight_decay': 0.01,
        'use_wandb': False,  # Disable wandb on Kaggle
        'output_dir': '/kaggle/working/models/checkpoints'
    }

def train_model(model: GeoLinguaModel, processed_data: List[Dict], config: Dict) -> str:
    """
    Train the GeoLingua model using GRPO.
    
    Args:
        model: GeoLingua model to train
        processed_data: Processed training data
        config: Training configuration
    
    Returns:
        Path to best model checkpoint
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting GRPO training...")
    logger.info(f"Training examples: {len(processed_data)}")
    
    # Split data into train/validation/test sets
    train_data, val_data, test_data = split_data_stratified(
        processed_data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    # Save splits for later use
    save_data_splits(train_data, val_data, test_data)
    
    logger.info(f"Train examples: {len(train_data)}")
    logger.info(f"Validation examples: {len(val_data)}")
    logger.info(f"Test examples: {len(test_data)}")

    # Print region distribution for each split
    from collections import Counter
    logger.info(f"Train region distribution: {Counter([x['region'] for x in train_data])}")
    logger.info(f"Val region distribution: {Counter([x['region'] for x in val_data])}")
    logger.info(f"Test region distribution: {Counter([x['region'] for x in test_data])}")
    
    # Create GeographicDataset objects from the data
    # We need to save the data as JSON files first
    train_data_path = "/kaggle/working/data/processed/train_data.json"
    val_data_path = "/kaggle/working/data/processed/val_data.json"
    
    os.makedirs("/kaggle/working/data/processed", exist_ok=True)
    
    # Save train data
    with open(train_data_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    # Save validation data  
    with open(val_data_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    # Create dataset objects
    train_dataset = GeographicDataset(train_data_path, model.tokenizer)
    val_dataset = GeographicDataset(val_data_path, model.tokenizer)

    if len(train_data) == 0 or len(val_data) == 0:
        logger.warning("One or more splits are empty. Falling back to random split.")
        random.shuffle(processed_data)
        n = len(processed_data)
        train_end = int(n * 0.7)
        val_end = train_end + int(n * 0.15)
        train_data = processed_data[:train_end]
        val_data = processed_data[train_end:val_end]
        test_data = processed_data[val_end:]

    if len(train_data) == 0:
        logger.warning("Train split is empty!")
    if len(val_data) == 0:
        logger.warning("Validation split is empty!")
    if len(test_data) == 0:
        logger.warning("Test split is empty!")
    
    # Create a config object for GRPOTrainer
    trainer_config = GRPOTrainingConfig(
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        max_grad_norm=config['max_grad_norm'],
        geographic_loss_weight=0.1,  # or your value
        regional_balance_weight=0.05,  # or your value
        consistency_loss_weight=0.02,  # or your value
        eval_steps=config['eval_steps'],
        save_steps=config['save_steps'],
        logging_steps=config['logging_steps'],
        output_dir=config['output_dir'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mixed_precision=True,
    )

    trainer = GRPOTrainer(
        model=model,
        config=trainer_config
    )
    
    # Start training
    best_model_path = trainer.train(train_dataset, val_dataset)
    
    # Ensure we return a string path
    if best_model_path is None:
        # Fallback to the last saved checkpoint
        best_model_path = os.path.join(config['output_dir'], 'checkpoint-latest')
    
    return best_model_path

def save_model_for_download(model_path: str, output_name: str = "geolingua_model.pth"):
    """
    Save the trained model for download from Kaggle.
    
    Args:
        model_path: Path to the trained model
        output_name: Name for the downloadable file
    """
    import shutil
    
    output_path = f"/kaggle/working/{output_name}"
    shutil.copy(model_path, output_path)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Model saved for download: {output_path}")

def main():
    """Main training function for Kaggle."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting GeoLingua training on Kaggle...")
    
    try:
        # Load processed data
        processed_data = load_processed_data()
        
        # Initialize model
        model = initialize_model()
        
        # Create training config
        config = create_training_config()
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Train model
        best_model_path = train_model(model, processed_data, config)
        
        # Save model for download
        save_model_for_download(best_model_path)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best model: {best_model_path}")
        
        # Print training summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Total training examples: {len(processed_data)}")
        print(f"Model: {MODEL_NAME}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Epochs: {config['num_epochs']}")
        print(f"Best model saved: {best_model_path}")
        print("Model ready for download from Kaggle!")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    setup_logging()
    main() 
