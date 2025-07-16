#!/usr/bin/env python3
"""
Evaluation script for GeoLingua model.
Evaluates the trained model on the held-out test set.
"""

import os
import sys
import json
import logging
import torch
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

# Add src and config to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
src_path = os.path.join(project_root, 'src')
config_path = os.path.join(project_root, 'config')

sys.path.insert(0, src_path)
sys.path.insert(0, config_path)

from src.models.basemodel import GeoLinguaModel
from config.model_config import *
from config.data_config import *

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_test_data(test_path: str = "data/processed/test_split.json"):
    """
    Load the test dataset.
    
    Args:
        test_path: Path to the test split file
        
    Returns:
        List of test examples
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}. Please run training first to create splits.")
    
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        logger.info(f"Loaded {len(test_data)} test examples from {test_path}")
        return test_data
        
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

def load_trained_model(model_path: str) -> GeoLinguaModel:
    """
    Load the trained model.
    
    Args:
        model_path: Path to the trained model checkpoint
        
    Returns:
        Loaded GeoLinguaModel
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Initialize model with same configuration as training
    model = GeoLinguaModel(
        model_name=MODEL_NAME,
        regions=['us_south', 'uk', 'australia', 'india', 'nigeria'],
        lora_config={
            'r': LORA_R,
            'lora_alpha': LORA_ALPHA,
            'lora_dropout': LORA_DROPOUT,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        }
    )
    
    # Load trained weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded trained model from {model_path}")
    logger.info(f"Model on device: {device}")
    
    return model

def create_test_dataset(test_data, tokenizer, max_length):
    class TestDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            text = item.get('text', '')
            region = item.get('region', 'unknown')
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': inputs['input_ids'].squeeze(0),
                'region': region
            }
    return TestDataset(test_data, tokenizer, max_length)

def evaluate_model(model: GeoLinguaModel, test_data: List[Dict]) -> Dict:
    """
    Evaluate the model on test data using batching for speed.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation (batched)...")

    # Group test data by region
    region_data = defaultdict(list)
    for item in test_data:
        region = item.get('region', 'unknown')
        region_data[region].append(item)

    # Evaluation metrics
    total_loss = 0.0
    region_metrics = defaultdict(lambda: {'loss': 0.0, 'count': 0})
    batch_size = 16  # You can adjust this based on your GPU/CPU memory
    max_length = getattr(model, 'max_length', 512)

    model.eval()
    with torch.no_grad():
        for region, items in region_data.items():
            logger.info(f"Evaluating region: {region} ({len(items)} examples)")
            region_loss = 0.0
            if len(items) == 0:
                continue
            dataset = create_test_dataset(items, model.tokenizer, max_length)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for batch in loader:
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['labels'].to(model.device)
                # If you have region_id mapping, use it here. For now, set all to 0.
                region_ids = torch.zeros(input_ids.size(0), dtype=torch.long).to(model.device)
                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        region_ids=region_ids,
                        labels=labels
                    )
                    loss = outputs.loss
                    region_loss += loss.item() * input_ids.size(0)
                    total_loss += loss.item() * input_ids.size(0)
                    region_metrics[region]['loss'] += loss.item() * input_ids.size(0)
                    region_metrics[region]['count'] += input_ids.size(0)
                except Exception as e:
                    logger.warning(f"Error evaluating batch: {e}")
                    continue

    # Calculate average losses
    total_examples = len(test_data)
    avg_loss = total_loss / total_examples if total_examples else 0.0
    for region in region_metrics:
        if region_metrics[region]['count'] > 0:
            region_metrics[region]['avg_loss'] = region_metrics[region]['loss'] / region_metrics[region]['count']

    # Compile results
    results = {
        'overall': {
            'total_examples': total_examples,
            'average_loss': avg_loss
        },
        'by_region': dict(region_metrics)
    }
    return results

def print_evaluation_results(results: Dict):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    overall = results['overall']
    print(f"Total test examples: {overall['total_examples']}")
    print(f"Overall average loss: {overall['average_loss']:.4f}")
    
    print("\nResults by region:")
    print("-" * 40)
    
    for region, metrics in results['by_region'].items():
        if metrics['count'] > 0:
            print(f"{region:12}: {metrics['count']:3d} examples, "
                  f"avg loss: {metrics['avg_loss']:.4f}")
    
    print("="*60)

def save_evaluation_results(results: Dict, output_path: str = "evaluation_results.json"):
    """
    Save evaluation results to a file.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save results
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluation results saved to {output_path}")

def main():
    """Main evaluation function."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting GeoLingua model evaluation...")
    
    try:
        # Load test data
        test_data = load_test_data()
        
        # Load trained model
        model_path = "models/final/geolingua_model.pth"  # Adjust path as needed
        model = load_trained_model(model_path)
        
        # Evaluate model
        results = evaluate_model(model, test_data)
        
        # Print results
        print_evaluation_results(results)
        
        # Save results
        save_evaluation_results(results)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    setup_logging()
    main() 
