#!/usr/bin/env python3
"""
Demo script showing proper data splitting for GeoLingua project.
This demonstrates the importance of train/validation/test splits.
"""

import os
import sys
import json
import random
from collections import defaultdict
from typing import List, Dict, Tuple

# Add src and config to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
src_path = os.path.join(project_root, 'src')
config_path = os.path.join(project_root, 'config')

sys.path.insert(0, src_path)
sys.path.insert(0, config_path)

def load_processed_data(data_path: str = "data/processed/processed_dataset.json"):
    """Load processed data for demonstration."""
    if not os.path.exists(data_path):
        print(f"Error: Processed data not found at {data_path}")
        print("Please run preprocessing first: python scripts/preprocess.py")
        return []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def split_data_stratified(processed_data: List[Dict], 
                         train_ratio: float = 0.7, 
                         val_ratio: float = 0.15, 
                         test_ratio: float = 0.15,
                         random_seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train/validation/test sets with stratification by region.
    
    This ensures each region is represented proportionally in all splits.
    """
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
    
    print("Data distribution by region:")
    for region, items in region_data.items():
        print(f"  {region}: {len(items)} examples")
    
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
        
        print(f"  {region}: train={train_end}, val={val_end-train_end}, test={n_items-val_end}")
    
    # Shuffle the final splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data

def analyze_splits(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
    """Analyze the data splits and show statistics."""
    
    print("\n" + "="*60)
    print("DATA SPLIT ANALYSIS")
    print("="*60)
    
    # Overall statistics
    total_examples = len(train_data) + len(val_data) + len(test_data)
    print(f"Total examples: {total_examples}")
    print(f"Train: {len(train_data)} ({len(train_data)/total_examples*100:.1f}%)")
    print(f"Validation: {len(val_data)} ({len(val_data)/total_examples*100:.1f}%)")
    print(f"Test: {len(test_data)} ({len(test_data)/total_examples*100:.1f}%)")
    
    # Regional distribution analysis
    print("\nRegional distribution in each split:")
    print("-" * 50)
    
    splits = {
        'Train': train_data,
        'Validation': val_data,
        'Test': test_data
    }
    
    for split_name, split_data in splits.items():
        region_counts = defaultdict(int)
        for item in split_data:
            region = item.get('region', 'unknown')
            region_counts[region] += 1
        
        print(f"\n{split_name} split:")
        for region in sorted(region_counts.keys()):
            count = region_counts[region]
            percentage = count / len(split_data) * 100 if split_data else 0
            print(f"  {region:12}: {count:3d} examples ({percentage:5.1f}%)")

def explain_benefits():
    """Explain the benefits of proper data splitting."""
    
    print("\n" + "="*60)
    print("WHY PROPER DATA SPLITTING IS IMPORTANT")
    print("="*60)
    
    benefits = [
        "1. **Prevents Overfitting**: Test set provides unbiased evaluation",
        "2. **Model Selection**: Validation set helps choose best hyperparameters",
        "3. **Fair Comparison**: Same test set for all model versions",
        "4. **Stratification**: Ensures all regions are represented in all splits",
        "5. **Reproducibility**: Fixed random seed ensures consistent splits",
        "6. **Realistic Performance**: Test set simulates real-world deployment"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    print("\n**Best Practices:**")
    print("  - Never use test set during training")
    print("  - Use validation set for hyperparameter tuning")
    print("  - Keep test set completely separate")
    print("  - Stratify by important features (region in our case)")
    print("  - Use fixed random seed for reproducibility")

def save_splits(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
    """Save the data splits to files."""
    
    output_dir = "data/processed"
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
        print(f"Saved {split_name} split to {output_path}")

def main():
    """Main demonstration function."""
    print("GeoLingua Data Splitting Demo")
    print("="*40)
    
    # Load data
    processed_data = load_processed_data()
    if not processed_data:
        return
    
    print(f"Loaded {len(processed_data)} examples")
    
    # Split data
    print("\nSplitting data with stratification by region...")
    train_data, val_data, test_data = split_data_stratified(
        processed_data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    # Analyze splits
    analyze_splits(train_data, val_data, test_data)
    
    # Explain benefits
    explain_benefits()
    
    # Save splits
    print("\nSaving data splits...")
    save_splits(train_data, val_data, test_data)
    
    print("\n" + "="*60)
    print("SPLITTING COMPLETE!")
    print("="*60)
    print("You can now use these splits for training:")
    print("  - train_data: For model training")
    print("  - val_data: For validation during training")
    print("  - test_data: For final evaluation (only after training)")
    print("\nFiles saved:")
    print("  - data/processed/train_split.json")
    print("  - data/processed/val_split.json")
    print("  - data/processed/test_split.json")

if __name__ == "__main__":
    main() 