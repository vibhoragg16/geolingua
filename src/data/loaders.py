import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import logging
import os
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict

from config.data_config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, PROCESSED_DATA_PATH
from config.model_config import BATCH_SIZE, MAX_LENGTH

logger = logging.getLogger(__name__)

class GeoLinguaDataset(Dataset):
    """
    Custom Dataset class for GeoLingua training data.
    """
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get input and output text
        input_text = item['input']
        output_text = item['output']
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize output (labels)
        output_encoding = self.tokenizer(
            output_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': output_encoding['input_ids'].flatten(),
            'region': item['region'],
            'source_type': item.get('source_type', 'unknown'),
            'geographic_scores': item.get('geographic_scores', {}),
            'metadata': item.get('metadata', {})
        }


class ConversationDataset(Dataset):
    """
    Dataset for conversation-style training.
    """
    
    def __init__(self, conversations: List[List[Dict]], tokenizer, max_length: int = MAX_LENGTH):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # Build conversation text
        conversation_text = ""
        for turn in conversation:
            speaker = turn.get('speaker', 'User')
            text = turn.get('text', '')
            conversation_text += f"{speaker}: {text}\n"
        
        # Tokenize
        encoding = self.tokenizer(
            conversation_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # For language modeling, labels are the same as input_ids
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten(),
            'conversation_length': len(conversation),
            'region': conversation[0].get('region', 'unknown') if conversation else 'unknown'
        }


class DataLoader:
    """
    Main data loading and management class.
    """
    
    def __init__(self, data_path: str = PROCESSED_DATA_PATH):
        self.data_path = data_path
        self.datasets = {}
        self.data_stats = {}
        
    def load_processed_data(self, filename: str) -> List[Dict]:
        """
        Load processed data from JSON file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            List of processed data items
        """
        file_path = os.path.join(self.data_path, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} items from {filename}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {e}")
            return []
    
    def load_all_regions_data(self) -> Dict[str, List[Dict]]:
        """
        Load data for all regions.
        
        Returns:
            Dictionary mapping region names to their data
        """
        region_data = {}
        
        # Look for processed data files
        if os.path.exists(self.data_path):
            for filename in os.listdir(self.data_path):
                if filename.endswith('.json'):
                    # Extract region from filename (e.g., 'us_south_processed.json')
                    region = filename.replace('_processed.json', '').replace('.json', '')
                    data = self.load_processed_data(filename)
                    if data:
                        region_data[region] = data
        
        return region_data
    
    def combine_regions_data(self, region_data: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Combine data from multiple regions.
        
        Args:
            region_data: Dictionary of region data
            
        Returns:
            Combined dataset
        """
        combined_data = []
        
        for region, data in region_data.items():
            logger.info(f"Adding {len(data)} items from {region}")
            combined_data.extend(data)
        
        logger.info(f"Combined dataset size: {len(combined_data)}")
        return combined_data
    
    def split_data(self, data: List[Dict], stratify_by: str = 'region') -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split data into train/validation/test sets.
        
        Args:
            data: Dataset to split
            stratify_by: Field to stratify by
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if not data:
            return [], [], []
        
        # Create stratification labels
        if stratify_by in data[0]:
            stratify_labels = [item[stratify_by] for item in data]
        else:
            stratify_labels = None
        
        # First split: separate test set
        train_val_data, test_data = train_test_split(
            data,
            test_size=TEST_RATIO,
            stratify=stratify_labels,
            random_state=42
        )
        
        # Second split: separate train and validation
        if stratify_labels:
            train_val_labels = [item[stratify_by] for item in train_val_data]
        else:
            train_val_labels = None
        
        val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_size,
            stratify=train_val_labels,
            random_state=42
        )
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_data, val_data, test_data
    
    def create_dataloaders(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict], 
                          tokenizer, batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for training.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset
            tokenizer: Tokenizer instance
            batch_size: Batch size
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = GeoLinguaDataset(train_data, tokenizer)
        val_dataset = GeoLinguaDataset(val_data, tokenizer)
        test_dataset = GeoLinguaDataset(test_data, tokenizer)
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def analyze_dataset(self, data: List[Dict]) -> Dict:
        """
        Analyze dataset statistics.
        
        Args:
            data: Dataset to analyze
            
        Returns:
            Dictionary of statistics
        """
        if not data:
            return {}
        
        stats = {
            'total_items': len(data),
            'regions': defaultdict(int),
            'source_types': defaultdict(int),
            'word_counts': [],
            'geographic_scores': defaultdict(list)
        }
        
        for item in data:
            # Count by region
            region = item.get('region', 'unknown')
            stats['regions'][region] += 1
            
            # Count by source type
            source_type = item.get('source_type', 'unknown')
            stats['source_types'][source_type] += 1
            
            # Collect word counts
            if 'metadata' in item and 'word_count' in item['metadata']:
                stats['word_counts'].append(item['metadata']['word_count'])
            
            # Collect geographic scores
            geo_scores = item.get('geographic_scores', {})
            for region, score in geo_scores.items():
                stats['geographic_scores'][region].append(score)
        
        # Calculate statistics
        if stats['word_counts']:
            stats['avg_word_count'] = np.mean(stats['word_counts'])
            stats['median_word_count'] = np.median(stats['word_counts'])
            stats['min_word_count'] = np.min(stats['word_counts'])
            stats['max_word_count'] = np.max(stats['word_counts'])
        
        # Calculate average geographic scores
        avg_geo_scores = {}
        for region, scores in stats['geographic_scores'].items():
            if scores:
                avg_geo_scores[region] = np.mean(scores)
        stats['avg_geographic_scores'] = avg_geo_scores
        
        return dict(stats)
    
    def save_dataset_stats(self, stats: Dict, output_path: str) -> None:
        """
        Save dataset statistics to file.
        
        Args:
            stats: Statistics dictionary
            output_path: Output file path
        """
        try:
            # Convert defaultdict to regular dict for JSON serialization
            stats_serializable = {}
            for key, value in stats.items():
                if isinstance(value, defaultdict):
                    stats_serializable[key] = dict(value)
                else:
                    stats_serializable[key] = value
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats_serializable, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved dataset statistics to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving dataset statistics: {e}")
    
    def load_conversation_data(self, filename: str) -> List[List[Dict]]:
        """
        Load conversation data from JSON file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            List of conversations
        """
        file_path = os.path.join(self.data_path, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
            
            logger.info(f"Loaded {len(conversations)} conversations from {filename}")
            return conversations
            
        except Exception as e:
            logger.error(f"Error loading conversations from {filename}: {e}")
            return []
    
    def create_region_specific_loaders(self, region_data: Dict[str, List[Dict]], 
                                     tokenizer, batch_size: int = BATCH_SIZE) -> Dict[str, Tuple]:
        """
        Create region-specific data loaders.
        
        Args:
            region_data: Dictionary of region data
            tokenizer: Tokenizer instance
            batch_size: Batch size
            
        Returns:
            Dictionary mapping regions to their dataloaders
        """
        region_loaders = {}
        
        for region, data in region_data.items():
            if not data:
                continue
            
            # Split data for this region
            train_data, val_data, test_data = self.split_data(data)
            
            # Create dataloaders
            train_loader, val_loader, test_loader = self.create_dataloaders(
                train_data, val_data, test_data, tokenizer, batch_size
            )
            
            region_loaders[region] = (train_loader, val_loader, test_loader)
            
        return region_loaders
    
    def get_batch_by_region(self, data: List[Dict], region: str, batch_size: int = BATCH_SIZE) -> List[Dict]:
        """
        Get a batch of data for a specific region.
        
        Args:
            data: Full dataset
            region: Target region
            batch_size: Batch size
            
        Returns:
            Batch of data for the region
        """
        region_data = [item for item in data if item.get('region') == region]
        
        if not region_data:
            return []
        
        # Randomly sample batch_size items
        if len(region_data) >= batch_size:
            indices = np.random.choice(len(region_data), batch_size, replace=False)
            return [region_data[i] for i in indices]
        else:
            return region_data


def main():
    """Test the data loaders."""
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load data for all regions
    region_data = data_loader.load_all_regions_data()
    print(f"Loaded data for {len(region_data)} regions")
    
    # Combine all data
    combined_data = data_loader.combine_regions_data(region_data)
    
    # Analyze dataset
    stats = data_loader.analyze_dataset(combined_data)
    print("\nDataset Statistics:")
    print(f"Total items: {stats.get('total_items', 0)}")
    print(f"Regions: {dict(stats.get('regions', {}))}")
    print(f"Average word count: {stats.get('avg_word_count', 0):.2f}")
    print(f"Geographic scores: {stats.get('avg_geographic_scores', {})}")
    
    # Split data
    train_data, val_data, test_data = data_loader.split_data(combined_data)
    
    print(f"\nData split:")
    print(f"Train: {len(train_data)}")
    print(f"Val: {len(val_data)}")
    print(f"Test: {len(test_data)}")


if __name__ == "__main__":
    main()