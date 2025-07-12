"""
GeoLingua Utility Functions
Helper functions for data processing, model utilities, and geographic text analysis.
"""

import re
import os
import json
import pickle
import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class GeoLinguaLogger:
    """Custom logger for GeoLingua project."""
    
    def __init__(self, name: str = "GeoLingua", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)

class TextProcessor:
    """Text preprocessing utilities for geographic language data."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Geographic-specific stop words
        self.geo_stop_words = {
            'us_south': {'yall', 'youse', 'aint'},
            'uk': {'whilst', 'amongst', 'colour'},
            'australia': {'mate', 'arvo', 'brekkie'},
            'india': {'ji', 'na', 'hai'},
            'nigeria': {'abi', 'sha', 'wetin'}
        }
    
    def clean_text(self, text: str, preserve_case: bool = False) -> str:
        """Clean text while preserving geographic markers."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation but keep some for linguistic analysis
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Keep contractions and geographic markers
        if not preserve_case:
            text = text.lower()
        
        return text.strip()
    
    def extract_geographic_markers(self, text: str, region: str) -> List[str]:
        """Extract geographic linguistic markers from text."""
        markers = []
        text_lower = text.lower()
        
        # Regional dialect markers
        dialect_patterns = {
            'us_south': [
                r"\by'all\b", r"\bain't\b", r"\bfixin' to\b", r"\breckon\b",
                r"\bmight could\b", r"\buseta\b", r"\bover yonder\b"
            ],
            'uk': [
                r"\bbloke\b", r"\bmate\b", r"\bquid\b", r"\bprop\b",
                r"\blorry\b", r"\blifft\b", r"\bqueue\b", r"\bwhilst\b"
            ],
            'australia': [
                r"\bmate\b", r"\barvo\b", r"\bbrekkie\b", r"\bbarbbie\b",
                r"\bno worries\b", r"\bfair dinkum\b", r"\bshe'll be right\b"
            ],
            'india': [
                r"\bji\b", r"\bna\b", r"\bhai\b", r"\bkya\b",
                r"\bachha\b", r"\bbhai\b", r"\bdidi\b"
            ],
            'nigeria': [
                r"\babi\b", r"\bsha\b", r"\bwetin\b", r"\bcomot\b",
                r"\bchop\b", r"\bwaka\b", r"\bsabi\b"
            ]
        }
        
        if region in dialect_patterns:
            for pattern in dialect_patterns[region]:
                matches = re.findall(pattern, text_lower)
                markers.extend(matches)
        
        return markers
    
    def tokenize_with_geo_context(self, text: str, region: str) -> List[str]:
        """Tokenize text while preserving geographic context."""
        # Clean text
        cleaned_text = self.clean_text(text, preserve_case=True)
        
        # Extract geographic markers before tokenization
        geo_markers = self.extract_geographic_markers(cleaned_text, region)
        
        # Tokenize
        tokens = word_tokenize(cleaned_text)
        
        # Add geographic context tokens
        if geo_markers:
            tokens.extend([f"<{region}_marker>" for _ in geo_markers])
        
        return tokens
    
    def calculate_text_stats(self, text: str) -> Dict[str, Any]:
        """Calculate various text statistics."""
        sentences = sent_tokenize(text)
        tokens = word_tokenize(text)
        
        return {
            'char_count': len(text),
            'word_count': len(tokens),
            'sentence_count': len(sentences),
            'avg_sentence_length': np.mean([len(word_tokenize(s)) for s in sentences]),
            'unique_words': len(set(tokens)),
            'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0
        }

class ModelUtils:
    """Utilities for model training and evaluation."""
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> int:
        """Count trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                            epoch: int, loss: float, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }, path)
    
    @staticmethod
    def load_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                            path: str) -> Tuple[int, float]:
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    
    @staticmethod
    def calculate_perplexity(loss: float) -> float:
        """Calculate perplexity from loss."""
        return torch.exp(torch.tensor(loss)).item()
    
    @staticmethod
    def get_device() -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

class DataUtils:
    """Data handling utilities."""
    
    @staticmethod
    def save_data(data: Any, path: str, format: str = 'json') -> None:
        """Save data in specified format."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        elif format == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(path, index=False)
            else:
                pd.DataFrame(data).to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def load_data(path: str, format: str = 'json') -> Any:
        """Load data from specified format."""
        if format == 'json':
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif format == 'pickle':
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif format == 'csv':
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def create_data_splits(data: List[Any], train_ratio: float = 0.8,
                          val_ratio: float = 0.1, test_ratio: float = 0.1,
                          random_state: int = 42) -> Tuple[List[Any], List[Any], List[Any]]:
        """Create train/validation/test splits."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        np.random.seed(random_state)
        np.random.shuffle(data)
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return data[:train_end], data[train_end:val_end], data[val_end:]
    
    @staticmethod
    def balance_dataset_by_region(data: List[Dict[str, Any]], region_key: str = 'region') -> List[Dict[str, Any]]:
        """Balance dataset across geographic regions."""
        region_data = {}
        
        # Group by region
        for item in data:
            region = item[region_key]
            if region not in region_data:
                region_data[region] = []
            region_data[region].append(item)
        
        # Find minimum count
        min_count = min(len(items) for items in region_data.values())
        
        # Sample equally from each region
        balanced_data = []
        for region, items in region_data.items():
            np.random.shuffle(items)
            balanced_data.extend(items[:min_count])
        
        np.random.shuffle(balanced_data)
        return balanced_data

class VisualizationUtils:
    """Visualization utilities for analysis."""
    
    @staticmethod
    def plot_training_metrics(train_losses: List[float], val_losses: List[float],
                            save_path: str = None) -> None:
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', marker='o')
        plt.plot(val_losses, label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_regional_distribution(data: List[Dict[str, Any]], region_key: str = 'region',
                                 save_path: str = None) -> None:
        """Plot distribution of data across regions."""
        regions = [item[region_key] for item in data]
        region_counts = pd.Series(regions).value_counts()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=region_counts.index, y=region_counts.values)
        plt.xlabel('Region')
        plt.ylabel('Count')
        plt.title('Data Distribution Across Regions')
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_text_length_distribution(texts: List[str], save_path: str = None) -> None:
        """Plot distribution of text lengths."""
        text_lengths = [len(text.split()) for text in texts]
        
        plt.figure(figsize=(10, 6))
        plt.hist(text_lengths, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Text Length (words)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Text Lengths')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class EvaluationUtils:
    """Evaluation utilities for model assessment."""
    
    @staticmethod
    def calculate_bleu_score(predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score for text generation."""
        try:
            from nltk.translate.bleu_score import sentence_bleu
            scores = []
            for pred, ref in zip(predictions, references):
                pred_tokens = word_tokenize(pred.lower())
                ref_tokens = [word_tokenize(ref.lower())]
                score = sentence_bleu(ref_tokens, pred_tokens)
                scores.append(score)
            return np.mean(scores)
        except ImportError:
            print("NLTK not available for BLEU calculation")
            return 0.0
    
    @staticmethod
    def calculate_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """Calculate classification metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def create_confusion_matrix(y_true: List[str], y_pred: List[str],
                              save_path: str = None) -> None:
        """Create and display confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        labels = sorted(list(set(y_true + y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Utility functions for geographic context
def get_regional_config(region: str) -> Dict[str, Any]:
    """Get configuration for a specific region."""
    from config.data_config import REGIONS, REDDIT_SUBREDDITS
    
    return {
        'name': REGIONS.get(region, region),
        'subreddits': REDDIT_SUBREDDITS.get(region, []),
        'code': region
    }

def setup_environment() -> Dict[str, Any]:
    """Setup the environment and return system information."""
    device = ModelUtils.get_device()
    
    info = {
        'device': str(device),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
    
    return info

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# Initialize logger
logger = GeoLinguaLogger()

# Initialize utilities
text_processor = TextProcessor()
model_utils = ModelUtils()
data_utils = DataUtils()
viz_utils = VisualizationUtils()
eval_utils = EvaluationUtils()
