import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import json
from pathlib import Path
# import wandb  # Commented out for Kaggle/no-wandb use
from tqdm import tqdm
import logging
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt

@dataclass
class GRPOTrainingConfig:
    """Configuration for GRPO training."""
    
    # Model and data paths
    model_config_path: str = "config/model_config.py"
    data_path: str = "data/processed"
    output_dir: str = "models/geolingua"
    
    # Training parameters
    num_epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # GRPO specific parameters
    geographic_loss_weight: float = 0.1
    regional_balance_weight: float = 0.05
    consistency_loss_weight: float = 0.02
    
    # Evaluation parameters
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Hardware settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True


class GeographicDataset(Dataset):
    """Dataset for geographic language modeling."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.region_to_id = {}
        self.id_to_region = {}
        
        # Setup logging first
        self.setup_logging()
        # Then load data
        self.load_data(data_path)
        # Debug: log dataset size and first sample
        print(f"[DEBUG] GeographicDataset loaded {len(self.data)} samples.")
        if len(self.data) > 0:
            print(f"[DEBUG] First sample: {self.data[0]}")
        else:
            print("[DEBUG] No data loaded in GeographicDataset!")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, data_path: str):
        """Load data from JSON files."""
        data_path_obj = Path(data_path)
        region_id = 0
        
        # Check if it's a single JSON file or a directory
        if data_path_obj.is_file():
            # Single JSON file - load all data
            with open(data_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            
            # Process each item
            for item in all_data:
                # Relaxed (for debugging)
                if item.get('input') and len(item['input'].strip()) > 0:
                    region_name = item.get('region', 'unknown')
                    
                    # Map region names to IDs
                    if region_name not in self.region_to_id:
                        self.region_to_id[region_name] = region_id
                        self.id_to_region[region_id] = region_name
                        region_id += 1
                    
                    processed_item = {
                        'text': item['input'],  # Use 'input' as the text
                        'region': region_name,
                        'region_id': self.region_to_id[region_name],
                        'metadata': {
                            'score': item.get('score', 0),
                            'type': item.get('type', 'unknown'),
                            'subreddit': item.get('subreddit', 'unknown')
                        }
                    }
                    self.data.append(processed_item)
        
        else:
            # Directory with multiple files (original behavior)
            data_dir = data_path_obj
            for json_file in data_dir.glob("*_reddit_data.json"):
                region_name = json_file.stem.replace("_reddit_data", "")
                
                # Map region names to IDs
                if region_name not in self.region_to_id:
                    self.region_to_id[region_name] = region_id
                    self.id_to_region[region_id] = region_name
                    region_id += 1
                
                # Load data
                with open(json_file, 'r') as f:
                    region_data = json.load(f)
                
                # Process each text sample
                for item in region_data:
                    # Relaxed (for debugging)
                    if item.get('input') and len(item['input'].strip()) > 0:
                        processed_item = {
                            'text': item['input'],  # Use 'input' as the text
                            'region': region_name,
                            'region_id': self.region_to_id[region_name],
                            'metadata': {
                                'score': item.get('score', 0),
                                'type': item.get('type', 'unknown'),
                                'subreddit': item.get('subreddit', 'unknown')
                            }
                        }
                        self.data.append(processed_item)
        
        self.logger.info(f"Loaded {len(self.data)} samples from {len(self.region_to_id)} regions")
        self.logger.info(f"Regions: {list(self.region_to_id.keys())}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'region_id': torch.tensor(item['region_id'], dtype=torch.long),
            'labels': encoding['input_ids'].squeeze(),  # For language modeling
            'text': item['text'],
            'region': item['region']
        }


class GRPOTrainer:
    """GRPO (Geographically Restricted Pre-trained Optimization) Trainer."""
    
    def __init__(self, model, config: GRPOTrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        
        # Training metrics
        self.training_stats = defaultdict(list)
        self.global_step = 0
        
        # Setup logging
        self.setup_logging()
        
        # Initialize wandb if available
        # try:
        #     wandb.init(
        #         project="geolingua-grpo",
        #         config=config.__dict__,
        #         name=f"grpo_training_{config.num_epochs}ep"
        #     )
        #     self.use_wandb = True
        # except:
        #     self.use_wandb = False
        #     self.logger.info("W&B not available, continuing without logging")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_dataloaders(self, train_dataset: GeographicDataset, 
                          val_dataset: Optional[GeographicDataset] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create training and validation dataloaders."""
        # Debug: log dataset sizes before DataLoader creation
        print(f"[DEBUG] train_dataset size: {len(train_dataset)}")
        if val_dataset:
            print(f"[DEBUG] val_dataset size: {len(val_dataset)}")
        if len(train_dataset) > 0:
            print(f"[DEBUG] First train sample: {train_dataset[0]}")
        else:
            print("[DEBUG] Train dataset is EMPTY before DataLoader creation!")
        if val_dataset and len(val_dataset) > 0:
            print(f"[DEBUG] First val sample: {val_dataset[0]}")
        elif val_dataset:
            print("[DEBUG] Val dataset is EMPTY before DataLoader creation!")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        return train_loader, val_loader
    
    def setup_optimizer_and_scheduler(self, train_loader: DataLoader):
        """Setup optimizer and learning rate scheduler."""
        
        # Setup optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-8
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
    
    def compute_grpo_loss(self, outputs: Dict[str, torch.Tensor], 
                         batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute GRPO loss with geographic regularization.
        
        Args:
            outputs: Model outputs
            batch: Training batch
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Base language modeling loss
        lm_loss = outputs['lm_loss']
        losses['lm_loss'] = lm_loss
        
        # Geographic classification loss
        geo_loss = outputs['geo_loss']
        losses['geo_loss'] = geo_loss
        
        # Regional balance loss - encourages similar performance across regions
        regional_balance_loss = self.compute_regional_balance_loss(outputs, batch)
        losses['regional_balance_loss'] = regional_balance_loss
        
        # Consistency loss - encourages similar representations for similar content
        consistency_loss = self.compute_consistency_loss(outputs, batch)
        losses['consistency_loss'] = consistency_loss
        
        # Total GRPO loss
        total_loss = (
            lm_loss +
            self.config.geographic_loss_weight * geo_loss +
            self.config.regional_balance_weight * regional_balance_loss +
            self.config.consistency_loss_weight * consistency_loss
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def compute_regional_balance_loss(self, outputs: Dict[str, torch.Tensor], 
                                    batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute regional balance loss to ensure fairness across regions."""
        
        # Get per-sample losses
        lm_logits = outputs['logits']
        labels = batch['labels']
        region_ids = batch['region_id']
        
        # Compute per-sample losses
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute losses without reduction
        per_sample_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )
        
        # Reshape to get per-sample losses
        per_sample_losses = per_sample_losses.view(labels.size(0), -1).mean(dim=1)
        
        # Compute regional variance
        unique_regions = torch.unique(region_ids)
        regional_losses = []
        
        for region_id in unique_regions:
            region_mask = region_ids == region_id
            if region_mask.sum() > 0:
                region_loss = per_sample_losses[region_mask].mean()
                regional_losses.append(region_loss)
        
        if len(regional_losses) > 1:
            regional_losses = torch.stack(regional_losses)
            balance_loss = regional_losses.var()
        else:
            balance_loss = torch.tensor(0.0, device=per_sample_losses.device)
        
        return balance_loss
    
    def compute_consistency_loss(self, outputs: Dict[str, torch.Tensor], 
                               batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute consistency loss to maintain coherent representations."""
        
        hidden_states = outputs['hidden_states']
        region_ids = batch['region_id']
        
        # Pool hidden states to get sentence representations
        pooled_states = hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Compute pairwise similarities within regions
        consistency_loss = torch.tensor(0.0, device=hidden_states.device)
        unique_regions = torch.unique(region_ids)
        
        for region_id in unique_regions:
            region_mask = region_ids == region_id
            region_states = pooled_states[region_mask]
            
            if region_states.size(0) > 1:
                # Compute pairwise cosine similarities
                normalized_states = F.normalize(region_states, p=2, dim=1)
                similarity_matrix = torch.mm(normalized_states, normalized_states.t())
                
                # Encourage high similarity within regions
                target_similarity = torch.ones_like(similarity_matrix)
                region_consistency = F.mse_loss(similarity_matrix, target_similarity)
                consistency_loss += region_consistency
        
        return consistency_loss
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        epoch_losses = defaultdict(list)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}

            # Print number of non-padding labels in batch
            if batch_idx == 0:
                num_nonpad = (batch['labels'] != -100).sum().item()
                print(f"Non-padding labels in batch: {num_nonpad}")

            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                region_ids=batch['region_id'],
                labels=batch['labels']
            )
            # Check for NaN values in outputs
            if batch_idx == 0:
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        print(f"{k}: has nan? {torch.isnan(v).any().item()}")
            
            # Compute GRPO losses
            losses = self.compute_grpo_loss(outputs, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            # Gradient clipping (lowered max_norm)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            # Check for NaN in gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients of {name}")
            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            
            # Log metrics
            for loss_name, loss_value in losses.items():
                epoch_losses[loss_name].append(loss_value.item())
            
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'LM Loss': f"{losses['lm_loss'].item():.4f}",
                'Geo Loss': f"{losses['geo_loss'].item():.4f}",
                'Total Loss': f"{losses['total_loss'].item():.4f}"
            })
            
            # Log to wandb
            # if self.use_wandb and self.global_step % self.config.logging_steps == 0:
            #     log_dict = {f"train/{k}": v.item() for k, v in losses.items()}
            #     log_dict['train/learning_rate'] = self.scheduler.get_last_lr()[0]
            #     wandb.log(log_dict, step=self.global_step)
            
            if batch_idx == 0:
                print("input_ids:", batch['input_ids'])
                print("labels:", batch['labels'])
        
        # Compute epoch averages
        epoch_avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        return epoch_avg_losses
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        
        self.model.eval()
        eval_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    region_ids=batch['region_id'],
                    labels=batch['labels']
                )
                
                # Compute losses
                losses = self.compute_grpo_loss(outputs, batch)
                
                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    eval_losses[loss_name].append(loss_value.item())
        
        # Compute averages
        eval_avg_losses = {k: np.mean(v) for k, v in eval_losses.items()}
        
        return eval_avg_losses
    
    def train(self, train_dataset: GeographicDataset, 
              val_dataset: Optional[GeographicDataset] = None):
        """Main training loop."""
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(train_dataset, val_dataset)
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler(train_loader)
        
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            self.logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train epoch
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Log training losses
            self.logger.info("Training losses:")
            for loss_name, loss_value in train_losses.items():
                self.logger.info(f"  {loss_name}: {loss_value:.4f}")
                self.training_stats[f"train_{loss_name}"].append(loss_value)
            
            # Evaluation
            if val_loader:
                eval_losses = self.evaluate(val_loader)
                self.logger.info("Validation losses:")
                for loss_name, loss_value in eval_losses.items():
                    self.logger.info(f"  {loss_name}: {loss_value:.4f}")
                    self.training_stats[f"val_{loss_name}"].append(loss_value)
                
                # Log to wandb
                # if self.use_wandb:
                #     log_dict = {f"val/{k}": v for k, v in eval_losses.items()}
                #     wandb.log(log_dict, step=self.global_step)
            
            # Save model
            if (epoch + 1) % (self.config.save_steps // len(train_loader)) == 0:
                self.save_model(epoch)
        
        # Save final model
        self.save_model("final")
        
        self.logger.info("Training completed!")
    
    def save_model(self, epoch):
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = output_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model state
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_stats': dict(self.training_stats),
            'global_step': self.global_step
        }, checkpoint_path / "model.pt")
        
        # Save tokenizer
        self.model.tokenizer.save_pretrained(checkpoint_path)
        
        self.logger.info(f"Model saved to {checkpoint_path}")
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Language modeling loss
        axes[0, 0].plot(self.training_stats['train_lm_loss'], label='Train')
        if 'val_lm_loss' in self.training_stats:
            axes[0, 0].plot(self.training_stats['val_lm_loss'], label='Val')
        axes[0, 0].set_title('Language Modeling Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Geographic loss
        axes[0, 1].plot(self.training_stats['train_geo_loss'], label='Train')
        if 'val_geo_loss' in self.training_stats:
            axes[0, 1].plot(self.training_stats['val_geo_loss'], label='Val')
        axes[0, 1].set_title('Geographic Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Regional balance loss
        axes[1, 0].plot(self.training_stats['train_regional_balance_loss'], label='Train')
        if 'val_regional_balance_loss' in self.training_stats:
            axes[1, 0].plot(self.training_stats['val_regional_balance_loss'], label='Val')
        axes[1, 0].set_title('Regional Balance Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        
        # Total loss
        axes[1, 1].plot(self.training_stats['train_total_loss'], label='Train')
        if 'val_total_loss' in self.training_stats:
            axes[1, 1].plot(self.training_stats['val_total_loss'], label='Val')
        axes[1, 1].set_title('Total GRPO Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(Path(self.config.output_dir) / "training_curves.png")
        plt.show()


# Example usage
def main():
    """Main training script."""
    
    # Load configuration
    config = GRPOTrainingConfig()
    
    # Load model (you would import your GeographicAdapter here)
    from geographic_adapter import GeographicAdapter, GeographicAdapterConfig
    
    model_config = GeographicAdapterConfig()
    model = GeographicAdapter(model_config)
    
    # Load datasets
    train_dataset = GeographicDataset(
        data_path="data/raw/reddit",
        tokenizer=model.tokenizer,
        max_length=model_config.max_length
    )
    
    # Split dataset for validation (simple split for demo)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # After saving splits, add:
    print(f"Train split size: {len(train_dataset)}")
    print(f"Val split size: {len(val_dataset)}")
    if train_dataset:
        print("First train sample:", train_dataset[0])
    else:
        print("Train data is EMPTY!")
    if val_dataset:
        print("First val sample:", val_dataset[0])
    else:
        print("Val data is EMPTY!")
    
    # Initialize trainer
    trainer = GRPOTrainer(model, config)
    
    # Start training
    trainer.train(train_dataset, val_dataset)
    
    # Plot training curves
    trainer.plot_training_curves()

if __name__ == "__main__":
    main()
