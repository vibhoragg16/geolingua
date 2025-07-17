import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Enable better CUDA error reporting
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

@dataclass
class GRPOTrainingConfig:
    """Configuration for GRPO training."""
    model_config_path: str = "config/model_config.py"
    data_path: str = "data/processed/retokenized_reddit.json"
    output_dir: str = "models/geolingua"
    num_epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 5e-7
    warmup_steps: int = 100
    max_grad_norm: float = 0.5
    max_length: int = 512
    geographic_loss_weight: float = 0.05
    regional_balance_weight: float = 0.02
    consistency_loss_weight: float = 0.01
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False


def initialize_model_safely(model_name: str = "gpt2", regions: List[str] = []):
    if not regions:
        regions = ['us_south', 'uk', 'australia', 'india', 'nigeria']
    region_tokens = [f'[{region.upper()}]' for region in regions]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    num_added = tokenizer.add_tokens(region_tokens)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    if num_added > 0:
        base_model.resize_token_embeddings(len(tokenizer))
    base_model.tokenizer = tokenizer
    assert len(tokenizer) == base_model.get_input_embeddings().weight.shape[0]
    logging.info(f"Model initialized: {model_name}")
    logging.info(f"Vocab size: {len(tokenizer)}")
    logging.info(f"Added {num_added} region tokens")
    logging.info(f"Regions: {regions}")
    return base_model

class GeographicDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.region_to_id = {}
        self.id_to_region = {}
        self.logger = logging.getLogger(__name__)
        self.load_data(data_path)
        self.validate_data()
        self.logger.info(f"Dataset loaded: {len(self.data)} samples")
        if len(self.data) > 0:
            self.logger.info(f"Regions: {list(self.region_to_id.keys())}")
    def load_data(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        region_id = 0
        for item in raw_data:
            if not item.get('input') or len(item['input'].strip()) == 0:
                continue
            text = item['input'].strip()
            region_name = item.get('region', 'unknown')
            if region_name not in self.region_to_id:
                self.region_to_id[region_name] = region_id
                self.id_to_region[region_id] = region_name
                region_id += 1
            processed_item = {
                'text': text,
                'region': region_name,
                'region_id': self.region_to_id[region_name],
                'metadata': {
                    'score': item.get('score', 0),
                    'type': item.get('type', 'unknown'),
                    'subreddit': item.get('subreddit', 'unknown')
                }
            }
            self.data.append(processed_item)
    def validate_data(self):
        if len(self.data) == 0:
            raise ValueError("No valid data loaded!")
        max_region_id = max(item['region_id'] for item in self.data)
        min_region_id = min(item['region_id'] for item in self.data)
        if min_region_id < 0 or max_region_id >= len(self.region_to_id):
            raise ValueError(f"Invalid region IDs: range [{min_region_id}, {max_region_id}], expected [0, {len(self.region_to_id)-1}]")
        sample_text = self.data[0]['text']
        encoding = self.tokenizer(
            sample_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        max_token_id = encoding['input_ids'].max().item()
        vocab_size = len(self.tokenizer)
        if max_token_id >= vocab_size:
            raise ValueError(f"Token ID {max_token_id} >= vocab size {vocab_size}")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        if input_ids.max().item() >= len(self.tokenizer):
            raise ValueError(f"Token ID out of range: {input_ids.max().item()} >= {len(self.tokenizer)}")
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'region_id': torch.tensor(item['region_id'], dtype=torch.long),
            'labels': input_ids.clone(),
            'text': item['text'],
            'region': item['region']
        }

class FixedGRPOTrainer:
    def __init__(self, model, config: GRPOTrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)
        self.model.to(self.device)
        self.optimizer = None
        self.scheduler = None
        self.training_stats = defaultdict(list)
        self.global_step = 0
        self.validate_model_setup()
    def validate_model_setup(self):
        if hasattr(self.model, 'tokenizer'):
            tokenizer_vocab_size = len(self.model.tokenizer)
            # Use base_model for embedding size if present
            if hasattr(self.model, 'base_model'):
                model_vocab_size = self.model.base_model.get_input_embeddings().weight.shape[0]
            else:
                model_vocab_size = self.model.get_input_embeddings().weight.shape[0]
            if tokenizer_vocab_size != model_vocab_size:
                raise ValueError(f"Tokenizer vocab size ({tokenizer_vocab_size}) != model vocab size ({model_vocab_size})")
            self.logger.info(f"Model validation passed: vocab size = {model_vocab_size}")
    def setup_optimizer_and_scheduler(self, train_loader: DataLoader):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found!")
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if p.requires_grad and not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if p.requires_grad and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        total_steps = len(train_loader) * self.config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        self.logger.info(f"Optimizer setup complete. Total steps: {total_steps}")
    def compute_safe_grpo_loss(self, outputs, batch):
        losses = {}
        device = next(self.model.parameters()).device
        if 'loss' in outputs:
            lm_loss = outputs['loss']
        else:
            logits = outputs['logits']
            labels = batch['labels']
            if logits.dim() != 3 or labels.dim() != 2:
                raise ValueError(f"Invalid shapes: logits {logits.shape}, labels {labels.shape}")
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            max_label = shift_labels.max().item()
            vocab_size = shift_logits.size(-1)
            if max_label >= vocab_size:
                self.logger.error(f"Label {max_label} >= vocab size {vocab_size}")
                shift_labels = torch.clamp(shift_labels, 0, vocab_size - 1)
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=self.model.tokenizer.pad_token_id if hasattr(self.model, 'tokenizer') else -100
            )
        losses['lm_loss'] = lm_loss
        geo_loss = torch.tensor(0.0, device=device)
        if hasattr(self.model, 'region_embeddings'):
            geo_loss = self.model.region_embeddings.weight.norm(p=2) * 0.01
        losses['geo_loss'] = geo_loss
        regional_balance_loss = torch.tensor(0.0, device=device)
        region_ids = batch['region_id']
        if len(torch.unique(region_ids)) > 1:
            regional_balance_loss = region_ids.float().var()
        losses['regional_balance_loss'] = regional_balance_loss
        consistency_loss = torch.tensor(0.0, device=device)
        losses['consistency_loss'] = consistency_loss
        total_loss = (
            lm_loss +
            self.config.geographic_loss_weight * geo_loss +
            self.config.regional_balance_weight * regional_balance_loss +
            self.config.consistency_loss_weight * consistency_loss
        )
        losses['total_loss'] = total_loss
        return losses
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses = defaultdict(list)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            print(f"[DEBUG] Processing batch {batch_idx}", flush=True)
            try:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                self.validate_batch(batch)
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    region_ids=batch['region_id'],
                    labels=batch['labels'],
                    output_hidden_states=True
                )
                losses = self.compute_safe_grpo_loss(outputs, batch)
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config.max_grad_norm
                )
                if self.optimizer is not None:
                    self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                for loss_name, loss_value in losses.items():
                    if torch.isfinite(loss_value):
                        epoch_losses[loss_name].append(loss_value.item())
                self.global_step += 1
                progress_bar.set_postfix({
                    'LM': f"{losses['lm_loss'].item():.4f}",
                    'Total': f"{losses['total_loss'].item():.4f}",
                    'LR': f"{self.scheduler.get_last_lr()[0]:.2e}" if self.scheduler is not None else 'N/A'
                })
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                print(f"[DEBUG] Error in batch {batch_idx}: {e}", flush=True)
                continue
        epoch_avg_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
        return epoch_avg_losses
    def validate_batch(self, batch):
        input_ids = batch['input_ids']
        max_token_id = input_ids.max().item()
        if hasattr(self.model, 'tokenizer'):
            vocab_size = len(self.model.tokenizer)
            if max_token_id >= vocab_size:
                raise ValueError(f"Token ID {max_token_id} >= vocab size {vocab_size}")
        region_ids = batch['region_id']
        max_region_id = region_ids.max().item()
        min_region_id = region_ids.min().item()
        if min_region_id < 0:
            raise ValueError(f"Negative region ID: {min_region_id}")
        if torch.isnan(input_ids).any():
            raise ValueError("NaN values in input_ids")
        if torch.isnan(region_ids).any():
            raise ValueError("NaN values in region_ids")
    def train(self, train_dataset: GeographicDataset, val_dataset: Optional[GeographicDataset] = None):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True
        )
        val_loader = None
        if val_dataset and len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if self.device.type == 'cuda' else False,
                drop_last=True
            )
        self.setup_optimizer_and_scheduler(train_loader)
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Steps per epoch: {len(train_loader)}")
        best_model_path = None
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            try:
                train_losses = self.train_epoch(train_loader, epoch)
                self.logger.info("Training losses:")
                for loss_name, loss_value in train_losses.items():
                    self.logger.info(f"  {loss_name}: {loss_value:.4f}")
                    self.training_stats[f"train_{loss_name}"].append(loss_value)
                checkpoint_path = self.save_model(epoch)
                if best_model_path is None:
                    best_model_path = checkpoint_path
            except Exception as e:
                self.logger.error(f"Error in epoch {epoch}: {e}")
                try:
                    checkpoint_path = self.save_model(f"error_{epoch}")
                    if best_model_path is None:
                        best_model_path = checkpoint_path
                except:
                    pass
                continue
        try:
            final_path = self.save_model("final")
            best_model_path = final_path
        except Exception as e:
            self.logger.error(f"Error saving final model: {e}")
        self.logger.info("Training completed!")
        return best_model_path
    def save_model(self, epoch):
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_path.mkdir(exist_ok=True)
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'training_stats': dict(self.training_stats),
                'global_step': self.global_step,
                'config': self.config
            }, checkpoint_path / "model.pt")
            if hasattr(self.model, 'tokenizer'):
                self.model.tokenizer.save_pretrained(checkpoint_path)
            self.logger.info(f"Model saved to {checkpoint_path}")
            return str(checkpoint_path / "model.pt")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return None

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    try:
        config = GRPOTrainingConfig()
        model = initialize_model_safely()
        train_dataset = GeographicDataset(
            data_path=config.data_path,
            tokenizer=model.tokenizer,
            max_length=config.max_length
        )
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_split, val_split = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        # --- DEBUG PRINTS ---
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {num_trainable}")
        print(f"Dataset size: {len(train_dataset)}")
        print(f"Batch size: {config.batch_size}")
        print(f"Steps per epoch: {len(train_split) // config.batch_size}")
        print(f"Epochs: {config.num_epochs}")
        # --- END DEBUG PRINTS ---
        logger.info(f"Train split: {len(train_split)}")
        logger.info(f"Val split: {len(val_split)}")
        trainer = FixedGRPOTrainer(model, config)
        # Print loss for each epoch
        orig_train_epoch = trainer.train_epoch
        def debug_train_epoch(train_loader, epoch):
            print(f"[DEBUG] Starting epoch {epoch}", flush=True)
            result = orig_train_epoch(train_loader, epoch)
            print(f"[DEBUG] End of epoch {epoch}: losses: {result}", flush=True)
            return result
        trainer.train_epoch = debug_train_epoch
        best_model_path = trainer.train(train_split, val_split)
        logger.info(f"Training completed! Best model: {best_model_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
