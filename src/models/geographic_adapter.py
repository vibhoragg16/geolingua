import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model, TaskType

@dataclass
class GeographicAdapterConfig:
    """Configuration for the Geographic Adapter."""
    
    # Model configuration
    base_model_name: str = "gpt2"
    max_length: int = 512
    
    # Geographic configuration
    num_regions: int = 5
    region_embedding_dim: int = 64
    geographic_hidden_dim: int = 256
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None  # Allow None, will be set in __post_init__
    
    # Training configuration
    use_geographic_loss: bool = True
    geographic_loss_weight: float = 0.1
    
    def __post_init__(self):
        # For GPT-2, use 'c_attn' and 'c_proj' as LoRA target modules
        if self.lora_target_modules is None:
            self.lora_target_modules = ["c_attn", "c_proj"]


class GeographicEmbedding(nn.Module):
    """Geographic embedding layer that encodes regional context."""
    
    def __init__(self, num_regions: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.num_regions = num_regions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # Region embedding lookup
        self.region_embedding = nn.Embedding(num_regions, embedding_dim)
        # Geographic context processor
        self.geographic_processor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        # Project region embedding to hidden_dim for attention
        self.region_to_hidden = nn.Linear(embedding_dim, hidden_dim)
        # Geographic attention for adaptive weighting
        self.geographic_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, region_ids: torch.Tensor, 
                text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for geographic embedding.
        
        Args:
            region_ids: Tensor of shape (batch_size,) with region IDs
            text_embeddings: Tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Geographic context embeddings
        """
        # Get region embeddings
        region_emb = self.region_embedding(region_ids)  # (batch_size, embedding_dim)
        # Process geographic context
        geo_context = self.geographic_processor(region_emb)  # (batch_size, embedding_dim)
        # Expand for sequence length
        batch_size, seq_len, hidden_dim = text_embeddings.shape
        geo_context_expanded = geo_context.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, embedding_dim)
        # Project region embedding to hidden_dim for attention
        geo_query = self.region_to_hidden(geo_context_expanded)  # (batch, seq_len, hidden_dim)
        # Apply geographic attention
        geo_attended, _ = self.geographic_attention(
            query=geo_query,
            key=text_embeddings,
            value=text_embeddings
        )
        return geo_attended


class GeographicAdapter(nn.Module):
    """Geographic adapter that modifies model behavior based on regional context."""
    
    def __init__(self, config: GeographicAdapterConfig):
        super().__init__()
        self.config = config
        
        # Load base model (GPT-2 by default)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Geographic components
        self.geographic_embedding = GeographicEmbedding(
            num_regions=config.num_regions,
            embedding_dim=config.region_embedding_dim,
            hidden_dim=self.base_model.config.hidden_size
        )
        
        # Adapter layers
        self.geographic_adapter = nn.ModuleDict({
            'down_proj': nn.Linear(self.base_model.config.hidden_size, config.geographic_hidden_dim),
            'up_proj': nn.Linear(config.geographic_hidden_dim, self.base_model.config.hidden_size),
            'activation': nn.ReLU(),
            'dropout': nn.Dropout(0.1)
        })
        
        # Geographic fusion layer
        self.geographic_fusion = nn.Sequential(
            nn.Linear(
                self.base_model.config.hidden_size + config.region_embedding_dim,
                self.base_model.config.hidden_size
            ),
            nn.LayerNorm(self.base_model.config.hidden_size),
            nn.Dropout(0.1)
        )
        
        # Geographic classifier for auxiliary task
        self.geographic_classifier = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, config.geographic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.geographic_hidden_dim, config.num_regions)
        )
        
        # Language modeling head
        self.lm_head = nn.Linear(
            self.base_model.config.hidden_size,
            self.tokenizer.vocab_size,
            bias=False
        )
        
        # Apply LoRA to base model
        self.apply_lora()
    
    def apply_lora(self):
        """Apply LoRA (Low-Rank Adaptation) to the base model."""
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.base_model = get_peft_model(self.base_model, lora_config)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                region_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the geographic adapter model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            region_ids: Region IDs of shape (batch_size,)
            labels: Optional labels for language modeling loss
            
        Returns:
            Dictionary containing logits and losses
        """
        # Get base model outputs
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Fix: use the last element of hidden_states
        hidden_states = base_outputs.hidden_states[-1]
        
        # Apply geographic adaptation
        adapted_states = self.apply_geographic_adaptation(hidden_states, region_ids)
        
        # Generate language modeling logits
        lm_logits = self.lm_head(adapted_states)
        
        # Geographic classification for auxiliary task
        pooled_states = adapted_states.mean(dim=1)  # Pool over sequence dimension
        geo_logits = self.geographic_classifier(pooled_states)
        
        # Calculate losses
        total_loss = None
        lm_loss = None
        geo_loss = None
        
        if labels is not None:
            # Language modeling loss
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            # Geographic classification loss
            if self.config.use_geographic_loss:
                geo_loss = F.cross_entropy(geo_logits, region_ids)
                total_loss = lm_loss + self.config.geographic_loss_weight * geo_loss
            else:
                total_loss = lm_loss
        
        return {
            'logits': lm_logits,
            'geo_logits': geo_logits,
            'loss': total_loss,
            'lm_loss': lm_loss,
            'geo_loss': geo_loss,
            'hidden_states': adapted_states
        }
    
    def apply_geographic_adaptation(self, hidden_states: torch.Tensor, 
                                  region_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply geographic adaptation to hidden states.
        
        Args:
            hidden_states: Hidden states from base model
            region_ids: Region IDs for geographic context
            
        Returns:
            Adapted hidden states
        """
        # Get geographic embeddings
        geo_embeddings = self.geographic_embedding(region_ids, hidden_states)  # [batch, seq_len, region_embedding_dim]
        # Adapter transformation
        adapted = self.geographic_adapter['down_proj'](hidden_states)
        adapted = self.geographic_adapter['activation'](adapted)
        adapted = self.geographic_adapter['dropout'](adapted)
        adapted = self.geographic_adapter['up_proj'](adapted)
        # Residual connection
        adapted = adapted + hidden_states
        # Use geo_embeddings directly (no pooling)
        geo_context = geo_embeddings  # [batch, seq_len, region_embedding_dim]
        # Debug prints (remove/comment after confirming fix)
        # print("adapted shape:", adapted.shape)
        # print("geo_context shape:", geo_context.shape)
        # Concatenate and fuse
        fused_input = torch.cat([adapted, geo_context], dim=-1)  # [batch, seq_len, hidden_dim + region_embedding_dim]
        # print("fused_input shape:", fused_input.shape)
        final_states = self.geographic_fusion(fused_input)
        return final_states
    
    def generate_with_region(self, prompt: str, region_id: int, 
                           max_length: int = 100, **kwargs) -> str:
        """
        Generate text with geographic context.
        
        Args:
            prompt: Input prompt
            region_id: Region ID for geographic context
            max_length: Maximum generation length
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        
        # Prepare region tensor
        region_tensor = torch.tensor([region_id], dtype=torch.long)
        
        # Generate with geographic context
        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                region_ids=region_tensor
            )
            
            # Use logits for generation (simplified)
            logits = outputs['logits']
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
            
            # For full generation, you'd implement beam search or sampling
            # This is a simplified version
            generated_ids = torch.cat([inputs['input_ids'], next_token_id.unsqueeze(0)], dim=1)
            
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# Example usage and testing
def test_geographic_adapter():
    """Test the geographic adapter model."""
    
    # Create configuration
    config = GeographicAdapterConfig(
        base_model_name="gpt2",
        num_regions=5,
        region_embedding_dim=64
    )
    
    # Initialize model
    model = GeographicAdapter(config)
    
    # Create sample inputs
    tokenizer = model.tokenizer
    sample_text = "Hello, how are you doing today?"
    
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_length
    )
    
    # Sample region IDs
    region_ids = torch.tensor([0, 1, 2])[:inputs['input_ids'].shape[0]]
    
    # Forward pass
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        region_ids=region_ids,
        labels=inputs['input_ids']  # For loss calculation
    )
    
    print("Model outputs:")
    print(f"LM Logits shape: {outputs['logits'].shape}")
    print(f"Geographic Logits shape: {outputs['geo_logits'].shape}")
    print(f"Total Loss: {outputs['loss']}")
    print(f"LM Loss: {outputs['lm_loss']}")
    print(f"Geo Loss: {outputs['geo_loss']}")
    
    # Test generation
    generated = model.generate_with_region(
        prompt="What's the weather like",
        region_id=0
    )
    print(f"Generated text: {generated}")

if __name__ == "__main__":
    test_geographic_adapter()
