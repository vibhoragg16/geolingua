import torch
import torch.nn as nn
#from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import logging
from typing import Dict, List, Optional, Tuple, Union
import os

from config.model_config import (
    MODEL_NAME, MAX_LENGTH, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    DEVICE, FP16, REGIONS
)

logger = logging.getLogger(__name__)

class GeoLinguaModel(nn.Module):
    """
    Base model class for GeoLingua project.
    Handles model initialization, tokenization, and basic inference.
    """
    
    def __init__(self, model_name: str = MODEL_NAME, load_in_8bit: bool = False, 
                 regions: Optional[List[str]] = None, lora_config: Optional[Dict[str, Union[str, int, float, List[str]]]] = None):
        super().__init__()
        self.model_name = model_name
        self.load_in_8bit = load_in_8bit
        self.regions = regions if regions is not None else ['us_south', 'uk', 'australia', 'india', 'nigeria']
        self.lora_config = lora_config if lora_config is not None else {}
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.config = None
        
        self._load_model()
        self._setup_tokenizer()
        
        # Setup LoRA if config provided
        if self.lora_config:
            target_modules = self.lora_config.get('target_modules', None)
            if isinstance(target_modules, list) or target_modules is None:
                self.setup_lora(target_modules=target_modules)
        
    def _load_model(self):
        """Load the base model and configuration."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load configuration
            self.config = AutoConfig.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=self.config,
                torch_dtype=torch.float16 if FP16 else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                load_in_8bit=self.load_in_8bit,
                trust_remote_code=True
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _setup_tokenizer(self):
        """Setup tokenizer with special tokens."""
        try:
            logger.info("Setting up tokenizer...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Add geographic region tokens
            special_tokens = []
            for region_code, region_name in REGIONS.items():
                special_tokens.extend([
                    f"<{region_code}>",
                    f"</{region_code}>",
                    f"[{region_name.upper()}]"
                ])
            
            if special_tokens:
                self.tokenizer.add_tokens(special_tokens)
                self.model.resize_token_embeddings(len(self.tokenizer))  # Ensure model can handle new tokens
                # Initialize new token embeddings to mean of old embeddings
                with torch.no_grad():
                    input_embeddings = self.model.get_input_embeddings().weight
                    num_new = len(special_tokens)
                    if num_new > 0:
                        mean_embed = input_embeddings[:-num_new].mean(dim=0)
                        input_embeddings[-num_new:] = mean_embed
            
            logger.info("Tokenizer setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up tokenizer: {str(e)}")
            raise
    
    def setup_lora(self, target_modules: Optional[List[str]] = None) -> None:
        """
        Setup LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
        
        Args:
            target_modules: List of module names to apply LoRA to
        """
        try:
            logger.info("Setting up LoRA configuration...")
            
            # Auto-detect target modules based on model architecture
            if target_modules is None:
                target_modules = self._get_target_modules()
            
            logger.info(f"Using target modules: {target_modules}")
            
            # Create LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=target_modules,
                bias="none"
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
            logger.info("LoRA setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up LoRA: {str(e)}")
            # Try with a more generic approach
            logger.info("Trying with generic target modules...")
            self._setup_lora_generic()
    
    def _get_target_modules(self) -> List[str]:
        """Auto-detect target modules for LoRA."""
        
        # Get all module names
        module_names = [name for name, _ in self.model.named_modules()]
        logger.info(f"Available modules: {module_names[:10]}...")  # Log first 10
        
        # Common patterns for different model types
        patterns = {
            'llama': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'gpt': ['c_attn', 'c_proj', 'c_fc'],
            'bert': ['query', 'key', 'value', 'dense'],
            't5': ['q', 'k', 'v', 'o'],
            'falcon': ['query_key_value', 'dense'],
            'mpt': ['Wqkv', 'out_proj'],
            'gpt2': ['c_attn', 'c_proj'],
            'bloom': ['query_key_value', 'dense'],
            'opt': ['q_proj', 'k_proj', 'v_proj', 'out_proj']
        }
        
        # Try to match model type
        model_name_lower = self.model_name.lower()
        for model_type, modules in patterns.items():
            if model_type in model_name_lower:
                # Check if these modules exist
                available_modules = [m for m in modules if any(m in name for name in module_names)]
                if available_modules:
                    logger.info(f"Detected {model_type} pattern, using: {available_modules}")
                    return available_modules
        
        # Fallback: look for common attention patterns
        attention_patterns = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'query', 'key', 'value', 'dense',
            'c_attn', 'c_proj',
            'Wqkv', 'out_proj'
        ]
        
        found_modules = []
        for pattern in attention_patterns:
            if any(pattern in name for name in module_names):
                found_modules.append(pattern)
        
        if found_modules:
            logger.info(f"Found attention modules: {found_modules}")
            return found_modules
        
        # Last resort: return empty list to avoid error
        logger.warning("No suitable target modules found, using empty list")
        return []
    
    def _setup_lora_generic(self):
        """Setup LoRA with generic configuration."""
        try:
            # Try with a very basic configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=[],  # Empty list to avoid errors
                bias="none"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRA setup with generic config complete")
            
        except Exception as e:
            logger.error(f"Failed to setup LoRA: {e}")
            logger.info("Continuing without LoRA...")
            # Continue without LoRA
    
    def tokenize_text(self, text: str, max_length: int = MAX_LENGTH) -> Dict[str, torch.Tensor]:
        """
        Tokenize input text.
        
        Args:
            text: Input text to tokenize
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing tokenized inputs
        """
        return self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1
    ) -> str:
        """
        Generate text using the model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            repetition_penalty: Repetition penalty
            
        Returns:
            Generated text
        """
        try:
            # Tokenize input
            inputs = self.tokenize_text(prompt)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Remove the original prompt from generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return ""
    
    def save_model(self, save_path: str) -> None:
        """
        Save the model and tokenizer.
        
        Args:
            save_path: Path to save the model
        """
        try:
            logger.info(f"Saving model to {save_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Save model
            self.model.save_pretrained(save_path)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(save_path)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, load_path: str) -> None:
        """
        Load a saved model and tokenizer.
        
        Args:
            load_path: Path to load the model from
        """
        try:
            logger.info(f"Loading model from {load_path}")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch.float16 if FP16 else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                load_path,
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                region_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Minimal forward pass for debugging: just call the HuggingFace model and return its outputs.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        return outputs
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "model_type": self.config.model_type if self.config else "unknown",
            "vocab_size": len(self.tokenizer) if self.tokenizer else 0,
            "max_length": MAX_LENGTH,
            "device": str(self.device),
            "fp16": FP16,
            "num_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0
        }


class GeoLinguaDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for GeoLingua training data.
    """
    
    def __init__(self, texts: List[str], labels: List[str], tokenizer, max_length: int = MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize label
        label_encoding = self.tokenizer(
            label,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_encoding['input_ids'].flatten()
        }


if __name__ == "__main__":
    # Test the base model
    try:
        model = GeoLinguaModel()
        print("Base model loaded successfully!")
        
        # Print model info
        info = model.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Test generation
        prompt = "Hello, how are you doing today?"
        generated = model.generate_text(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")
        
        # Dummy forward pass test for NaNs
        import torch
        ids = torch.tensor([[1, 2, 3, 4, 5] + [model.tokenizer.eos_token_id] * (512-5)]).to(model.device)
        mask = torch.ones_like(ids)
        labels = ids.clone()
        region_ids = torch.tensor([0]).to(model.device)
        out = model(input_ids=ids, attention_mask=mask, region_ids=region_ids, labels=labels)
        print("\nDummy forward pass NaN check:")
        print({k: v if not isinstance(v, torch.Tensor) else torch.isnan(v).any().item() for k, v in out.items()})
        
    except Exception as e:
        print(f"Error: {str(e)}")
